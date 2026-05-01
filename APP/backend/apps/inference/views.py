import threading
from pathlib import Path
from datetime import datetime, timezone

from django.http import HttpResponse

from django.conf import settings
from django.db import transaction
from rest_framework.views import APIView

from apps.common.responses import success, error
from apps.common.utils import generate_request_id
from apps.users.models import MiniProgramUser

from .models import InferenceRecord
from .serializers import EnhanceImageRequestSerializer
from .services import preprocess, predictor, isp, storage, aesthetic
from .services.exceptions import InferenceServiceException


def _run_aesthetic_async(record_id: int, original_path: str, enhanced_path: str) -> None:
    """后台线程：调用 Venus 评分并写回数据库。"""
    import django
    django.db.close_old_connections()
    try:
        before = aesthetic.score(original_path)
        after = aesthetic.score(enhanced_path)
        status = "done" if (before is not None and after is not None) else "unavailable"
        InferenceRecord.objects.filter(pk=record_id).update(
            aesthetic_status=status,
            aesthetic_before=before,
            aesthetic_after=after,
        )
    except Exception:
        InferenceRecord.objects.filter(pk=record_id).update(aesthetic_status="failed")

# 专家共识置信度（静态数据，无需每次推理重新计算）
_PARAM_CONFIDENCE = {
    "ev_compensation": "medium",
    "white_balance": "reference_only",
    "contrast": "medium",
    "brightness": "medium",
    "shadows": "high",
    "highlights": "medium",
    "saturation": "high",
    "vibrance": "high",
}

_MODEL_META = {
    "baseline":   {"model_name": "MobileVenus Baseline",    "psnr_reference": "32.05 dB"},
    "distill_v2": {"model_name": "MobileVenus Distill v2",  "psnr_reference": "33.15 dB"},
    "distill_v4": {"model_name": "MobileVenus Distill v4",  "psnr_reference": "33.11 dB"},
}


class EnhanceImageAPIView(APIView):
    def post(self, request):
        serializer = EnhanceImageRequestSerializer(data=request.data)
        if not serializer.is_valid():
            first_err = next(iter(serializer.errors.values()))
            msg = first_err[0] if first_err else "invalid request"
            code_attr = getattr(first_err[0], "code", None) if first_err else None
            err_code = int(code_attr) if code_attr and str(code_attr).isdigit() else 1000
            return error(err_code, str(msg))

        file_obj = serializer.validated_data["file"]
        model_version = serializer.validated_data["model_version"]
        client_id = serializer.validated_data.get("client_id", "")

        user = None
        if client_id:
            user, _ = MiniProgramUser.objects.get_or_create(client_id=client_id)

        request_id = generate_request_id()

        with transaction.atomic():
            record = InferenceRecord.objects.create(
                request_id=request_id,
                user=user,
                original_image=file_obj,
                model_version=model_version,
                status="pending",
            )

        original_path = record.original_image.path

        try:
            # 预处理
            tensor = preprocess.run(original_path)

            # 推理
            params, elapsed_ms = predictor.run(tensor, model_version)

            # ISP 渲染
            enhanced_img = isp.render(original_path, params)

            # 保存增强图
            enhanced_rel = storage.save_enhanced(enhanced_img, record.original_image.name)

            meta = {
                **_MODEL_META.get(model_version, {}),
                "backbone": "MobileViT-Small (1.93M params)",
                "pipeline": "SemanticDistill + Gamma-aware ISP",
                "inference_time_ms": elapsed_ms,
                "image_width": enhanced_img.width,
                "image_height": enhanced_img.height,
            }

            with transaction.atomic():
                record.enhanced_image = enhanced_rel
                record.parameters = params
                record.meta_info = meta
                record.status = "success"
                record.aesthetic_status = "running"
                record.finished_at = datetime.now(tz=timezone.utc)
                record.save()
                try:
                    with open(original_path, "rb") as f:
                        record.original_image_data = f.read()
                    enhanced_abs_tmp = str(Path(settings.MEDIA_ROOT) / enhanced_rel)
                    with open(enhanced_abs_tmp, "rb") as f:
                        record.enhanced_image_data = f.read()
                    record.save(update_fields=["original_image_data", "enhanced_image_data"])
                except Exception:
                    pass

            # 异步启动 Venus 评分（不阻塞响应）
            enhanced_abs = str(Path(settings.MEDIA_ROOT) / enhanced_rel)
            t = threading.Thread(
                target=_run_aesthetic_async,
                args=(record.pk, original_path, enhanced_abs),
                daemon=True,
            )
            t.start()

        except InferenceServiceException as exc:
            with transaction.atomic():
                record.status = "failed"
                record.error_code = exc.code
                record.error_message = exc.message
                record.finished_at = datetime.now(tz=timezone.utc)
                record.save()
            return error(int(exc.code), exc.message, status=422)

        return success({
            "request_id": request_id,
            "original_image_url": storage.media_url(record.original_image.name),
            "enhanced_image_url": storage.media_url(enhanced_rel),
            "parameters": params,
            "parameter_confidence": _PARAM_CONFIDENCE,
            "meta_info": meta,
            "aesthetic_status": "running",
            "aesthetic_before": None,
            "aesthetic_after": None,
        })


class LatestResultAPIView(APIView):
    def get(self, request):
        client_id = request.query_params.get("client_id", "").strip()
        if not client_id:
            return error(1000, "client_id is required")

        record = (
            InferenceRecord.objects
            .filter(user__client_id=client_id, status="success")
            .first()
        )
        if not record:
            return error(4040, "no result found", status=404)

        return success({
            "request_id": record.request_id,
            "original_image_url": storage.media_url(record.original_image.name),
            "enhanced_image_url": storage.media_url(record.enhanced_image.name),
            "parameters": record.parameters,
            "parameter_confidence": _PARAM_CONFIDENCE,
            "meta_info": record.meta_info,
        })


class AestheticScoreAPIView(APIView):
    """轮询端点：返回指定 request_id 的 Venus 美学评分状态。"""

    def get(self, request, request_id: str):
        try:
            record = InferenceRecord.objects.get(request_id=request_id)
        except InferenceRecord.DoesNotExist:
            return error(4040, "record not found", status=404)

        return success({
            "request_id": request_id,
            "aesthetic_status": record.aesthetic_status,
            "aesthetic_before": record.aesthetic_before,
            "aesthetic_after": record.aesthetic_after,
        })


class PhotoListAPIView(APIView):
    """GET /api/v1/photos/  —— 全部成功的推理结果（按时间倒序）。

    可选 query param:
      - client_id : 只返回该用户的照片
      - limit     : 每页条数（默认 20，最大 100）
      - offset    : 偏移量（默认 0）
    """

    def get(self, request):
        client_id = request.query_params.get("client_id", "").strip()
        try:
            limit = min(int(request.query_params.get("limit", 20)), 100)
            offset = max(int(request.query_params.get("offset", 0)), 0)
        except ValueError:
            return error(1000, "limit and offset must be integers")

        qs = InferenceRecord.objects.filter(status="success")
        if client_id:
            qs = qs.filter(user__client_id=client_id)

        total = qs.count()
        records = qs[offset: offset + limit]

        return success({
            "total": total,
            "limit": limit,
            "offset": offset,
            "results": [_serialize_record(r) for r in records],
        })


class PhotoDetailAPIView(APIView):
    """GET /api/v1/photos/<pk>/  —— 单条推理结果详情。"""

    def get(self, request, pk: int):
        try:
            record = InferenceRecord.objects.get(pk=pk, status="success")
        except InferenceRecord.DoesNotExist:
            return error(4040, "record not found", status=404)
        return success(_serialize_record(record))


def _serialize_record(rec: InferenceRecord) -> dict:
    base = getattr(settings, "MEDIA_BASE_URL", "").rstrip("/")
    return {
        "id": rec.pk,
        "request_id": rec.request_id,
        "model_version": rec.model_version,
        "saved_at": rec.finished_at.isoformat() if rec.finished_at else rec.created_at.isoformat(),
        "original_image_url": f"{base}/api/v1/photos/{rec.pk}/original/",
        "enhanced_image_url": f"{base}/api/v1/photos/{rec.pk}/enhanced/",
        "parameters": rec.parameters,
        "meta_info": rec.meta_info,
        "aesthetic_status": rec.aesthetic_status,
        "aesthetic_before": rec.aesthetic_before,
        "aesthetic_after": rec.aesthetic_after,
    }


class PhotoImageAPIView(APIView):
    """GET /api/v1/photos/<pk>/original/  或  /api/v1/photos/<pk>/enhanced/
    直接从数据库返回图片二进制，无需依赖文件系统。
    """

    def get(self, request, pk: int, image_type: str):
        try:
            record = InferenceRecord.objects.get(pk=pk)
        except InferenceRecord.DoesNotExist:
            return error(4040, "record not found", status=404)

        if image_type == "original":
            data = record.original_image_data
        elif image_type == "enhanced":
            data = record.enhanced_image_data
        else:
            return error(1000, "image_type must be 'original' or 'enhanced'")

        if not data:
            return error(4040, "image data not available", status=404)

        return HttpResponse(bytes(data), content_type="image/jpeg")


class HealthAPIView(APIView):
    def get(self, request):
        loaded = predictor.ModelRegistry.loaded_versions()
        return success({
            "models_loaded": loaded,
            "default_version": "distill_v4",
        })
