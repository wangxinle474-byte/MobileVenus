import { describe, it, expect, vi, beforeEach } from "vitest";
import type { EnhanceResult } from "@/types/api";

// ─── Mock uni.uploadFile ──────────────────────────────────────────────────────

const MOCK_RESULT: EnhanceResult = {
  request_id: "req-abc",
  original_image_url: "http://server/orig.jpg",
  enhanced_image_url: "http://server/enh.jpg",
  parameters: {
    ev_compensation: 0.3,
    contrast: 1.0,
    highlights: 0.0,
    shadows: 0.0,
    brightness: 0.0,
    vibrance: 0.0,
    saturation: 1.0,
    white_balance: 5500,
  },
  parameter_confidence: {
    ev_compensation: "high",
    contrast: "medium",
    highlights: "reference_only",
    shadows: "reference_only",
    brightness: "reference_only",
    vibrance: "reference_only",
    saturation: "high",
    white_balance: "medium",
  },
  meta_info: {
    model_name: "DistillV4",
    backbone: "MobileViT-S",
    pipeline: "SemanticDistill",
    inference_time_ms: 42,
    image_width: 1920,
    image_height: 1080,
    psnr_reference: "27.6 dB",
  },
  aesthetic_status: "done",
  aesthetic_before: { overall: 6.5, dimensions: { composition: 6.0, lighting: 7.0, color: 6.5, clarity: 6.8, subject: 6.2 } },
  aesthetic_after: { overall: 7.8, dimensions: { composition: 7.5, lighting: 8.0, color: 7.8, clarity: 7.6, subject: 7.9 } },
};

function successUpload(data: unknown) {
  vi.mocked(uni.uploadFile).mockImplementation(({ success }: any) => {
    success?.({ statusCode: 200, data: JSON.stringify({ code: 0, message: "ok", data }) });
    return {} as UniApp.UploadTask;
  });
}

function failUpload(code: number, message: string) {
  vi.mocked(uni.uploadFile).mockImplementation(({ success }: any) => {
    success?.({ statusCode: 200, data: JSON.stringify({ code, message, data: null }) });
    return {} as UniApp.UploadTask;
  });
}

// ─── inference.ts ─────────────────────────────────────────────────────────────

describe("enhance()", () => {
  beforeEach(() => {
    vi.mocked(uni.uploadFile).mockClear();
  });

  it("resolves with EnhanceResult on success", async () => {
    successUpload(MOCK_RESULT);
    const { enhance } = await import("@/services/inference");
    const result = await enhance("/tmp/img.jpg", "distill_v4", "mv_client");
    expect(result.request_id).toBe("req-abc");
    expect(result.meta_info.inference_time_ms).toBe(42);
  });

  it("rejects when server returns non-zero code", async () => {
    failUpload(2001, "模型推理失败");
    const { enhance } = await import("@/services/inference");
    await expect(enhance("/tmp/img.jpg", "baseline", "mv_client")).rejects.toThrow(
      "模型推理失败"
    );
  });

  it("passes correct model_version to formData", async () => {
    successUpload(MOCK_RESULT);
    const { enhance } = await import("@/services/inference");
    await enhance("/tmp/img.jpg", "distill_v2", "mv_client");
    expect(uni.uploadFile).toHaveBeenCalledWith(
      expect.objectContaining({
        formData: expect.objectContaining({ model_version: "distill_v2" }),
      })
    );
  });
});

// ─── api.ts – get() ───────────────────────────────────────────────────────────

describe("get()", () => {
  beforeEach(() => {
    vi.mocked(uni.request).mockClear();
  });

  it("calls uni.request with correct URL", async () => {
    vi.mocked(uni.request).mockImplementation(({ success }: any) => {
      success?.({ data: { code: 0, message: "ok", data: { models_loaded: [] } } });
      return {} as UniApp.RequestTask;
    });
    const { get } = await import("@/services/api");
    const resp = await get("/health/");
    expect(resp.code).toBe(0);
    expect(uni.request).toHaveBeenCalledWith(
      expect.objectContaining({ url: expect.stringContaining("/health/") })
    );
  });
});
