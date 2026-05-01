from rest_framework.views import APIView
from apps.common.responses import success
from .models import MiniProgramUser


class RegisterClientView(APIView):
    def post(self, request):
        client_id = request.data.get("client_id", "").strip()
        if not client_id:
            from apps.common.responses import error
            return error(1000, "client_id is required")
        user, _ = MiniProgramUser.objects.get_or_create(client_id=client_id)
        return success({"client_id": user.client_id})
