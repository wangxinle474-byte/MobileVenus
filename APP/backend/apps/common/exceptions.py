import traceback
from rest_framework.views import exception_handler
from rest_framework.response import Response
from rest_framework import status


def custom_exception_handler(exc, context):
    response = exception_handler(exc, context)

    if response is not None:
        if response.status_code == 400:
            return Response(
                {"code": 1000, "message": str(response.data), "data": None},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if response.status_code == 404:
            return Response(
                {"code": 4040, "message": "resource not found", "data": None},
                status=status.HTTP_404_NOT_FOUND,
            )
        if response.status_code == 405:
            return Response(
                {"code": 4050, "message": "method not allowed", "data": None},
                status=status.HTTP_405_METHOD_NOT_ALLOWED,
            )

    tb = traceback.format_exc()
    print("[5000] Unhandled exception:", tb)
    from django.conf import settings
    detail = tb if settings.DEBUG else "internal server error"
    return Response(
        {"code": 5000, "message": detail, "data": None},
        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
