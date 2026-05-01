from rest_framework.response import Response


def success(data=None, message="success", status=200):
    return Response({"code": 0, "message": message, "data": data}, status=status)


def error(code, message, status=400):
    return Response({"code": code, "message": message, "data": None}, status=status)
