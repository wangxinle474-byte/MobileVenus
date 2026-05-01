import pytest
from django.test import RequestFactory
from rest_framework.test import APIRequestFactory

from apps.common.responses import success, error
from apps.common.utils import generate_request_id


class TestResponses:
    def test_success_default(self):
        resp = success({"key": "value"})
        assert resp.status_code == 200
        assert resp.data["code"] == 0
        assert resp.data["message"] == "success"
        assert resp.data["data"] == {"key": "value"}

    def test_success_no_data(self):
        resp = success()
        assert resp.data["data"] is None

    def test_success_custom_message(self):
        resp = success(message="ok")
        assert resp.data["message"] == "ok"

    def test_error_default(self):
        resp = error(1001, "file missing")
        assert resp.status_code == 400
        assert resp.data["code"] == 1001
        assert resp.data["message"] == "file missing"
        assert resp.data["data"] is None

    def test_error_custom_status(self):
        resp = error(4040, "not found", status=404)
        assert resp.status_code == 404


class TestUtils:
    def test_generate_request_id_format(self):
        rid = generate_request_id()
        assert rid.startswith("req_")
        parts = rid.split("_")
        assert len(parts) == 3

    def test_generate_request_id_unique(self):
        ids = {generate_request_id() for _ in range(50)}
        assert len(ids) == 50
