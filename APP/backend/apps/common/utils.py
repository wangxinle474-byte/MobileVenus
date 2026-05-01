import uuid
from datetime import datetime


def generate_request_id() -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"req_{ts}_{short}"
