from schemas import ErrorResponse


class ChurnServiceError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400, details: dict | None = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


def error_detail(code: str, message: str, details: dict | None = None) -> dict:
    return ErrorResponse(code=code, message=message, details=details or {}).model_dump()
