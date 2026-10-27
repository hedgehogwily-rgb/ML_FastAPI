import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from core.errors import ChurnServiceError, error_detail

logger = logging.getLogger("churn")


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        if isinstance(exc.detail, dict) and "code" in exc.detail:
            content = exc.detail
        else:
            content = error_detail("http_error", str(exc.detail))
        logger.error(
            "request failed status=%s code=%s message=%s",
            exc.status_code,
            content["code"],
            content["message"],
        )
        return JSONResponse(status_code=exc.status_code, content=content)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        errors = [
            {
                "field": ".".join(str(part) for part in err["loc"]),
                "message": err["msg"],
                "type": err["type"],
            }
            for err in exc.errors()
        ]
        specific = [item for item in errors if item["type"] != "list_type"]
        if specific:
            errors = specific
        feature_errors = {"missing", "extra_forbidden"}
        code = "invalid_feature_count" if any(item["type"] in feature_errors for item in errors) else "invalid_type"
        logger.error("request failed status=422 code=%s message=Request validation failed", code)
        return JSONResponse(
            status_code=422,
            content=error_detail(code, "Request validation failed", {"errors": errors}),
        )

    @app.exception_handler(ChurnServiceError)
    async def churn_service_error_handler(request: Request, exc: ChurnServiceError):
        logger.error(
            "request failed status=%s code=%s message=%s",
            exc.status_code,
            exc.code,
            exc.message,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_detail(exc.code, exc.message, exc.details),
        )
