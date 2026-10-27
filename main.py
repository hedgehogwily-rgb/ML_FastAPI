from fastapi import FastAPI

from api.errors import register_exception_handlers
from api.routes import router
from core.config import setup_logging
from ml.pipeline import load_saved_model

setup_logging()

app = FastAPI(
    title="Churn service",
    description="Сервис предсказания оттока клиентов.",
)
app.include_router(router)
register_exception_handlers(app)
load_saved_model()
