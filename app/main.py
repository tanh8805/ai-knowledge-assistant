from fastapi import FastAPI

from app.api.routes import health
from app.core.config import get_settings
from app.core.logging import setup_logging

settings = get_settings()
setup_logging(settings.app.log_level)

app = FastAPI(title=settings.app.name)
app.include_router(health.router)


@app.get("/")
def root() -> dict[str, str]:
    return {"name": app.title, "docs": "/docs"}
