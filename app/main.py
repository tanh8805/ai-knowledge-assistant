from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.api.routes import documents, health
from app.core.config import MissingAPIKeyError, get_settings
from app.core.logging import setup_logging

settings = get_settings()
setup_logging(settings.app.log_level)

app = FastAPI(title=settings.app.name)
app.include_router(health.router)
app.include_router(documents.router)


@app.exception_handler(MissingAPIKeyError)
def missing_api_key(_: Request, error: MissingAPIKeyError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": f"Provider not configured: {error}"},
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"name": app.title, "docs": "/docs"}
