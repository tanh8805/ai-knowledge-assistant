from fastapi import FastAPI

from app.api.routes import health

app = FastAPI(title="ai-knowledge-assistant")
app.include_router(health.router)


@app.get("/")
def root() -> dict[str, str]:
    return {"name": app.title, "docs": "/docs"}
