from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routers.predictions import router
from api.routers.pdf import router as pdf_router
from api.routers.metrics import router as metrics_router

APP_DIR = Path(__file__).resolve().parent.parent / "app"

app = FastAPI(
    title="EvidenCite API",
    description="Academic Citation Function Classification",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(router)
app.include_router(pdf_router)
app.include_router(metrics_router)

app.mount(
    "/static",
    StaticFiles(directory=APP_DIR / "static"),
    name="static",
)


@app.get("/", include_in_schema=False)
def homepage():
    return FileResponse(APP_DIR / "index.html")
