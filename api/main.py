from api.routes import router

from fastapi import FastAPI

app = FastAPI(
    title="EvidenCite API",
    description="Academic Citation Function Classification",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}
app.include_router(router)
