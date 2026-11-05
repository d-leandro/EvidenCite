from fastapi import FastAPI

app = FastAPI(
    title="EvidenCite API",
    description="Academic Citation Function Classification",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}
