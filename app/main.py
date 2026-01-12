from fastapi import FastAPI
import logging

# Configure logging to show INFO level logs in Render console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

from app.api.routes.health import router as health_router
from app.api.routes.upload import router as upload_router
from app.api.routes.query import router as query_router

app = FastAPI(
    title="FinDoc AI",
    description="AI-powered financial risk & compliance analyzer",
    version="1.0"
)

@app.get("/")
def read_root():
    return {"message": "FinDoc AI API is running. Check /docs for API documentation."}

app.include_router(health_router)
app.include_router(upload_router)
app.include_router(query_router)

if __name__ == "__main__":
    # How to run locally:
    # python -m app.main
    import uvicorn
    # Use 8001 to avoid conflicts
    uvicorn.run("app.main:app", host="127.0.0.1", port=8001, reload=False)
