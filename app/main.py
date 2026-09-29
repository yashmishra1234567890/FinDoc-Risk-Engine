import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from dotenv import load_dotenv

load_dotenv()

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

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "https://yashmishra1234567890-findoc-risk-engine-frontendapp-0k0ooh.streamlit.app",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "FinDoc AI API is running. Check /docs for API documentation."}

app.include_router(health_router)
app.include_router(upload_router)
app.include_router(query_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
