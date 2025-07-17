import sys
sys.path.append('/home/ubuntu/pizza_violation_detection')

import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import config
from events import startup_event, shutdown_event
from routers import router as api_router
from websocket import router as websocket_router

# Configure root logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="Pizza Violation Detection API",
    description="Real-time pizza store scooper violation detection system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include consolidated router
app.include_router(api_router)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Startup & shutdown
app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info"
    )