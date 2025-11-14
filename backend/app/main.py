import os
import logging
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager
import uvicorn

from app.api.voice_management import router as voice_router
from app.api.emotion_engine import router as emotion_router
from app.api.celebrity_voices import router as celebrity_router
from app.core.config import settings
from app.core.database import get_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("Starting Voice AI Suite...")
    yield
    logger.info("Shutting down Voice AI Suite...")

app = FastAPI(
    title="Voice AI Suite",
    description="Enterprise-grade voice cloning, TTS, and emotion processing services",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(voice_router, prefix="/api/voice", tags=["Voice Management"])
app.include_router(emotion_router, prefix="/api/emotion", tags=["Emotion Engine"])
app.include_router(celebrity_router, prefix="/api/marketplace", tags=["Celebrity Voices"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Voice AI Suite",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "voice_management": "/api/voice",
            "emotion_engine": "/api/emotion",
            "celebrity_marketplace": "/api/marketplace"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "voice-ai-suite",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )