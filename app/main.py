from contextlib import asynccontextmanager
import aiohttp
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import engine, Base, SessionLocal

# Import all SQLAlchemy models so that Base.metadata has knowledge of all tables before create_all is invoked
import app.models  # noqa: F401
from app.api.v1.endpoints import health
from app.api.v1.endpoints.voice import close_voice_resources
from app.api.v1.router import api_router
from app.seed import seed_doctors


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Context Manager
    
    Handles application startup and shutdown events:
    - Startup: Automatically creates SQLite database tables for Doctor, Patient, Appointment if they do not exist.
               Safely seeds initial sample doctors if the database is newly initialized.
               Creates a shared aiohttp.ClientSession for outbound HTTP services.
    - Shutdown: Cancels active voice pipelines, closes WebRTC handler, and closes shared aiohttp session.
    """
    # Create all database tables on application startup
    Base.metadata.create_all(bind=engine)

    # Safely seed initial sample doctors if none exist
    with SessionLocal() as db:
        seed_doctors(db)

    app.state.aiohttp_session = aiohttp.ClientSession()

    try:
        yield
    finally:
        await close_voice_resources()
        await app.state.aiohttp_session.close()


# Initialize FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Backend API for MediVoice AI - Voice-enabled Doctor Appointment Assistant. "
        "Provides doctor scheduling, patient auto-registration, conflict prevention, "
        "and appointment management."
    ),
    version="1.1.0",
    docs_url="/docs",          # Interactive Swagger UI documentation
    redoc_url="/redoc",        # Alternative ReDoc documentation
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS (Cross-Origin Resource Sharing) Configuration
# ---------------------------------------------------------------------------
# This enables the existing React frontend running on http://localhost:3000
# to make asynchronous HTTP API requests to this FastAPI backend without browser CORS errors.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Router Registration
# ---------------------------------------------------------------------------
# 1. Mount health router directly at root level so GET /health is immediately available
app.include_router(health.router)

# 2. Mount versioned API router under /api/v1 for modular scaling
app.include_router(api_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
def root():
    """
    Root API endpoint providing friendly welcome info and quick links.
    """
    return {
        "message": f"Welcome to {settings.PROJECT_NAME} API",
        "documentation": "/docs",
        "health_check": "/health",
        "version": "1.1.0",
        "endpoints": {
            "doctors": "/api/v1/doctors",
            "appointments": "/api/v1/appointments",
        },
    }
