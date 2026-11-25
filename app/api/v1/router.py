from fastapi import APIRouter
from app.api.v1.endpoints import health, doctors, appointments, voice

# Create the top-level API v1 router
api_router = APIRouter()

# Register modular endpoint routers
api_router.include_router(health.router)
api_router.include_router(doctors.router)
api_router.include_router(appointments.router)
api_router.include_router(voice.router)
