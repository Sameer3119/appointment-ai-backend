from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.database import get_db
from app.schemas.health import HealthResponse
from app.config import settings

# APIRouter instance for health endpoints
router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Application Health Check",
    description=(
        "Returns the operational status of the Voice AI Doctor Appointment Assistant backend "
        "and checks connectivity to the SQLite database."
    ),
)
def health_check(db: Session = Depends(get_db)):
    """
    Health Check Endpoint: GET /health
    
    1. Tests SQLite database connectivity by executing a simple SELECT 1 query.
    2. Returns application status, database connectivity status, and environment.
    """
    db_status = "connected"
    try:
        # Lightweight query to verify database is responsive
        db.execute(text("SELECT 1"))
    except Exception as exc:
        db_status = f"disconnected: {str(exc)}"

    return HealthResponse(
        status="ok" if db_status == "connected" else "degraded",
        message="Voice AI Doctor Appointment Assistant Backend is running",
        database=db_status,
        environment=settings.APP_ENV,
    )
