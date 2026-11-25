from datetime import date as date_type, datetime, time, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.doctor import Doctor
from app.models.appointment import Appointment
from app.schemas.doctor import (
    DoctorResponse,
    DoctorAvailabilityResponse,
    TimeSlot,
)

# Import service layer
from app.services.doctor_service import list_doctors as service_list_doctors, get_doctor as service_get_doctor, get_doctor_availability as service_get_doctor_availability

router = APIRouter(prefix="/doctors", tags=["Doctors"])


@router.get(
    "",
    response_model=List[DoctorResponse],
    summary="List all doctors",
    description="Retrieves doctors with optional filtering by specialty (or speciality) and availability.",
)
def list_doctors(
    specialty: Optional[str] = Query(None, description="Filter by specialty (e.g. Cardiology)"),
    speciality: Optional[str] = Query(None, description="Alternative spelling for specialty"),
    is_available: Optional[bool] = Query(None, description="Filter by doctor availability status"),
    skip: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(100, ge=1, le=100, description="Pagination limit"),
    db: Session = Depends(get_db),
):
    """Delegate to service layer for doctor listing."""
    return service_list_doctors(db, specialty, speciality, is_available, skip, limit)


@router.get(
    "/{doctor_id}",
    response_model=DoctorResponse,
    summary="Get doctor details",
    description="Retrieves detailed profile for a specific doctor by their ID.",
)
def get_doctor(doctor_id: int, db: Session = Depends(get_db)):
    """Delegate to service layer for single doctor retrieval."""
    return service_get_doctor(db, doctor_id)


@router.get(
    "/{doctor_id}/availability",
    response_model=DoctorAvailabilityResponse,
    summary="Get doctor available appointment slots",
    description="Calculates 30-minute appointment slots for a doctor on a given date (09:00 - 17:00). Excludes booked slots.",
)
@router.get(
    "/{doctor_id}/slots",
    response_model=DoctorAvailabilityResponse,
    include_in_schema=False,  # Convenient alias
)
def get_doctor_availability(
    doctor_id: int,
    date: date_type = Query(..., description="Target date in YYYY-MM-DD format (e.g. 2026-11-26)"),
    db: Session = Depends(get_db),
):
    """Delegate to service layer for availability calculation."""
    return service_get_doctor_availability(db, doctor_id, date)
