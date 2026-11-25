from typing import List, Optional
from datetime import date as date_type, datetime

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.appointment import AppointmentCreate, AppointmentResponse

# Service layer imports
from app.services.appointment_service import (
    book_appointment as service_book_appointment,
    list_appointments as service_list_appointments,
    get_appointments_by_phone as service_get_appointments_by_phone,
    get_appointment as service_get_appointment,
    cancel_appointment as service_cancel_appointment,
)

router = APIRouter(prefix="/appointments", tags=["Appointments"])


@router.post(
    "",
    response_model=AppointmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Book a new appointment",
    description=(
        "Books an appointment with a doctor. Automatically identifies or registers the patient "
        "using their phone number, validates doctor availability, and prevents double-booking."
    ),
)
def create_appointment(payload: AppointmentCreate, db: Session = Depends(get_db)):
    """Delegate to service layer for appointment creation."""
    return service_book_appointment(db, payload)


@router.get(
    "",
    response_model=List[AppointmentResponse],
    summary="List all appointments",
    description="Retrieves appointments with optional filtering by doctor, patient, status, or date.",
)
def list_appointments(
    doctor_id: Optional[int] = Query(None, description="Filter by doctor ID"),
    patient_id: Optional[int] = Query(None, description="Filter by patient ID"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status (scheduled, confirmed, cancelled)"),
    date: Optional[date_type] = Query(None, description="Filter by appointment date (YYYY-MM-DD)"),
    skip: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(100, ge=1, le=100, description="Pagination limit"),
    db: Session = Depends(get_db),
):
    """Delegate to service layer for listing appointments."""
    return service_list_appointments(
        db,
        doctor_id=doctor_id,
        patient_id=patient_id,
        status_filter=status_filter,
        date=date,
        skip=skip,
        limit=limit,
    )


@router.get(
    "/by-phone/{phone}",
    response_model=List[AppointmentResponse],
    summary="Get appointments by patient phone number",
    description="Retrieves all appointments linked to a patient identified by their contact phone number.",
)
def get_appointments_by_phone(phone: str, db: Session = Depends(get_db)):
    """Delegate to service layer for phone‑based lookup."""
    return service_get_appointments_by_phone(db, phone)


@router.get(
    "/{appointment_id}",
    response_model=AppointmentResponse,
    summary="Get appointment details",
    description="Retrieves details for a specific appointment by ID.",
)
def get_appointment(appointment_id: int, db: Session = Depends(get_db)):
    """Delegate to service layer for single appointment retrieval."""
    return service_get_appointment(db, appointment_id)


@router.patch(
    "/{appointment_id}/cancel",
    response_model=AppointmentResponse,
    summary="Cancel an appointment",
    description="Cancels an existing appointment. Marks the status as 'cancelled' and frees the doctor slot.",
)
def cancel_appointment(appointment_id: int, db: Session = Depends(get_db)):
    """Delegate to service layer for cancellation."""
    return service_cancel_appointment(db, appointment_id)
