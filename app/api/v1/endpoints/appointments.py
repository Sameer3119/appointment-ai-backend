from datetime import date as date_type, datetime, time, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.doctor import Doctor
from app.models.patient import Patient
from app.models.appointment import Appointment
from app.schemas.appointment import (
    AppointmentCreate,
    AppointmentResponse,
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
def create_appointment(
    payload: AppointmentCreate,
    db: Session = Depends(get_db),
):
    """
    Create a new appointment with comprehensive validation:
    1. Validates that the appointment date is in the future.
    2. Verifies that the doctor exists and is accepting appointments.
    3. Finds an existing patient by phone or creates a new patient record.
    4. Checks for conflicts to prevent double-booking the doctor at the same slot.
    5. Saves the appointment within a safe database transaction.
    """
    # -----------------------------------------------------------------------
    # 1. Appointment Date & Time Validation
    # -----------------------------------------------------------------------
    appt_dt = payload.appointment_date
    if appt_dt.tzinfo is not None:
        appt_dt = appt_dt.replace(tzinfo=None)

    # Prevent booking appointments in the past (allow a 1-minute grace margin for execution drift)
    now = datetime.now()
    if appt_dt < now - timedelta(minutes=1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Appointment date and time must be in the future.",
        )

    # Validate clinic working hours (09:00 AM to 05:00 PM)
    if not (time(9, 0) <= appt_dt.time() <= time(16, 30)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Appointments can only be scheduled during clinic hours between 09:00 AM and 04:30 PM.",
        )

    # -----------------------------------------------------------------------
    # 2. Doctor Existence & Availability Verification
    # -----------------------------------------------------------------------
    doctor = db.query(Doctor).filter(Doctor.id == payload.doctor_id).first()
    if not doctor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Doctor with id {payload.doctor_id} not found.",
        )

    if not doctor.is_available:
        prefix = "" if doctor.name.startswith("Dr.") else "Dr. "
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{prefix}{doctor.name} is currently not available for new appointments.",
        )

    # -----------------------------------------------------------------------
    # 3. Patient Identification / Auto-Registration by Phone
    # -----------------------------------------------------------------------
    patient: Optional[Patient] = None

    if payload.patient_phone:
        clean_phone = payload.patient_phone.strip()
        if len(clean_phone) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A valid patient phone number must be provided.",
            )

        # Lookup patient by phone
        patient = db.query(Patient).filter(Patient.phone == clean_phone).first()

        if not patient:
            # If patient does not exist, create a new patient record
            if not payload.patient_name or not payload.patient_name.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="patient_name is required when booking for a new patient.",
                )

            patient = Patient(
                name=payload.patient_name.strip(),
                phone=clean_phone,
                email=payload.patient_email.strip() if payload.patient_email else None,
                date_of_birth=payload.patient_date_of_birth,
                gender=payload.patient_gender,
            )
            db.add(patient)
            db.flush()  # Generates patient.id within the active transaction

    elif payload.patient_id:
        patient = db.query(Patient).filter(Patient.id == payload.patient_id).first()
        if not patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient with id {payload.patient_id} not found.",
            )

    if not patient:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to identify or create patient for this booking.",
        )

    # -----------------------------------------------------------------------
    # 4. Double-Booking & Slot Conflict Prevention
    # -----------------------------------------------------------------------
    slot_duration = timedelta(minutes=30)

    # Check if this doctor already has an active (non-cancelled) appointment overlapping this time
    doctor_conflict = (
        db.query(Appointment)
        .filter(
            Appointment.doctor_id == doctor.id,
            Appointment.status != "cancelled",
            Appointment.appointment_date > (appt_dt - slot_duration),
            Appointment.appointment_date < (appt_dt + slot_duration),
        )
        .first()
    )
    if doctor_conflict:
        prefix = "" if doctor.name.startswith("Dr.") else "Dr. "
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"{prefix}{doctor.name} already has an appointment booked at "
                f"{doctor_conflict.appointment_date.strftime('%Y-%m-%d %H:%M')}. "
                "Please choose an alternative available slot."
            ),
        )

    # Check if this patient already has an active appointment at the exact same time
    patient_conflict = (
        db.query(Appointment)
        .filter(
            Appointment.patient_id == patient.id,
            Appointment.status != "cancelled",
            Appointment.appointment_date == appt_dt,
        )
        .first()
    )
    if patient_conflict:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Patient '{patient.name}' already has an active appointment at "
                f"{appt_dt.strftime('%Y-%m-%d %H:%M')}."
            ),
        )

    # -----------------------------------------------------------------------
    # 5. Database Transaction & Record Creation
    # -----------------------------------------------------------------------
    try:
        new_appointment = Appointment(
            doctor_id=doctor.id,
            patient_id=patient.id,
            appointment_date=appt_dt,
            status=payload.status or "scheduled",
            reason=payload.reason,
            notes=payload.notes,
        )
        db.add(new_appointment)
        db.commit()
        db.refresh(new_appointment)
        return new_appointment
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record appointment: {str(exc)}",
        )


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
    """
    List appointments with flexible filtering.
    """
    query = db.query(Appointment)

    if doctor_id is not None:
        query = query.filter(Appointment.doctor_id == doctor_id)

    if patient_id is not None:
        query = query.filter(Appointment.patient_id == patient_id)

    if status_filter is not None:
        query = query.filter(Appointment.status == status_filter)

    if date is not None:
        start_of_day = datetime.combine(date, time(0, 0, 0))
        end_of_day = datetime.combine(date, time(23, 59, 59))
        query = query.filter(
            Appointment.appointment_date >= start_of_day,
            Appointment.appointment_date <= end_of_day,
        )

    return query.order_by(Appointment.appointment_date.asc()).offset(skip).limit(limit).all()


@router.get(
    "/by-phone/{phone}",
    response_model=List[AppointmentResponse],
    summary="Get appointments by patient phone number",
    description="Retrieves all appointments linked to a patient identified by their contact phone number.",
)
def get_appointments_by_phone(
    phone: str,
    db: Session = Depends(get_db),
):
    """
    Retrieve appointments for a patient identified by phone number.
    Returns an empty list if the patient is not found or has no bookings.
    """
    clean_phone = phone.strip()
    patient = db.query(Patient).filter(Patient.phone == clean_phone).first()
    if not patient:
        return []

    return (
        db.query(Appointment)
        .filter(Appointment.patient_id == patient.id)
        .order_by(Appointment.appointment_date.desc())
        .all()
    )


@router.get(
    "/{appointment_id}",
    response_model=AppointmentResponse,
    summary="Get appointment details",
    description="Retrieves details for a specific appointment by ID.",
)
def get_appointment(
    appointment_id: int,
    db: Session = Depends(get_db),
):
    """
    Retrieve a single appointment by ID.
    Raises 404 if the appointment does not exist.
    """
    appointment = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if not appointment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Appointment with id {appointment_id} not found.",
        )
    return appointment


@router.patch(
    "/{appointment_id}/cancel",
    response_model=AppointmentResponse,
    summary="Cancel an appointment",
    description="Cancels an existing appointment. Marks the status as 'cancelled' and frees the doctor slot.",
)
def cancel_appointment(
    appointment_id: int,
    db: Session = Depends(get_db),
):
    """
    Cancel an appointment by ID.
    - Raises 404 if not found.
    - Raises 400 if already cancelled.
    - Sets status to 'cancelled' and commits within a safe transaction.
    """
    appointment = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if not appointment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Appointment with id {appointment_id} not found.",
        )

    if appointment.status == "cancelled":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Appointment with id {appointment_id} is already cancelled.",
        )

    try:
        appointment.status = "cancelled"
        db.commit()
        db.refresh(appointment)
        return appointment
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel appointment: {str(exc)}",
        )
