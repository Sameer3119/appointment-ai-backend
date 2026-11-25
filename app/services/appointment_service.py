"""Appointment service layer encapsulating business logic for appointments."""

from datetime import datetime, time, timedelta
from typing import List, Optional

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.doctor import Doctor
from app.models.patient import Patient
from app.models.appointment import Appointment
from app.schemas.appointment import AppointmentCreate, AppointmentResponse


def get_or_create_patient(
    db: Session,
    phone: Optional[str] = None,
    name: Optional[str] = None,
    email: Optional[str] = None,
    date_of_birth: Optional[datetime] = None,
    gender: Optional[str] = None,
    patient_id: Optional[int] = None,
) -> Patient:
    """Find an existing patient by phone or id, or create a new one.

    Raises HTTPException on validation errors.
    """
    patient: Optional[Patient] = None
    if phone:
        clean_phone = phone.strip()
        if len(clean_phone) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A valid patient phone number must be provided.",
            )
        patient = db.query(Patient).filter(Patient.phone == clean_phone).first()
        if not patient:
            if not name or not name.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="patient_name is required when booking for a new patient.",
                )
            patient = Patient(
                name=name.strip(),
                phone=clean_phone,
                email=email.strip() if email else None,
                date_of_birth=date_of_birth,
                gender=gender,
            )
            db.add(patient)
            db.flush()  # assign id
    elif patient_id:
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient with id {patient_id} not found.",
            )
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to identify or create patient for this booking.",
        )
    return patient


def validate_appointment_datetime(appt_dt: datetime) -> None:
    """Validate that the appointment datetime is in the future and within clinic hours.

    Clinic hours: 09:00‑16:30 (30‑minute slots ending at 17:00).
    """
    if appt_dt.tzinfo is not None:
        appt_dt = appt_dt.replace(tzinfo=None)
    now = datetime.now()
    if appt_dt < now - timedelta(minutes=1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Appointment date and time must be in the future.",
        )
    if not (time(9, 0) <= appt_dt.time() <= time(16, 30)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Appointments can only be scheduled during clinic hours between 09:00 AM and 04:30 PM.",
        )


def check_doctor_conflict(db: Session, doctor: Doctor, appt_dt: datetime) -> None:
    """Raise 409 if the doctor already has an overlapping active appointment."""
    slot_duration = timedelta(minutes=30)
    conflict = (
        db.query(Appointment)
        .filter(
            Appointment.doctor_id == doctor.id,
            Appointment.status != "cancelled",
            Appointment.appointment_date >= (appt_dt - slot_duration),
            Appointment.appointment_date <= (appt_dt + slot_duration),
        )
        .first()
    )
    if conflict:
        prefix = "" if doctor.name.startswith("Dr.") else "Dr. "
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"{prefix}{doctor.name} already has an appointment booked at "
                f"{conflict.appointment_date.strftime('%Y-%m-%d %H:%M')}. "
                "Please choose an alternative available slot."
            ),
        )


def check_patient_conflict(db: Session, patient: Patient, appt_dt: datetime) -> None:
    """Raise 409 if the patient already has an active appointment at the same time."""
    conflict = (
        db.query(Appointment)
        .filter(
            Appointment.patient_id == patient.id,
            Appointment.status != "cancelled",
            Appointment.appointment_date == appt_dt,
        )
        .first()
    )
    if conflict:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Patient '{patient.name}' already has an active appointment at "
                f"{appt_dt.strftime('%Y-%m-%d %H:%M')}."
            ),
        )


def book_appointment(db: Session, payload: AppointmentCreate) -> Appointment:
    """Full booking flow: validation, patient handling, conflict checks, persistence."""
    # 1. Validate datetime
    validate_appointment_datetime(payload.appointment_date)

    # 2. Doctor existence & availability
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

    # 3. Patient identification / creation
    patient = get_or_create_patient(
        db,
        phone=payload.patient_phone,
        name=payload.patient_name,
        email=payload.patient_email,
        date_of_birth=payload.patient_date_of_birth,
        gender=payload.patient_gender,
        patient_id=payload.patient_id,
    )

    # 4. Conflict checks
    check_doctor_conflict(db, doctor, payload.appointment_date)
    check_patient_conflict(db, patient, payload.appointment_date)

    # 5. Persist appointment within a transaction
    try:
        new_appt = Appointment(
            doctor_id=doctor.id,
            patient_id=patient.id,
            appointment_date=payload.appointment_date,
            status=payload.status or "scheduled",
            reason=payload.reason,
            notes=payload.notes,
        )
        db.add(new_appt)
        db.commit()
        db.refresh(new_appt)
        return new_appt
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record appointment: {str(exc)}",
        )


def list_appointments(
    db: Session,
    doctor_id: Optional[int] = None,
    patient_id: Optional[int] = None,
    status_filter: Optional[str] = None,
    date: Optional[datetime.date] = None,
    skip: int = 0,
    limit: int = 100,
) -> List[Appointment]:
    """Retrieve appointments with flexible optional filters."""
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


def get_appointments_by_phone(db: Session, phone: str) -> List[Appointment]:
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


def get_appointment(db: Session, appointment_id: int) -> Appointment:
    appt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if not appt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Appointment with id {appointment_id} not found.",
        )
    return appt


def cancel_appointment(db: Session, appointment_id: int) -> Appointment:
    appt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if not appt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Appointment with id {appointment_id} not found.",
        )
    if appt.status == "cancelled":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Appointment with id {appointment_id} is already cancelled.",
        )
    try:
        appt.status = "cancelled"
        db.commit()
        db.refresh(appt)
        return appt
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel appointment: {str(exc)}",
        )
