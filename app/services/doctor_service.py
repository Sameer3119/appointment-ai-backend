"""Doctor service layer providing business logic for doctor related operations."""

from datetime import datetime, time, timedelta
from typing import List, Optional

from sqlalchemy.orm import Session

from app.models.doctor import Doctor
from app.models.appointment import Appointment
from app.schemas.doctor import DoctorResponse, DoctorAvailabilityResponse, TimeSlot


def list_doctors(
    db: Session,
    specialty: Optional[str] = None,
    speciality: Optional[str] = None,
    is_available: Optional[bool] = None,
    skip: int = 0,
    limit: int = 100,
) -> List[DoctorResponse]:
    """Return a list of doctors optionally filtered by specialty and availability."""
    query = db.query(Doctor)
    target_specialty = specialty or speciality
    if target_specialty:
        query = query.filter(Doctor.specialty.ilike(f"%{target_specialty.strip()}%"))
    if is_available is not None:
        query = query.filter(Doctor.is_available == is_available)
    return query.order_by(Doctor.id.asc()).offset(skip).limit(limit).all()


def get_doctor(db: Session, doctor_id: int) -> Doctor:
    """Retrieve a doctor by id or raise 404 HTTPException."""
    from fastapi import HTTPException, status

    doctor = db.query(Doctor).filter(Doctor.id == doctor_id).first()
    if not doctor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Doctor with id {doctor_id} not found",
        )
    return doctor


def get_doctor_availability(
    db: Session,
    doctor_id: int,
    date: datetime.date,
) -> DoctorAvailabilityResponse:
    """Calculate 30‑minute appointment slots for a given doctor/date.

    Excludes slots already booked by active appointments and respects the
    doctor's ``is_available`` flag.
    """
    from fastapi import HTTPException, status

    doctor = db.query(Doctor).filter(Doctor.id == doctor_id).first()
    if not doctor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Doctor with id {doctor_id} not found",
        )

    slot_duration_minutes = 30
    start_hour = 9
    end_hour = 17

    # Define day boundaries for the query
    start_of_day = datetime.combine(date, time(0, 0, 0))
    end_of_day = datetime.combine(date, time(23, 59, 59))

    # Existing active appointments for that day
    booked_appointments = (
        db.query(Appointment)
        .filter(
            Appointment.doctor_id == doctor_id,
            Appointment.appointment_date >= start_of_day,
            Appointment.appointment_date <= end_of_day,
            Appointment.status != "cancelled",
        )
        .all()
    )
    booked_times = {appt.appointment_date.strftime("%H:%M") for appt in booked_appointments}

    slots: List[TimeSlot] = []
    available_slots: List[str] = []
    doctor_is_active = doctor.is_available

    current_slot_start = datetime.combine(date, time(start_hour, 0))
    day_end_dt = datetime.combine(date, time(end_hour, 0))

    while current_slot_start + timedelta(minutes=slot_duration_minutes) <= day_end_dt:
        slot_end = current_slot_start + timedelta(minutes=slot_duration_minutes)
        time_str = current_slot_start.strftime("%H:%M")
        end_time_str = slot_end.strftime("%H:%M")
        is_free = doctor_is_active and (time_str not in booked_times)
        slots.append(
            TimeSlot(
                start_time=time_str,
                end_time=end_time_str,
                datetime_slot=current_slot_start,
                is_available=is_free,
            )
        )
        if is_free:
            available_slots.append(time_str)
        current_slot_start = slot_end

    return DoctorAvailabilityResponse(
        doctor_id=doctor.id,
        doctor_name=doctor.name,
        specialty=doctor.specialty,
        date=date.isoformat(),
        slot_duration_minutes=slot_duration_minutes,
        available_slots=available_slots,
        slots=slots,
    )
