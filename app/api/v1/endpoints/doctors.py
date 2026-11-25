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
    """
    List doctors with optional filtering:
    - specialty/speciality: Case-insensitive match on medical specialty
    - is_available: Filter by True/False
    """
    query = db.query(Doctor)

    target_specialty = specialty or speciality
    if target_specialty:
        # Case-insensitive substring match
        query = query.filter(Doctor.specialty.ilike(f"%{target_specialty.strip()}%"))

    if is_available is not None:
        query = query.filter(Doctor.is_available == is_available)

    return query.order_by(Doctor.id.asc()).offset(skip).limit(limit).all()


@router.get(
    "/{doctor_id}",
    response_model=DoctorResponse,
    summary="Get doctor details",
    description="Retrieves detailed profile for a specific doctor by their ID.",
)
def get_doctor(doctor_id: int, db: Session = Depends(get_db)):
    """
    Retrieve doctor details by doctor ID.
    Raises 404 if the doctor does not exist.
    """
    doctor = db.query(Doctor).filter(Doctor.id == doctor_id).first()
    if not doctor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Doctor with id {doctor_id} not found",
        )
    return doctor


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
    """
    Retrieve available appointment slots for a doctor on a specified date.
    
    Working hours: 09:00 AM to 05:00 PM (17:00)
    Slot duration: 30 minutes
    
    Any slot already booked by an active (non-cancelled) appointment is marked unavailable.
    If the doctor's is_available status is False, all slots are marked unavailable.
    """
    doctor = db.query(Doctor).filter(Doctor.id == doctor_id).first()
    if not doctor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Doctor with id {doctor_id} not found",
        )

    slot_duration_minutes = 30
    start_hour = 9   # 09:00 AM
    end_hour = 17    # 05:00 PM

    # Define the time boundary for the queried date
    start_of_day = datetime.combine(date, time(0, 0, 0))
    end_of_day = datetime.combine(date, time(23, 59, 59))

    # Fetch existing active appointments for this doctor on the requested date
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

    # Set of booked appointment times in "HH:MM" format
    booked_times = {appt.appointment_date.strftime("%H:%M") for appt in booked_appointments}

    slots: List[TimeSlot] = []
    available_slots: List[str] = []

    # If the doctor is generally unavailable, no slots can be booked
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
