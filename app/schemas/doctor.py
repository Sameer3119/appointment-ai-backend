from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class DoctorBase(BaseModel):
    """Base Pydantic schema for Doctor with shared attributes."""
    name: str = Field(..., min_length=1, max_length=100, description="Full name of the doctor")
    specialty: str = Field(..., min_length=1, max_length=100, description="Medical specialization")
    email: Optional[str] = Field(None, max_length=120, description="Contact email address")
    phone: Optional[str] = Field(None, max_length=30, description="Contact phone number")
    is_available: bool = Field(True, description="Availability flag for booking")


class DoctorCreate(DoctorBase):
    """Schema for creating a new Doctor record."""
    pass


class DoctorResponse(DoctorBase):
    """Schema for Doctor output responses, supporting ORM mapping."""
    id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class TimeSlot(BaseModel):
    """Schema representing an individual time slot for appointment scheduling."""
    start_time: str = Field(..., description="Slot start time in HH:MM format (e.g. '09:00')")
    end_time: str = Field(..., description="Slot end time in HH:MM format (e.g. '09:30')")
    datetime_slot: datetime = Field(..., description="Full datetime representation of the slot start")
    is_available: bool = Field(..., description="Whether this slot is available for booking")


class DoctorAvailabilityResponse(BaseModel):
    """Schema for doctor availability responses showing all slots on a specified date."""
    doctor_id: int = Field(..., description="Doctor's ID")
    doctor_name: str = Field(..., description="Doctor's full name")
    specialty: str = Field(..., description="Doctor's medical specialty")
    date: str = Field(..., description="Date for availability in YYYY-MM-DD format")
    slot_duration_minutes: int = Field(default=30, description="Duration of each appointment slot in minutes")
    available_slots: List[str] = Field(
        default_factory=list,
        description="Quick list of available start times (e.g. ['09:00', '09:30', '10:00'])",
    )
    slots: List[TimeSlot] = Field(
        default_factory=list,
        description="Complete list of all time slots with individual availability status",
    )
