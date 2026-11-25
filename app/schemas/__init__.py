"""
Pydantic Schemas Package
Exports schemas used for request validation and response serialization.
"""
from app.schemas.health import HealthResponse
from app.schemas.doctor import (
    DoctorBase,
    DoctorCreate,
    DoctorResponse,
    TimeSlot,
    DoctorAvailabilityResponse,
)
from app.schemas.patient import PatientBase, PatientCreate, PatientResponse
from app.schemas.appointment import (
    AppointmentBase,
    AppointmentCreate,
    AppointmentResponse,
)

__all__ = [
    "HealthResponse",
    "DoctorBase",
    "DoctorCreate",
    "DoctorResponse",
    "TimeSlot",
    "DoctorAvailabilityResponse",
    "PatientBase",
    "PatientCreate",
    "PatientResponse",
    "AppointmentBase",
    "AppointmentCreate",
    "AppointmentResponse",
]
