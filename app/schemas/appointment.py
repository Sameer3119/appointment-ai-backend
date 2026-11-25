from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator
from app.schemas.doctor import DoctorResponse
from app.schemas.patient import PatientResponse


class AppointmentBase(BaseModel):
    """Base Pydantic schema for Appointment with core attributes."""
    doctor_id: int = Field(..., description="ID of the doctor assigned to the appointment")
    appointment_date: datetime = Field(..., description="Scheduled date and time of appointment")
    status: str = Field(
        default="scheduled",
        description="Appointment status: scheduled, confirmed, cancelled, completed",
    )
    reason: Optional[str] = Field(None, description="Primary reason or symptoms for the visit")
    notes: Optional[str] = Field(None, description="Additional consultation or staff notes")


class AppointmentCreate(AppointmentBase):
    """
    Schema for booking a new appointment.
    
    Supports identifying patient by patient_id OR patient_phone.
    If patient_phone is provided and no patient exists, a patient record
    will be automatically created using patient_name, patient_email, etc.
    """
    patient_id: Optional[int] = Field(None, description="ID of an existing patient (if known)")
    patient_phone: Optional[str] = Field(None, description="Patient contact phone number (used to find or create patient)")
    patient_name: Optional[str] = Field(None, description="Patient's full name (required if creating a new patient)")
    patient_email: Optional[str] = Field(None, description="Patient's email address")
    patient_date_of_birth: Optional[str] = Field(None, description="Patient's date of birth (YYYY-MM-DD)")
    patient_gender: Optional[str] = Field(None, description="Patient's gender")

    @model_validator(mode="after")
    def validate_patient_identifier(self):
        """Ensures that either patient_id or patient_phone is provided."""
        if not self.patient_id and not self.patient_phone:
            raise ValueError("Either patient_id or patient_phone must be provided to book an appointment")
        return self


class AppointmentResponse(AppointmentBase):
    """Schema for Appointment output responses, supporting nested ORM relationships."""
    id: int
    patient_id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    doctor: Optional[DoctorResponse] = None
    patient: Optional[PatientResponse] = None

    model_config = ConfigDict(from_attributes=True)
