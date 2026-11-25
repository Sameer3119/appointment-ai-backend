from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class PatientBase(BaseModel):
    """Base Pydantic schema for Patient with shared attributes."""
    name: str = Field(..., min_length=1, max_length=100, description="Patient's full name")
    phone: str = Field(..., min_length=1, max_length=30, description="Patient's contact phone")
    email: Optional[str] = Field(None, max_length=120, description="Patient's optional email")
    date_of_birth: Optional[str] = Field(None, max_length=20, description="Date of birth in YYYY-MM-DD")
    gender: Optional[str] = Field(None, max_length=20, description="Gender identity")


class PatientCreate(PatientBase):
    """Schema for creating a new Patient record."""
    pass


class PatientResponse(PatientBase):
    """Schema for Patient output responses, supporting ORM mapping."""
    id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
