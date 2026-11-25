"""
SQLAlchemy ORM Models Module
Exports Base and all database entities for clean imports across the project.
"""
from app.database import Base
from app.models.doctor import Doctor
from app.models.patient import Patient
from app.models.appointment import Appointment

__all__ = ["Base", "Doctor", "Patient", "Appointment"]
