from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class Appointment(Base):
    """
    SQLAlchemy ORM Model representing an Appointment booking.
    
    Links a Doctor and a Patient through foreign keys.
    Tracks scheduled time, booking status, patient symptoms/reasons, and doctor notes.
    """
    __tablename__ = "appointments"

    # Primary key identifier
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Foreign Keys referencing Doctor and Patient
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)

    # Appointment Details
    appointment_date = Column(DateTime, nullable=False, index=True)
    status = Column(
        String(30),
        default="scheduled",
        nullable=False,
        index=True,
    )  # Status options: scheduled, confirmed, cancelled, completed
    reason = Column(Text, nullable=True)  # Primary reason for the visit (e.g. "Voice consultation follow-up")
    notes = Column(Text, nullable=True)   # Additional clinical or operational notes

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships back to Doctor and Patient models
    doctor = relationship("Doctor", back_populates="appointments")
    patient = relationship("Patient", back_populates="appointments")

    def __repr__(self) -> str:
        return f"<Appointment(id={self.id}, doctor_id={self.doctor_id}, patient_id={self.patient_id}, date='{self.appointment_date}', status='{self.status}')>"
