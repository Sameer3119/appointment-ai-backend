from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class Doctor(Base):
    """
    SQLAlchemy ORM Model representing a Doctor in the system.
    
    Stores doctor's contact details, specialty, and availability status.
    Has a one-to-many relationship with appointments.
    """
    __tablename__ = "doctors"

    # Primary key identifier
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Doctor details
    name = Column(String(100), nullable=False, index=True)
    specialty = Column(String(100), nullable=False, index=True)
    email = Column(String(120), unique=True, index=True, nullable=True)
    phone = Column(String(30), nullable=True)

    # Availability flag for scheduling (True = active/accepting bookings)
    is_available = Column(Boolean, default=True, nullable=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationship: A doctor has multiple scheduled appointments
    appointments = relationship(
        "Appointment",
        back_populates="doctor",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Doctor(id={self.id}, name='{self.name}', specialty='{self.specialty}')>"
