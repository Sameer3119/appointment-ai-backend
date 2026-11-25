from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class Patient(Base):
    """
    SQLAlchemy ORM Model representing a Patient in the clinic.
    
    Stores patient demographics and contact information.
    Has a one-to-many relationship with appointments.
    """
    __tablename__ = "patients"

    # Primary key identifier
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Patient details
    name = Column(String(100), nullable=False, index=True)
    phone = Column(String(30), nullable=False, index=True)  # Key contact number
    email = Column(String(120), nullable=True, index=True)
    date_of_birth = Column(String(20), nullable=True)       # Format: YYYY-MM-DD
    gender = Column(String(20), nullable=True)              # E.g. Male, Female, Other

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationship: A patient can have multiple appointments
    appointments = relationship(
        "Appointment",
        back_populates="patient",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Patient(id={self.id}, name='{self.name}', phone='{self.phone}')>"
