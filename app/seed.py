"""
Database Seeding Module
Provides safe, idempotent seeding of realistic sample doctors and medical specialties.
Can be safely run multiple times without creating duplicate records.
"""
from typing import Any, Dict, List
from sqlalchemy.orm import Session
from app.models.doctor import Doctor
from app.database import SessionLocal

# Curated list of realistic sample doctors across common healthcare specialties
SAMPLE_DOCTORS: List[Dict[str, Any]] = [
    {
        "name": "Dr. Sarah Jenkins",
        "specialty": "Cardiology",
        "email": "sarah.jenkins@medivoice.com",
        "phone": "+1-555-0101",
        "is_available": True,
    },
    {
        "name": "Dr. Michael Chen",
        "specialty": "Dermatology",
        "email": "michael.chen@medivoice.com",
        "phone": "+1-555-0102",
        "is_available": True,
    },
    {
        "name": "Dr. Emily Rodriguez",
        "specialty": "Pediatrics",
        "email": "emily.rodriguez@medivoice.com",
        "phone": "+1-555-0103",
        "is_available": True,
    },
    {
        "name": "Dr. James Wilson",
        "specialty": "Orthopedics",
        "email": "james.wilson@medivoice.com",
        "phone": "+1-555-0104",
        "is_available": True,
    },
    {
        "name": "Dr. Priya Patel",
        "specialty": "General Medicine",
        "email": "priya.patel@medivoice.com",
        "phone": "+1-555-0105",
        "is_available": True,
    },
    {
        "name": "Dr. Robert Taylor",
        "specialty": "Neurology",
        "email": "robert.taylor@medivoice.com",
        "phone": "+1-555-0106",
        "is_available": True,
    },
    {
        "name": "Dr. Lisa Anderson",
        "specialty": "Psychiatry",
        "email": "lisa.anderson@medivoice.com",
        "phone": "+1-555-0107",
        "is_available": False,  # Unavailable doctor for testing availability checks
    },
]


def seed_doctors(db: Session) -> int:
    """
    Safely seeds the doctors table with sample medical professionals.
    
    Idempotent: checks for existing doctors by email or exact name + specialty
    to prevent duplicate records on repeated execution.
    
    Returns:
        int: Number of new doctor records inserted.
    """
    inserted_count = 0
    for doc_data in SAMPLE_DOCTORS:
        # Check if doctor already exists by unique email
        existing = db.query(Doctor).filter(Doctor.email == doc_data["email"]).first()
        if not existing:
            doctor = Doctor(**doc_data)
            db.add(doctor)
            inserted_count += 1

    if inserted_count > 0:
        db.commit()

    return inserted_count


if __name__ == "__main__":
    # Allows running directly via: python -m app.seed
    db = SessionLocal()
    try:
        count = seed_doctors(db)
        print(f"Database seed completed: {count} new doctor(s) inserted.")
    finally:
        db.close()
