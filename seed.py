"""
Standalone Database Seeding Runner
Usage: python seed.py
"""
from app.database import SessionLocal
from app.seed import seed_doctors

if __name__ == "__main__":
    db = SessionLocal()
    try:
        inserted = seed_doctors(db)
        print(f"[Seed] Successfully executed. {inserted} new doctor(s) added.")
    finally:
        db.close()
