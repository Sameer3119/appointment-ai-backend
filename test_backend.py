"""
MediVoice AI Backend - Comprehensive Test Suite
Tests all endpoints using an isolated in-memory SQLite database.
Guarantees that the development database (appointments.db) is never modified during test execution.
"""
from datetime import datetime, timedelta
import os
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.models.doctor import Doctor
from app.models.patient import Patient
from app.models.appointment import Appointment
from app.seed import seed_doctors, SAMPLE_DOCTORS

# ---------------------------------------------------------------------------
# Setup Isolated In-Memory Test Database
# ---------------------------------------------------------------------------
TEST_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    """Dependency override providing isolated in-memory DB sessions."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Apply dependency override on FastAPI app
app.dependency_overrides[get_db] = override_get_db


def run_tests():
    print("=" * 70)
    print("MEDIVOICE AI BACKEND TEST SUITE (ISOLATED IN-MEMORY DATABASE)")
    print("=" * 70)

    # Record mod-time and size of appointments.db if present to ensure it's not touched
    dev_db_path = "appointments.db"
    dev_db_initial_stat = None
    if os.path.exists(dev_db_path):
        dev_db_initial_stat = (os.path.getmtime(dev_db_path), os.path.getsize(dev_db_path))

    # Create tables in the isolated test database
    Base.metadata.create_all(bind=test_engine)

    with TestClient(app) as client:
        # -------------------------------------------------------------------
        # 1. Health Checks
        # -------------------------------------------------------------------
        print("\n[TEST 1] Health Check Endpoints")
        res = client.get("/health")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        assert res.json().get("status") == "ok"
        assert res.json().get("database") == "connected"
        print("  ✓ GET /health returns 200 OK with connected database")

        res_v1 = client.get("/api/v1/health")
        assert res_v1.status_code == 200
        print("  ✓ GET /api/v1/health returns 200 OK")

        # -------------------------------------------------------------------
        # 2. Database Seeding & Idempotency
        # -------------------------------------------------------------------
        print("\n[TEST 2] Database Seeding & Idempotency Check")
        with TestingSessionLocal() as db:
            first_seed_count = seed_doctors(db)
            print(f"  First seed inserted: {first_seed_count} doctors")
            assert first_seed_count == len(SAMPLE_DOCTORS), f"Expected {len(SAMPLE_DOCTORS)}, got {first_seed_count}"

            # Running seed a second time must NOT create duplicates
            second_seed_count = seed_doctors(db)
            print(f"  Second seed inserted: {second_seed_count} doctors")
            assert second_seed_count == 0, f"Expected 0 duplicates, got {second_seed_count}"

            total_doctors = db.query(Doctor).count()
            assert total_doctors == len(SAMPLE_DOCTORS), f"Total doctors mismatch: {total_doctors}"
        print("  ✓ Doctor seed is fully idempotent (no duplicate records)")

        # -------------------------------------------------------------------
        # 3. Doctor Endpoints
        # -------------------------------------------------------------------
        print("\n[TEST 3] Doctor Listing & Filtering Endpoints")
        # List all doctors
        res_docs = client.get("/api/v1/doctors")
        assert res_docs.status_code == 200
        docs = res_docs.json()
        assert len(docs) == len(SAMPLE_DOCTORS)
        print(f"  ✓ GET /api/v1/doctors returns all {len(docs)} doctors")

        # Filter by specialty: Cardiology
        res_cardio = client.get("/api/v1/doctors?specialty=Cardiology")
        assert res_cardio.status_code == 200
        cardio_docs = res_cardio.json()
        assert len(cardio_docs) >= 1
        assert all(d["specialty"] == "Cardiology" for d in cardio_docs)
        print(f"  ✓ GET /api/v1/doctors?specialty=Cardiology returns {len(cardio_docs)} doctor(s)")

        # Filter by alternative spelling: speciality=Dermatology
        res_derma = client.get("/api/v1/doctors?speciality=Dermatology")
        assert res_derma.status_code == 200
        derma_docs = res_derma.json()
        assert len(derma_docs) >= 1
        assert derma_docs[0]["specialty"] == "Dermatology"
        print("  ✓ GET /api/v1/doctors?speciality=... handles British spelling")

        # Filter by availability
        res_avail = client.get("/api/v1/doctors?is_available=false")
        assert res_avail.status_code == 200
        unavail_docs = res_avail.json()
        assert len(unavail_docs) >= 1
        assert unavail_docs[0]["is_available"] is False
        unavailable_doctor_id = unavail_docs[0]["id"]
        print(f"  ✓ GET /api/v1/doctors?is_available=false returns unavailable doctor ID {unavailable_doctor_id}")

        # Get doctor details
        active_doctor_id = docs[0]["id"]
        res_detail = client.get(f"/api/v1/doctors/{active_doctor_id}")
        assert res_detail.status_code == 200
        assert res_detail.json()["id"] == active_doctor_id
        print(f"  ✓ GET /api/v1/doctors/{active_doctor_id} returns doctor details")

        # Get non-existent doctor
        res_missing_doc = client.get("/api/v1/doctors/99999")
        assert res_missing_doc.status_code == 404
        print("  ✓ GET /api/v1/doctors/99999 correctly returns 404 Not Found")

        # -------------------------------------------------------------------
        # 4. Doctor Availability & Slot Generation
        # -------------------------------------------------------------------
        print("\n[TEST 4] Doctor Availability & Slot Calculation")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        # Check availability for active doctor
        res_slots = client.get(f"/api/v1/doctors/{active_doctor_id}/availability?date={tomorrow}")
        assert res_slots.status_code == 200
        slot_data = res_slots.json()
        assert slot_data["doctor_id"] == active_doctor_id
        assert slot_data["date"] == tomorrow
        assert slot_data["slot_duration_minutes"] == 30
        assert len(slot_data["slots"]) == 16  # 09:00 to 17:00 in 30-min increments = 16 slots
        assert "09:00" in slot_data["available_slots"]
        assert "16:30" in slot_data["available_slots"]
        print(f"  ✓ GET /api/v1/doctors/{active_doctor_id}/availability returns 16 slots (09:00 - 17:00)")

        # Check availability for unavailable doctor
        res_unavail_slots = client.get(f"/api/v1/doctors/{unavailable_doctor_id}/availability?date={tomorrow}")
        assert res_unavail_slots.status_code == 200
        assert len(res_unavail_slots.json()["available_slots"]) == 0
        print("  ✓ Unavailable doctor returns 0 available slots")

        # -------------------------------------------------------------------
        # 5. Appointment Booking & Patient Auto-Registration
        # -------------------------------------------------------------------
        print("\n[TEST 5] Appointment Booking & Patient Auto-Registration")
        test_patient_phone = "+1-555-987-6543"
        test_patient_name = "Jane Doe"
        booking_datetime = f"{tomorrow}T10:00:00"

        # Book appointment with patient phone (patient does not exist yet)
        booking_payload = {
            "doctor_id": active_doctor_id,
            "appointment_date": booking_datetime,
            "patient_phone": test_patient_phone,
            "patient_name": test_patient_name,
            "patient_email": "jane.doe@example.com",
            "reason": "Routine cardiology consultation",
            "notes": "Referred by GP",
        }
        res_booking = client.post("/api/v1/appointments", json=booking_payload)
        assert res_booking.status_code == 201, f"Booking failed: {res_booking.text}"
        booking_data = res_booking.json()
        assert booking_data["status"] == "scheduled"
        assert booking_data["doctor_id"] == active_doctor_id
        assert booking_data["patient"]["phone"] == test_patient_phone
        assert booking_data["patient"]["name"] == test_patient_name
        created_appointment_id = booking_data["id"]
        created_patient_id = booking_data["patient_id"]
        print(f"  ✓ POST /api/v1/appointments created appointment ID {created_appointment_id}")
        print(f"  ✓ Automatically registered new patient ID {created_patient_id} ({test_patient_name})")

        # -------------------------------------------------------------------
        # 6. Availability Update Verification
        # -------------------------------------------------------------------
        print("\n[TEST 6] Verification of Slot Availability After Booking")
        res_slots_after = client.get(f"/api/v1/doctors/{active_doctor_id}/availability?date={tomorrow}")
        assert res_slots_after.status_code == 200
        available_after = res_slots_after.json()["available_slots"]
        assert "10:00" not in available_after, "Booked slot '10:00' should no longer be available!"
        assert "09:30" in available_after
        assert "10:30" in available_after
        print("  ✓ 10:00 AM slot is correctly removed from available slots")

        # -------------------------------------------------------------------
        # 7. Double-Booking Prevention & Validation Errors
        # -------------------------------------------------------------------
        print("\n[TEST 7] Double-Booking Prevention & Validation Errors")
        # Attempt exact same booking (same doctor, same time)
        res_conflict = client.post("/api/v1/appointments", json={
            "doctor_id": active_doctor_id,
            "appointment_date": booking_datetime,
            "patient_phone": "+1-555-111-2222",
            "patient_name": "Second Patient",
        })
        assert res_conflict.status_code == 409, f"Expected 409 Conflict, got {res_conflict.status_code}"
        print(f"  ✓ Exact slot double-booking blocked with 409 Conflict: '{res_conflict.json()['detail']}'")

        # Attempt booking unavailable doctor
        res_bad_doc = client.post("/api/v1/appointments", json={
            "doctor_id": unavailable_doctor_id,
            "appointment_date": f"{tomorrow}T11:00:00",
            "patient_phone": test_patient_phone,
            "patient_name": test_patient_name,
        })
        assert res_bad_doc.status_code == 400
        print("  ✓ Booking with unavailable doctor blocked with 400 Bad Request")

        # Attempt booking non-existent doctor
        res_no_doc = client.post("/api/v1/appointments", json={
            "doctor_id": 99999,
            "appointment_date": f"{tomorrow}T11:00:00",
            "patient_phone": test_patient_phone,
            "patient_name": test_patient_name,
        })
        assert res_no_doc.status_code == 404
        print("  ✓ Booking with non-existent doctor blocked with 404 Not Found")

        # Attempt booking in the past
        past_datetime = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%dT10:00:00")
        res_past = client.post("/api/v1/appointments", json={
            "doctor_id": active_doctor_id,
            "appointment_date": past_datetime,
            "patient_phone": test_patient_phone,
            "patient_name": test_patient_name,
        })
        assert res_past.status_code == 400
        print("  ✓ Booking in the past blocked with 400 Bad Request")

        # Attempt booking outside clinic hours (e.g. 02:00 AM)
        res_night = client.post("/api/v1/appointments", json={
            "doctor_id": active_doctor_id,
            "appointment_date": f"{tomorrow}T02:00:00",
            "patient_phone": test_patient_phone,
            "patient_name": test_patient_name,
        })
        assert res_night.status_code == 400
        print("  ✓ Booking outside clinic hours blocked with 400 Bad Request")

        # Attempt booking for new patient without providing patient_name
        res_no_name = client.post("/api/v1/appointments", json={
            "doctor_id": active_doctor_id,
            "appointment_date": f"{tomorrow}T14:00:00",
            "patient_phone": "+1-555-000-9999",
        })
        assert res_no_name.status_code == 400
        print("  ✓ Booking new patient without name blocked with 400 Bad Request")

        # -------------------------------------------------------------------
        # 8. Appointment Retrieval Endpoints
        # -------------------------------------------------------------------
        print("\n[TEST 8] Appointment Retrieval Endpoints")
        # List all appointments
        res_all_appts = client.get("/api/v1/appointments")
        assert res_all_appts.status_code == 200
        appts = res_all_appts.json()
        assert len(appts) >= 1
        print(f"  ✓ GET /api/v1/appointments returned {len(appts)} appointment(s)")

        # Get specific appointment by ID
        res_one_appt = client.get(f"/api/v1/appointments/{created_appointment_id}")
        assert res_one_appt.status_code == 200
        assert res_one_appt.json()["id"] == created_appointment_id
        print(f"  ✓ GET /api/v1/appointments/{created_appointment_id} returned appointment details")

        # Get appointments by patient phone
        res_phone_appts = client.get(f"/api/v1/appointments/by-phone/{test_patient_phone}")
        assert res_phone_appts.status_code == 200
        phone_appts = res_phone_appts.json()
        assert len(phone_appts) >= 1
        assert phone_appts[0]["id"] == created_appointment_id
        print(f"  ✓ GET /api/v1/appointments/by-phone/{test_patient_phone} returned patient's bookings")

        # Unknown phone returns empty list
        res_unknown_phone = client.get("/api/v1/appointments/by-phone/0000000000")
        assert res_unknown_phone.status_code == 200
        assert res_unknown_phone.json() == []
        print("  ✓ GET /api/v1/appointments/by-phone/0000000000 returned empty list []")

        # -------------------------------------------------------------------
        # 9. Appointment Cancellation
        # -------------------------------------------------------------------
        print("\n[TEST 9] Appointment Cancellation & Slot Re-opening")
        res_cancel = client.patch(f"/api/v1/appointments/{created_appointment_id}/cancel")
        assert res_cancel.status_code == 200
        assert res_cancel.json()["status"] == "cancelled"
        print(f"  ✓ PATCH /api/v1/appointments/{created_appointment_id}/cancel cancelled the appointment")

        # Attempt to cancel an already cancelled appointment
        res_cancel_again = client.patch(f"/api/v1/appointments/{created_appointment_id}/cancel")
        assert res_cancel_again.status_code == 400
        print("  ✓ Cancelling already cancelled appointment blocked with 400 Bad Request")

        # Attempt to cancel non-existent appointment
        res_cancel_missing = client.patch("/api/v1/appointments/99999/cancel")
        assert res_cancel_missing.status_code == 404
        print("  ✓ Cancelling non-existent appointment returns 404 Not Found")

        # Verify that after cancellation, 10:00 AM slot becomes available again!
        res_slots_after_cancel = client.get(f"/api/v1/doctors/{active_doctor_id}/availability?date={tomorrow}")
        assert res_slots_after_cancel.status_code == 200
        available_after_cancel = res_slots_after_cancel.json()["available_slots"]
        assert "10:00" in available_after_cancel, "Cancelled slot '10:00' must be available again!"
        print("  ✓ 10:00 AM slot is restored to available slots after cancellation")

    # Clean up test database tables
    Base.metadata.drop_all(bind=test_engine)

    # -----------------------------------------------------------------------
    # 10. Verify Development Database Was Untouched
    # -----------------------------------------------------------------------
    print("\n[TEST 10] Development Database Isolation Check")
    if dev_db_initial_stat is not None and os.path.exists(dev_db_path):
        current_stat = (os.path.getmtime(dev_db_path), os.path.getsize(dev_db_path))
        assert current_stat == dev_db_initial_stat, "ERROR: Development database (appointments.db) was modified!"
        print("  ✓ Confirmed: Development database (appointments.db) was NOT modified during test run.")
    else:
        print("  ✓ Test database operated in pure in-memory isolation.")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! (10/10 TEST SUITES VERIFIED)")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
