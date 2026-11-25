# MediVoice AI - Voice Doctor Appointment Assistant Backend

Production-ready backend foundation for **MediVoice AI - Voice AI Doctor Appointment Assistant** built with **FastAPI**, **SQLite**, **SQLAlchemy**, and **Pydantic**.

---

## Tech Stack

- **Python 3.11+**
- **FastAPI**: Modern, asynchronous web framework with OpenAPI / Swagger integration.
- **SQLite**: Embedded SQL database with zero-setup configuration.
- **SQLAlchemy 2.0**: Python SQL toolkit and ORM mapping.
- **Pydantic v2**: High-performance data validation and settings management.
- **Uvicorn**: High-performance ASGI production server.

---

## Features & Implementation

1. **Doctor Management**:
   - `GET /api/v1/doctors` – List doctors with filtering by specialty (`specialty` or `speciality`) and availability (`is_available`).
   - `GET /api/v1/doctors/{doctor_id}` – Retrieve doctor details.
   - `GET /api/v1/doctors/{doctor_id}/availability?date=YYYY-MM-DD` – Retrieve 30-minute appointment slots (09:00 - 17:00), excluding booked slots.

2. **Appointment Scheduling**:
   - `POST /api/v1/appointments` – Book an appointment. Validates doctor availability, working hours, future date, and prevents double-booking.
   - `GET /api/v1/appointments` – List all appointments with filtering by doctor, patient, status, or date.
   - `GET /api/v1/appointments/{appointment_id}` – Retrieve specific appointment details.
   - `GET /api/v1/appointments/by-phone/{phone}` – Retrieve appointments for a patient identified by phone number.
   - `PATCH /api/v1/appointments/{appointment_id}/cancel` – Cancel an appointment (re-opens the doctor's time slot).

3. **Smart Patient Identification**:
   - Identify patients using their phone number (`patient_phone`).
   - If the patient does not exist, automatically creates a patient record using supplied details (`patient_name`, `patient_email`, etc.).
   - If the patient already exists, links the booking directly to their account.

4. **Conflict & Double-Booking Prevention**:
   - Blocks double-booking the same doctor at the same date and time (`409 Conflict`).
   - Prevents overlapping appointment bookings within the 30-minute window.
   - Blocks booking unavailable doctors (`400 Bad Request`).
   - Blocks bookings in the past or outside clinic hours (`400 Bad Request`).

5. **Safe Database Seeding**:
   - Seeding script `seed.py` / `app/seed.py` creates realistic sample doctors across Cardiology, Dermatology, Pediatrics, Orthopedics, General Medicine, Neurology, and Psychiatry.
   - Fully idempotent: multiple executions will never create duplicate records.
   - Automatically executed during initial application startup.

6. **Frontend Integration**:
   - Pre-configured CORS allows the React frontend on `http://localhost:3000` to connect without CORS errors.

---

## Project Structure

```text
doc backend/
│
├── app/
│   ├── __init__.py               # Marks 'app' as a Python package
│   ├── main.py                   # FastAPI app instance, CORS middleware & lifespan events
│   ├── config.py                 # Application settings and environment variables
│   ├── database.py               # SQLite engine, SessionLocal & get_db dependency
│   ├── seed.py                   # Sample doctors data & idempotent seeding logic
│   │
│   ├── models/                   # SQLAlchemy ORM database models
│   │   ├── __init__.py           # Exports Base, Doctor, Patient, Appointment
│   │   ├── doctor.py             # Doctor database model
│   │   ├── patient.py            # Patient database model
│   │   └── appointment.py        # Appointment database model with relationships
│   │
│   ├── schemas/                  # Pydantic schemas for validation & serialization
│   │   ├── __init__.py           # Exports all schemas
│   │   ├── health.py             # HealthCheck response schema
│   │   ├── doctor.py             # DoctorBase, DoctorResponse, TimeSlot, DoctorAvailabilityResponse
│   │   ├── patient.py            # PatientBase, PatientCreate, PatientResponse
│   │   └── appointment.py        # AppointmentBase, AppointmentCreate, AppointmentResponse
│   │
│   └── api/                      # Versioned API routes
│       ├── __init__.py
│       └── v1/
│           ├── __init__.py
│           ├── router.py         # Consolidates v1 endpoints
│           └── endpoints/
│               ├── __init__.py
│               ├── health.py     # GET /health implementation
│               ├── doctors.py    # Doctor listing, details & slot availability
│               └── appointments.py # Booking, cancellation, by-phone lookup
│
├── seed.py                       # Standalone runner for database seeding
├── test_backend.py               # Comprehensive isolated in-memory test suite
├── .env                          # Local environment variables
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Project dependencies
└── README.md                     # Setup and documentation
```

---

## Windows Setup & Run Guide

### Step 1: Open Terminal in Project Directory
```powershell
cd "d:\doc backend"
```

### Step 2: Activate the Virtual Environment
**In PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```
*(If PowerShell execution policy error occurs, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

**In Command Prompt (CMD):**
```cmd
venv\Scripts\activate.bat
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Seed Sample Doctors (Optional / Automatic)
The application automatically seeds sample doctors on startup if none exist. You can also seed manually anytime:
```powershell
python seed.py
```

### Step 5: Run the Server
```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Interactive API documentation will be available at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## Running the Automated Test Suite

The test suite runs against an **isolated in-memory SQLite database** (`sqlite:///:memory:` with `StaticPool`), ensuring your real `appointments.db` is never modified during tests:

```powershell
python test_backend.py
```

To run bytecode validation:
```powershell
python -m compileall app
```
