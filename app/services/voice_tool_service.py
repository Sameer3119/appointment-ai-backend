"""Business-logic layer for MediVoice AI's Pipecat voice assistant tools.

Wraps the existing synchronous ``doctor_service`` and ``appointment_service``
functions for safe asynchronous execution inside a voice call, enforcing
per-call phone confirmation and normalized phone ownership checks.
"""

import asyncio
from dataclasses import dataclass
from datetime import date as date_type, datetime
import re
from typing import Any, Callable, Dict, Optional

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.patient import Patient
from app.schemas.appointment import AppointmentCreate
from app.services import appointment_service, doctor_service


def normalize_phone(phone: str) -> str:
    """Normalize a phone number by removing formatting while preserving a leading '+'.

    Strips spaces, hyphens, parentheses, dots, and other non-digit characters.
    Does not infer or alter country codes; ``+15551234567`` and ``5551234567``
    remain distinct.

    Raises:
        ValueError: If the phone string is empty or contains fewer than 3 digits.
    """
    if not isinstance(phone, str):
        raise ValueError("Phone number must be a string.")

    stripped = phone.strip()
    if not stripped:
        raise ValueError("Phone number cannot be empty.")

    has_leading_plus = stripped.startswith("+")
    digits = re.sub(r"\D", "", stripped)

    if len(digits) < 3:
        raise ValueError("A valid phone number with at least 3 digits is required.")

    return f"+{digits}" if has_leading_plus else digits


@dataclass
class VoiceCallState:
    """Per-call state holding pending and confirmed caller phone identity.

    Instantiated once per voice call (never shared globally). Only the
    application/voice pipeline layer may promote or set ``confirmed_phone``
    after explicit verbal confirmation from the patient.
    """

    pending_phone: Optional[str] = None
    confirmed_phone: Optional[str] = None

    def set_pending_phone(self, phone: str) -> str:
        """Normalize and store a candidate phone number awaiting confirmation."""
        normalized = normalize_phone(phone)
        if normalized != self.confirmed_phone:
            self.confirmed_phone = None
        self.pending_phone = normalized
        return normalized

    def confirm_pending_phone(self) -> str:
        """Promote the pending phone number to confirmed after explicit user confirmation."""
        if not self.pending_phone:
            raise ValueError("No pending phone number is available to confirm.")
        self.confirmed_phone = self.pending_phone
        return self.confirmed_phone

    def confirm_phone(self, phone: str) -> str:
        """Directly record a normalized, pipeline-confirmed phone number."""
        normalized = normalize_phone(phone)
        self.pending_phone = normalized
        self.confirmed_phone = normalized
        return normalized

    def clear(self) -> None:
        """Reset pending and confirmed phone state."""
        self.pending_phone = None
        self.confirmed_phone = None

    @property
    def has_confirmed_phone(self) -> bool:
        """Return True when a confirmed phone number is present for this call."""
        return bool(self.confirmed_phone)


def _find_patient_by_normalized_phone(db: Session, normalized_phone: str) -> Optional[Patient]:
    """Find an existing patient whose stored phone matches ``normalized_phone``.

    Preserves the patient's original stored ``phone`` value without modification.
    """
    exact = db.query(Patient).filter(Patient.phone == normalized_phone).first()
    if exact:
        return exact

    for patient in db.query(Patient).all():
        try:
            if normalize_phone(patient.phone) == normalized_phone:
                return patient
        except ValueError:
            continue
    return None


def _check_doctor_availability_sync(
    session_factory: Callable[[], Session],
    date_str: str,
    doctor_id: Optional[int] = None,
    specialty: Optional[str] = None,
) -> Dict[str, Any]:
    """Synchronous worker for checking doctor availability."""
    try:
        target_date = date_type.fromisoformat(date_str.strip())
    except (ValueError, AttributeError):
        return {
            "success": False,
            "error": "Invalid date format. Please provide date as YYYY-MM-DD.",
        }

    db = session_factory()
    try:
        if doctor_id is not None:
            availability = doctor_service.get_doctor_availability(db, doctor_id, target_date)
            return {
                "success": True,
                "date": availability.date,
                "doctors": [
                    {
                        "doctor_id": availability.doctor_id,
                        "doctor_name": availability.doctor_name,
                        "specialty": availability.specialty,
                        "available_slots": availability.available_slots,
                    }
                ],
            }

        doctors = doctor_service.list_doctors(db, specialty=specialty, is_available=True)
        if not doctors:
            return {
                "success": False,
                "error": (
                    f"No available doctors found for specialty '{specialty}'."
                    if specialty
                    else "No available doctors found."
                ),
            }

        results = []
        for doc in doctors:
            availability = doctor_service.get_doctor_availability(db, doc.id, target_date)
            results.append(
                {
                    "doctor_id": availability.doctor_id,
                    "doctor_name": availability.doctor_name,
                    "specialty": availability.specialty,
                    "available_slots": availability.available_slots,
                }
            )
        return {
            "success": True,
            "date": target_date.isoformat(),
            "doctors": results,
        }
    except HTTPException as exc:
        return {"success": False, "status_code": exc.status_code, "error": exc.detail}
    finally:
        db.close()


async def check_doctor_availability(
    date: str,
    doctor_id: Optional[int] = None,
    specialty: Optional[str] = None,
    session_factory: Callable[[], Session] = SessionLocal,
) -> Dict[str, Any]:
    """Check available 30-minute appointment slots for a doctor or specialty on a date."""
    return await asyncio.to_thread(
        _check_doctor_availability_sync,
        session_factory,
        date,
        doctor_id,
        specialty,
    )


def _book_appointment_sync(
    session_factory: Callable[[], Session],
    call_state: VoiceCallState,
    doctor_id: int,
    appointment_datetime: str,
    patient_name: str,
    patient_email: Optional[str] = None,
    reason: Optional[str] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """Synchronous worker for booking an appointment using a confirmed call phone."""
    if not call_state or not call_state.confirmed_phone:
        return {
            "success": False,
            "error": (
                "Cannot book appointment: patient phone number has not been verbally "
                "confirmed in this call."
            ),
        }

    if not patient_name or not patient_name.strip():
        return {
            "success": False,
            "error": "Patient name is required to book an appointment.",
        }

    try:
        confirmed_normalized = normalize_phone(call_state.confirmed_phone)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    try:
        appt_dt = datetime.fromisoformat(appointment_datetime.strip())
    except (ValueError, AttributeError):
        return {
            "success": False,
            "error": "Invalid appointment_datetime format. Use ISO format YYYY-MM-DDTHH:MM:SS.",
        }

    db = session_factory()
    try:
        clean_email = patient_email.strip() or None if patient_email else None
        clean_reason = reason.strip() or None if reason else None
        clean_notes = notes.strip() or None if notes else None

        existing_patient = _find_patient_by_normalized_phone(db, confirmed_normalized)
        if existing_patient is not None:
            payload = AppointmentCreate(
                doctor_id=doctor_id,
                appointment_date=appt_dt,
                patient_id=existing_patient.id,
                reason=clean_reason,
                notes=clean_notes,
            )
        else:
            payload = AppointmentCreate(
                doctor_id=doctor_id,
                appointment_date=appt_dt,
                patient_phone=confirmed_normalized,
                patient_name=patient_name.strip(),
                patient_email=clean_email,
                reason=clean_reason,
                notes=clean_notes,
            )

        appointment = appointment_service.book_appointment(db, payload)
        return {
            "success": True,
            "message": (
                f"Appointment #{appointment.id} booked with "
                f"{appointment.doctor.name if appointment.doctor else f'Doctor #{doctor_id}'} "
                f"on {appointment.appointment_date.strftime('%Y-%m-%d at %H:%M')}."
            ),
            "appointment": {
                "id": appointment.id,
                "doctor_id": appointment.doctor_id,
                "doctor_name": appointment.doctor.name if appointment.doctor else None,
                "patient_id": appointment.patient_id,
                "patient_name": appointment.patient.name if appointment.patient else patient_name.strip(),
                "appointment_date": appointment.appointment_date.isoformat(),
                "status": appointment.status,
            },
        }
    except HTTPException as exc:
        return {"success": False, "status_code": exc.status_code, "error": exc.detail}
    except ValidationError as exc:
        return {"success": False, "error": str(exc)}
    finally:
        db.close()


async def book_appointment(
    call_state: VoiceCallState,
    doctor_id: int,
    appointment_datetime: str,
    patient_name: str,
    patient_email: Optional[str] = None,
    reason: Optional[str] = None,
    notes: Optional[str] = None,
    session_factory: Callable[[], Session] = SessionLocal,
) -> Dict[str, Any]:
    """Book an appointment after verifying the call state has a confirmed phone number."""
    return await asyncio.to_thread(
        _book_appointment_sync,
        session_factory,
        call_state,
        doctor_id,
        appointment_datetime,
        patient_name,
        patient_email,
        reason,
        notes,
    )


def _cancel_appointment_sync(
    session_factory: Callable[[], Session],
    call_state: VoiceCallState,
    appointment_id: int,
) -> Dict[str, Any]:
    """Synchronous worker for cancelling an appointment after verifying ownership."""
    if not call_state or not call_state.confirmed_phone:
        return {
            "success": False,
            "error": (
                "Cannot cancel appointment: patient phone number has not been verbally "
                "confirmed in this call."
            ),
        }

    try:
        confirmed_normalized = normalize_phone(call_state.confirmed_phone)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    db = session_factory()
    try:
        appointment = appointment_service.get_appointment(db, appointment_id)
        if not appointment.patient or not appointment.patient.phone:
            return {
                "success": False,
                "error": "Unable to verify appointment ownership.",
            }

        owner_normalized = normalize_phone(appointment.patient.phone)
        if owner_normalized != confirmed_normalized:
            return {
                "success": False,
                "status_code": 403,
                "error": (
                    f"Appointment #{appointment_id} does not belong to the confirmed "
                    "patient phone number and cannot be cancelled."
                ),
            }

        cancelled = appointment_service.cancel_appointment(db, appointment_id)
        return {
            "success": True,
            "message": (
                f"Appointment #{cancelled.id} on "
                f"{cancelled.appointment_date.strftime('%Y-%m-%d at %H:%M')} has been cancelled."
            ),
            "appointment": {
                "id": cancelled.id,
                "doctor_id": cancelled.doctor_id,
                "patient_id": cancelled.patient_id,
                "appointment_date": cancelled.appointment_date.isoformat(),
                "status": cancelled.status,
            },
        }
    except HTTPException as exc:
        return {"success": False, "status_code": exc.status_code, "error": exc.detail}
    finally:
        db.close()


async def cancel_appointment(
    call_state: VoiceCallState,
    appointment_id: int,
    session_factory: Callable[[], Session] = SessionLocal,
) -> Dict[str, Any]:
    """Cancel an appointment after verifying confirmed phone ownership."""
    return await asyncio.to_thread(
        _cancel_appointment_sync,
        session_factory,
        call_state,
        appointment_id,
    )
