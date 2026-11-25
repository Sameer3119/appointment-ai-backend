"""Pipecat 1.17.0 function-tool schemas and per-call handler registration."""

from typing import Any, Callable, Dict, Optional

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams, LLMService
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.services import voice_tool_service
from app.services.voice_tool_service import VoiceCallState


def build_appointment_tools() -> ToolsSchema:
    """Build the Pipecat ToolsSchema for MediVoice AI appointment tools.

    Note: Phone numbers and confirmation flags are intentionally excluded from
    Gemini tool arguments; identity is enforced per call via ``VoiceCallState``.
    """
    check_availability_schema = FunctionSchema(
        name="check_doctor_availability",
        description=(
            "Check available 30-minute appointment slots for a doctor or medical "
            "specialty on a specific date (YYYY-MM-DD)."
        ),
        properties={
            "date": {
                "type": "string",
                "description": "Target date in YYYY-MM-DD format.",
            },
            "doctor_id": {
                "type": ["integer", "null"],
                "description": "Optional ID of a specific doctor. Omit if not filtering by doctor ID.",
            },
            "specialty": {
                "type": ["string", "null"],
                "description": "Optional medical specialty (e.g., Cardiology, Dermatology). Omit if not specified.",
            },
        },
        required=["date"],
    )

    book_appointment_schema = FunctionSchema(
        name="book_appointment",
        description=(
            "Book an appointment for the caller after their phone number has been "
            "repeated back and verbally confirmed."
        ),
        properties={
            "doctor_id": {
                "type": "integer",
                "description": "ID of the doctor to book with.",
            },
            "appointment_datetime": {
                "type": "string",
                "description": "Appointment date and time in ISO format YYYY-MM-DDTHH:MM:SS.",
            },
            "patient_name": {
                "type": "string",
                "description": "Patient's full name.",
            },
            "email": {
                "type": ["string", "null"],
                "description": "Optional patient email address. Omit or pass empty string if not provided.",
            },
            "reason": {
                "type": ["string", "null"],
                "description": "Optional primary reason or symptoms for the visit. Omit or pass empty string if not provided.",
            },
        },
        required=["doctor_id", "appointment_datetime", "patient_name"],
    )

    cancel_appointment_schema = FunctionSchema(
        name="cancel_appointment",
        description=(
            "Cancel an existing appointment belonging to the caller after their "
            "phone number has been verbally confirmed."
        ),
        properties={
            "appointment_id": {
                "type": "integer",
                "description": "ID of the appointment to cancel.",
            },
        },
        required=["appointment_id"],
    )

    return ToolsSchema(
        standard_tools=[
            check_availability_schema,
            book_appointment_schema,
            cancel_appointment_schema,
        ]
    )


def _coerce_optional_int(value: Any, field_name: str) -> Optional[int]:
    """Coerce an optional tool argument to int or raise ValueError."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"'{field_name}' must be an integer.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field_name}' must be a valid integer.") from exc


def register_appointment_tools(
    llm: LLMService,
    call_state: VoiceCallState,
    *,
    session_factory: Callable[[], Session] = SessionLocal,
) -> ToolsSchema:
    """Register MediVoice AI appointment tool handlers on ``llm`` bound to ``call_state``.

    Each call gets its own ``VoiceCallState`` instance bound via closures so
    caller identity is never shared across calls and cannot be spoofed by LLM arguments.
    """
    if call_state is None:
        raise ValueError("call_state must be a per-call VoiceCallState instance.")

    async def handle_check_doctor_availability(params: FunctionCallParams) -> None:
        args: Dict[str, Any] = dict(params.arguments or {})
        try:
            date_val = args.get("date")
            if not isinstance(date_val, str) or not date_val.strip():
                await params.result_callback(
                    {"success": False, "error": "Please provide a valid date in YYYY-MM-DD format."}
                )
                return

            doctor_id = _coerce_optional_int(args.get("doctor_id"), "doctor_id")
            specialty_val = args.get("specialty")
            specialty = str(specialty_val).strip() or None if specialty_val is not None else None

            result = await voice_tool_service.check_doctor_availability(
                date=date_val.strip(),
                doctor_id=doctor_id,
                specialty=specialty,
                session_factory=session_factory,
            )
            await params.result_callback(result)
        except ValueError as exc:
            await params.result_callback({"success": False, "error": str(exc)})
        except Exception as exc:
            await params.result_callback(
                {"success": False, "error": f"Unable to check availability right now: {exc}"}
            )

    async def handle_book_appointment(params: FunctionCallParams) -> None:
        args: Dict[str, Any] = dict(params.arguments or {})
        try:
            doctor_id = _coerce_optional_int(args.get("doctor_id"), "doctor_id")
            if doctor_id is None:
                await params.result_callback(
                    {"success": False, "error": "A valid doctor_id is required to book."}
                )
                return

            appt_dt = args.get("appointment_datetime")
            if not isinstance(appt_dt, str) or not appt_dt.strip():
                await params.result_callback(
                    {
                        "success": False,
                        "error": "appointment_datetime is required in YYYY-MM-DDTHH:MM:SS format.",
                    }
                )
                return

            patient_name = args.get("patient_name")
            if not isinstance(patient_name, str) or not patient_name.strip():
                await params.result_callback(
                    {"success": False, "error": "patient_name is required to book."}
                )
                return

            email_val = args.get("email")
            email = str(email_val).strip() or None if email_val is not None else None
            reason_val = args.get("reason")
            reason = str(reason_val).strip() or None if reason_val is not None else None

            result = await voice_tool_service.book_appointment(
                call_state=call_state,
                doctor_id=doctor_id,
                appointment_datetime=appt_dt.strip(),
                patient_name=patient_name.strip(),
                patient_email=email,
                reason=reason,
                session_factory=session_factory,
            )
            await params.result_callback(result)
        except ValueError as exc:
            await params.result_callback({"success": False, "error": str(exc)})
        except Exception as exc:
            await params.result_callback(
                {"success": False, "error": f"Unable to book appointment right now: {exc}"}
            )

    async def handle_cancel_appointment(params: FunctionCallParams) -> None:
        args: Dict[str, Any] = dict(params.arguments or {})
        try:
            appointment_id = _coerce_optional_int(args.get("appointment_id"), "appointment_id")
            if appointment_id is None:
                await params.result_callback(
                    {"success": False, "error": "A valid appointment_id is required to cancel."}
                )
                return

            result = await voice_tool_service.cancel_appointment(
                call_state=call_state,
                appointment_id=appointment_id,
                session_factory=session_factory,
            )
            await params.result_callback(result)
        except ValueError as exc:
            await params.result_callback({"success": False, "error": str(exc)})
        except Exception as exc:
            await params.result_callback(
                {"success": False, "error": f"Unable to cancel appointment right now: {exc}"}
            )

    llm.register_function("check_doctor_availability", handle_check_doctor_availability)
    llm.register_function("book_appointment", handle_book_appointment)
    llm.register_function("cancel_appointment", handle_cancel_appointment)

    return build_appointment_tools()
