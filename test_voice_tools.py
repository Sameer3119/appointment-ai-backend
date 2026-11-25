"""Isolated test suite for app/services/voice_tool_service.py.

Uses an in-memory SQLite database so development appointments.db is untouched.
"""

import asyncio
from datetime import datetime, timedelta
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from aiortc.rtcicetransport import connection_kwargs
from fastapi.testclient import TestClient
import httpx
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.turns.user_start import (
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import (
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)

from app.config import Settings
from app.database import Base
from app.main import app as fastapi_app
from app.models.appointment import Appointment
from app.models.doctor import Doctor
from app.models.patient import Patient
from app.seed import seed_doctors
from app.services.voice_tool_service import (
    VoiceCallState,
    book_appointment,
    cancel_appointment,
    check_doctor_availability,
    normalize_phone,
)
from app.voice.ice import (
    MeteredTurnError,
    fetch_ice_servers,
    prioritize_ice_servers,
    to_rtc_ice_servers,
)
from app.voice.phone_confirmation import PhoneConfirmationTracker
from app.voice.pipeline import (
    VoiceConfigurationError,
    build_voice_pipeline,
    validate_voice_config,
)
from app.voice.tools import register_appointment_tools


TEST_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


class TrackingSessionFactory:
    """Wraps a sessionmaker to verify every created session is closed."""

    def __init__(self, inner_factory):
        self._inner_factory = inner_factory
        self.created_count = 0
        self.closed_count = 0

    def __call__(self):
        session = self._inner_factory()
        self.created_count += 1
        original_close = session.close

        def tracked_close():
            self.closed_count += 1
            return original_close()

        session.close = tracked_close
        return session


async def run_voice_tool_tests():
    print("=" * 70)
    print("MEDIVOICE AI VOICE TOOL SERVICE TEST SUITE (IN-MEMORY DB)")
    print("=" * 70)

    dev_db_path = "appointments.db"
    dev_db_initial_stat = None
    if os.path.exists(dev_db_path):
        dev_db_initial_stat = (os.path.getmtime(dev_db_path), os.path.getsize(dev_db_path))

    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)

    with TestingSessionLocal() as db:
        seed_doctors(db)
        first_doc = db.query(Doctor).filter(Doctor.is_available == True).first()  # noqa: E712
        active_doctor_id = first_doc.id
        # Pre-seed a patient with formatted phone to verify preservation
        formatted_patient = Patient(name="Alice Formatted", phone="+1-555-111-2222")
        db.add(formatted_patient)
        db.commit()
        db.refresh(formatted_patient)
        formatted_patient_id = formatted_patient.id

    session_tracker = TrackingSessionFactory(TestingSessionLocal)
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    # 1. Phone normalization tests
    print("\n[TEST 1] Phone Normalization")
    assert normalize_phone(" +1 (555) 111-2222 ") == "+15551112222"
    assert normalize_phone("555-111-2222") == "5551112222"
    assert normalize_phone("+15551112222") != normalize_phone("5551112222")
    assert normalize_phone("+15551112222") != normalize_phone("+445551112222")
    print("  ✓ Formatting stripped, leading '+' preserved, country codes kept distinct")

    # 2. Availability lookup
    print("\n[TEST 2] Availability Lookup")
    avail_by_id = await check_doctor_availability(
        date=tomorrow,
        doctor_id=active_doctor_id,
        session_factory=session_tracker,
    )
    assert avail_by_id["success"] is True, avail_by_id
    assert len(avail_by_id["doctors"]) == 1
    assert "10:00" in avail_by_id["doctors"][0]["available_slots"]

    avail_by_spec = await check_doctor_availability(
        date=tomorrow,
        specialty="Cardiology",
        session_factory=session_tracker,
    )
    assert avail_by_spec["success"] is True, avail_by_spec
    assert len(avail_by_spec["doctors"]) >= 1
    print("  ✓ check_doctor_availability works by doctor_id and specialty")

    # 3. Booking without confirmed phone is rejected
    print("\n[TEST 3] Reject Booking Without Confirmed Phone")
    unconfirmed_state = VoiceCallState()
    res_no_phone = await book_appointment(
        call_state=unconfirmed_state,
        doctor_id=active_doctor_id,
        appointment_datetime=f"{tomorrow}T10:00:00",
        patient_name="Bob Unconfirmed",
        session_factory=session_tracker,
    )
    assert res_no_phone["success"] is False
    assert "confirmed" in res_no_phone["error"].lower()

    unconfirmed_state.set_pending_phone("+1-555-333-4444")
    res_pending_only = await book_appointment(
        call_state=unconfirmed_state,
        doctor_id=active_doctor_id,
        appointment_datetime=f"{tomorrow}T10:00:00",
        patient_name="Bob Unconfirmed",
        session_factory=session_tracker,
    )
    assert res_pending_only["success"] is False
    assert "confirmed" in res_pending_only["error"].lower()
    print("  ✓ Booking rejected when phone is missing or only pending")

    # 4. Booking with confirmed phone succeeds & preserves existing formatted phone
    print("\n[TEST 4] Booking With Confirmed Phone")
    alice_state = VoiceCallState()
    alice_state.set_pending_phone("+1 (555) 111-2222")
    alice_state.confirm_pending_phone()

    res_book = await book_appointment(
        call_state=alice_state,
        doctor_id=active_doctor_id,
        appointment_datetime=f"{tomorrow}T10:00:00",
        patient_name="Alice Formatted",
        reason="Follow-up consultation",
        session_factory=session_tracker,
    )
    assert res_book["success"] is True, res_book
    appt_id = res_book["appointment"]["id"]
    assert res_book["appointment"]["patient_id"] == formatted_patient_id

    with TestingSessionLocal() as db:
        reloaded_alice = db.query(Patient).filter(Patient.id == formatted_patient_id).first()
        assert reloaded_alice.phone == "+1-555-111-2222"
    print(f"  ✓ Booked appointment #{appt_id} and preserved original stored phone format")

    # 5. Cancellation by another patient's phone is rejected
    print("\n[TEST 5] Reject Cancellation by Different Patient")
    other_state = VoiceCallState()
    other_state.confirm_phone("+1-555-999-8888")

    res_wrong_owner = await cancel_appointment(
        call_state=other_state,
        appointment_id=appt_id,
        session_factory=session_tracker,
    )
    assert res_wrong_owner["success"] is False
    assert res_wrong_owner.get("status_code") == 403

    with TestingSessionLocal() as db:
        still_active = db.query(Appointment).filter(Appointment.id == appt_id).first()
        assert still_active.status == "scheduled"
    print("  ✓ Cancellation by different caller rejected with 403; appointment remains scheduled")

    # 6. Cancellation by the correct patient succeeds (despite different phone formatting)
    print("\n[TEST 6] Cancellation by Correct Patient")
    res_cancel = await cancel_appointment(
        call_state=alice_state,
        appointment_id=appt_id,
        session_factory=session_tracker,
    )
    assert res_cancel["success"] is True, res_cancel
    assert res_cancel["appointment"]["status"] == "cancelled"
    print(f"  ✓ Appointment #{appt_id} cancelled by confirmed owner")

    # 8. Pipecat Tool Registration & Per-Call Closure Isolation
    print("\n[TEST 8] Pipecat Tool Registration & Per-Call Closure Isolation")
    llm_call_a = GroqLLMService(api_key="test-key-unused")
    state_call_a = VoiceCallState()
    tools_a = register_appointment_tools(
        llm_call_a, state_call_a, session_factory=session_tracker
    )

    tool_names = [t.name for t in tools_a.standard_tools]
    assert tool_names == [
        "check_doctor_availability",
        "book_appointment",
        "cancel_appointment",
    ]
    for name in tool_names:
        assert llm_call_a.has_function(name), f"Missing registered function: {name}"

    schema_map = {t.name: t for t in tools_a.standard_tools}
    assert set(schema_map["check_doctor_availability"].properties.keys()) == {
        "date",
        "doctor_id",
        "specialty",
    }
    assert set(schema_map["book_appointment"].properties.keys()) == {
        "doctor_id",
        "appointment_datetime",
        "patient_name",
        "email",
        "reason",
    }
    assert set(schema_map["cancel_appointment"].properties.keys()) == {"appointment_id"}
    for schema in schema_map.values():
        assert "phone_confirmed" not in schema.properties
        assert "patient_phone" not in schema.properties

    dummy_ctx = LLMContext(tools=tools_a)

    async def invoke_registered(llm_svc, fn_name, arguments):
        captured = []

        async def _cb(result, *, properties=None):
            captured.append(result)

        handler = llm_svc._functions[fn_name].handler
        params = FunctionCallParams(
            function_name=fn_name,
            tool_call_id="call-1",
            arguments=arguments,
            llm=llm_svc,
            pipeline_worker=None,
            context=dummy_ctx,
            result_callback=_cb,
        )
        await handler(params)
        assert len(captured) == 1
        return captured[0]

    # Availability via registered handler
    reg_avail = await invoke_registered(
        llm_call_a,
        "check_doctor_availability",
        {"date": tomorrow, "doctor_id": active_doctor_id},
    )
    assert reg_avail["success"] is True

    # Invalid arg handling
    reg_bad_arg = await invoke_registered(
        llm_call_a,
        "book_appointment",
        {
            "doctor_id": "not-an-int",
            "appointment_datetime": f"{tomorrow}T11:00:00",
            "patient_name": "Test User",
        },
    )
    assert reg_bad_arg["success"] is False

    # Cannot bypass phone confirmation with extra LLM argument
    reg_bypass = await invoke_registered(
        llm_call_a,
        "book_appointment",
        {
            "doctor_id": active_doctor_id,
            "appointment_datetime": f"{tomorrow}T11:00:00",
            "patient_name": "Bypass Attempt",
            "phone_confirmed": True,
        },
    )
    assert reg_bypass["success"] is False
    assert "confirmed" in reg_bypass["error"].lower()

    # Second call with its own confirmed state does not leak to call A
    llm_call_b = GroqLLMService(api_key="test-key-unused")
    state_call_b = VoiceCallState()
    state_call_b.confirm_phone("+1-555-777-8888")
    register_appointment_tools(llm_call_b, state_call_b, session_factory=session_tracker)

    reg_still_unconfirmed = await invoke_registered(
        llm_call_a,
        "book_appointment",
        {
            "doctor_id": active_doctor_id,
            "appointment_datetime": f"{tomorrow}T11:00:00",
            "patient_name": "Call A Unconfirmed",
        },
    )
    assert reg_still_unconfirmed["success"] is False

    reg_book_b = await invoke_registered(
        llm_call_b,
        "book_appointment",
        {
            "doctor_id": active_doctor_id,
            "appointment_datetime": f"{tomorrow}T11:00:00",
            "patient_name": "Call B Confirmed",
            "email": "callb@example.com",
            "reason": "Checkup",
        },
    )
    assert reg_book_b["success"] is True, reg_book_b
    appt_b_id = reg_book_b["appointment"]["id"]

    assert schema_map["book_appointment"].properties["email"]["type"] == ["string", "null"]
    assert schema_map["book_appointment"].properties["reason"]["type"] == ["string", "null"]
    assert schema_map["check_doctor_availability"].properties["doctor_id"]["type"] == ["integer", "null"]
    assert schema_map["check_doctor_availability"].properties["specialty"]["type"] == ["string", "null"]

    reg_cancel_b = await invoke_registered(
        llm_call_b,
        "cancel_appointment",
        {"appointment_id": appt_b_id},
    )
    assert reg_cancel_b["success"] is True, reg_cancel_b

    # Booking with omitted email & reason succeeds
    reg_book_omitted = await invoke_registered(
        llm_call_b,
        "book_appointment",
        {
            "doctor_id": active_doctor_id,
            "appointment_datetime": f"{tomorrow}T11:30:00",
            "patient_name": "Omitted Fields Patient",
        },
    )
    assert reg_book_omitted["success"] is True, reg_book_omitted

    # Booking with empty-string and null email & reason on a new patient converts to None and succeeds
    llm_call_c = GroqLLMService(api_key="test-key-unused")
    state_call_c = VoiceCallState()
    state_call_c.confirm_phone("+1-555-777-9999")
    register_appointment_tools(llm_call_c, state_call_c, session_factory=session_tracker)

    reg_book_nulls = await invoke_registered(
        llm_call_c,
        "book_appointment",
        {
            "doctor_id": active_doctor_id,
            "appointment_datetime": f"{tomorrow}T14:00:00",
            "patient_name": "Null Fields Patient",
            "email": "   ",
            "reason": None,
        },
    )
    assert reg_book_nulls["success"] is True, reg_book_nulls
    with TestingSessionLocal() as db:
        saved_null_appt = db.query(Appointment).filter(Appointment.id == reg_book_nulls["appointment"]["id"]).first()
        assert saved_null_appt.reason is None
        assert saved_null_appt.patient.email is None

    # Invalid booking (outside clinic hours) is still rejected
    reg_book_invalid = await invoke_registered(
        llm_call_b,
        "book_appointment",
        {
            "doctor_id": active_doctor_id,
            "appointment_datetime": f"{tomorrow}T22:00:00",
            "patient_name": "After Hours Patient",
        },
    )
    assert reg_book_invalid["success"] is False
    assert reg_book_invalid.get("status_code") == 400
    print("  ✓ Registered schemas, closure isolation, optional null/empty fields, and safeguards verified")

    # 9. Phone Confirmation State Machine Safeguards
    print("\n[TEST 9] Phone Confirmation State Machine Safeguards")
    seq_state = VoiceCallState()
    tracker = PhoneConfirmationTracker(seq_state)

    # ISO date must not be captured as a phone number
    tracker.process_user_transcript(f"I want an appointment on {tomorrow}")
    assert seq_state.pending_phone is None
    assert seq_state.confirmed_phone is None

    # Candidate phone captured -> still unconfirmed
    tracker.process_user_transcript("My phone number is 555-234-5678")
    assert seq_state.pending_phone == "5552345678"
    assert seq_state.confirmed_phone is None
    assert tracker.readback_completed is False

    # Premature "yes" before assistant readback must NOT confirm
    tracker.process_user_transcript("yes")
    assert seq_state.confirmed_phone is None

    # Assistant reads back the number
    tracker.append_assistant_text("Let me confirm, is your number 555-234-5678?")
    tracker.complete_assistant_turn()
    assert tracker.readback_completed is True
    assert seq_state.confirmed_phone is None

    # Trick phrase containing "yes" AND "no"/"isn't" must REJECT and clear
    tracker.process_user_transcript("no, yes isn't correct")
    assert seq_state.pending_phone is None
    assert seq_state.confirmed_phone is None

    # Re-capture, read back, ambiguous phrase -> stays unconfirmed, then explicit confirm
    tracker.process_user_transcript("Actually it is +1 (555) 234-5679")
    tracker.append_assistant_text("Got it, +1 555-234-5679. Is that right?")
    tracker.complete_assistant_turn()
    tracker.process_user_transcript("uh maybe let me check")
    assert seq_state.confirmed_phone is None
    tracker.process_user_transcript("Yes, that's correct!")
    assert seq_state.confirmed_phone == "+15552345679"
    print("  ✓ Phone confirmation enforces capture -> readback -> strict confirmation")

    # 10. Missing Voice Config, Pipeline Creation, Per-Call Isolation & Cleanup
    print("\n[TEST 10] Voice Config, Pipeline Creation, Isolation & Cleanup")
    missing_cfg = Settings(
        _env_file=None,
        GROQ_API_KEY=None,
        ELEVENLABS_API_KEY=None,
        ELEVENLABS_VOICE_ID=None,
    )
    try:
        validate_voice_config(missing_cfg)
        raise AssertionError("Expected VoiceConfigurationError for missing config")
    except VoiceConfigurationError as exc:
        assert "GROQ_API_KEY" in str(exc)
        assert "ELEVENLABS_API_KEY" in str(exc)
        assert "ELEVENLABS_VOICE_ID" in str(exc)

    dummy_cfg = Settings(
        _env_file=None,
        GROQ_API_KEY="dummy-groq-key",
        ELEVENLABS_API_KEY="dummy-el-key",
        ELEVENLABS_VOICE_ID="dummy-voice-id",
    )
    conn_1 = SmallWebRTCConnection()
    conn_2 = SmallWebRTCConnection()
    task_1, transport_1, state_1, _ = build_voice_pipeline(conn_1, dummy_cfg)
    task_2, transport_2, state_2, _ = build_voice_pipeline(conn_2, dummy_cfg)

    assert state_1 is not state_2
    assert task_1 is not task_2
    assert transport_1 is not transport_2
    state_1.confirm_phone("+15550001111")
    assert state_2.confirmed_phone is None

    # Verify explicit user-turn stop strategy and VAD stop_secs on user_aggregator
    def _iter_processors(root):
        for proc in getattr(root, "processors", []):
            yield proc
            yield from _iter_processors(proc)

    user_aggregators = [
        proc for proc in _iter_processors(task_1.pipeline) if isinstance(proc, LLMUserAggregator)
    ]
    assert len(user_aggregators) == 1
    user_agg = user_aggregators[0]
    vad_params = user_agg._params.vad_analyzer.params
    assert vad_params.confidence == 0.8
    assert vad_params.start_secs == 0.3
    assert vad_params.stop_secs == 0.3
    assert vad_params.min_volume == 0.65
    assert user_agg._params.user_mute_strategies == []
    strategies = user_agg._user_turn_controller.user_turn_strategies
    assert [type(s) for s in strategies.start] == [
        VADUserTurnStartStrategy,
        TranscriptionUserTurnStartStrategy,
    ]
    assert strategies.start[1]._use_interim is False
    assert [type(s) for s in strategies.stop] == [SpeechTimeoutUserTurnStopStrategy]
    assert not any(isinstance(s, TurnAnalyzerUserTurnStopStrategy) for s in strategies.stop)
    assert strategies.stop[0]._user_speech_timeout == 0.8

    # Verify ElevenLabs STT background noise filtering settings
    from pipecat.services.elevenlabs.stt import CommitStrategy, ElevenLabsRealtimeSTTService

    stt_services = [
        p for p in _iter_processors(task_1.pipeline) if isinstance(p, ElevenLabsRealtimeSTTService)
    ]
    assert len(stt_services) == 1
    assert stt_services[0]._commit_strategy == CommitStrategy.MANUAL
    assert stt_services[0]._settings.filter_background_audio is True
    assert stt_services[0]._settings.no_verbatim is True

    # Verify low-latency Groq, ElevenLabs TTS, and transport chunking settings
    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

    groq_services = [p for p in _iter_processors(task_1.pipeline) if isinstance(p, GroqLLMService)]
    tts_services = [p for p in _iter_processors(task_1.pipeline) if isinstance(p, ElevenLabsTTSService)]
    assert len(groq_services) == 1
    assert groq_services[0]._settings.reasoning_effort == "low"
    assert len(tts_services) == 1
    assert tts_services[0]._auto_mode is True
    assert transport_1._params.audio_out_10ms_chunks == 2

    # Verify privacy-safe speech timing hooks are attached without leaking text or secrets
    from pipecat.frames.frames import LLMTextFrame, OutputAudioRawFrame
    from app.voice.speech_timing import attach_speech_timing_logs

    timing_logs = []
    from loguru import logger as loguru_logger
    sink_id = loguru_logger.add(lambda m: timing_logs.append(str(m)), level="DEBUG")
    try:
        stats = attach_speech_timing_logs(groq_services[0], tts_services[0], transport_1)
        await groq_services[0]._push_llm_text("SecretPatientName 555-123-4567")
        await transport_1.output().write_audio_frame(
            OutputAudioRawFrame(audio=b"\x00" * 960, sample_rate=24000, num_channels=1)
        )
        assert stats.llm_chunks == 1
        assert stats.transport_out_frames == 1
        joined_logs = "\n".join(timing_logs)
        assert "SecretPatientName" not in joined_logs
        assert "555-123-4567" not in joined_logs
        assert "dummy-groq-key" not in joined_logs
        assert "dummy-el-key" not in joined_logs
    finally:
        loguru_logger.remove(sink_id)

    await task_1.cancel()
    await task_2.cancel()

    # Verify FastAPI lifespan creates & closes shared aiohttp session and /voice/offer rejects when unconfigured
    from app.api.v1.endpoints import voice as voice_endpoint

    orig_validate = voice_endpoint.validate_voice_config

    def _raise_missing():
        raise VoiceConfigurationError("Missing voice config")

    voice_endpoint.validate_voice_config = _raise_missing
    try:
        with TestClient(fastapi_app) as client:
            assert hasattr(fastapi_app.state, "aiohttp_session")
            assert fastapi_app.state.aiohttp_session.closed is False
            res_offer = client.post(
                "/api/v1/voice/offer",
                json={"sdp": "v=0", "type": "offer"},
            )
            assert res_offer.status_code == 503
            res_ice = client.get("/api/v1/voice/ice-config")
            assert res_ice.status_code == 200
            assert "iceServers" in res_ice.json()
        assert fastapi_app.state.aiohttp_session.closed is True
    finally:
        voice_endpoint.validate_voice_config = orig_validate
    print("  ✓ Missing config returns 503, pipelines are isolated, and resources close cleanly")

    # 11. Metered.ca TURN Prioritization & Fallback
    print("\n[TEST 11] Metered.ca TURN Prioritization & Fallback")
    sample_metered = [
        {"urls": "stun:stun.relay.metered.ca:80"},
        {"urls": "turn:global.relay.metered.ca:80", "username": "u1", "credential": "p1"},
        {"urls": "turn:global.relay.metered.ca:80?transport=tcp", "username": "u1", "credential": "p1"},
        {"urls": "turn:global.relay.metered.ca:443", "username": "u1", "credential": "p1"},
        {"urls": "turns:global.relay.metered.ca:443?transport=tcp", "username": "u1", "credential": "p1"},
    ]
    ordered = prioritize_ice_servers(sample_metered)
    turn_entries = [s for s in ordered if str(s["urls"]).startswith("turn")]
    assert turn_entries[0]["urls"] == "turns:global.relay.metered.ca:443?transport=tcp"

    aiortc_kwargs = connection_kwargs(to_rtc_ice_servers(ordered))
    assert aiortc_kwargs["turn_server"] == ("global.relay.metered.ca", 443)
    assert aiortc_kwargs["turn_ssl"] is True
    assert aiortc_kwargs["turn_transport"] == "tcp"

    # Mocked Metered HTTP fetch
    def _mock_metered_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "doctor_appointment_ai.metered.live"
        assert request.url.path == "/api/v1/turn/credentials"
        assert request.url.params.get("apiKey") == "mock-metered-key"
        return httpx.Response(200, json=sample_metered)

    metered_cfg = Settings(
        _env_file=None,
        METERED_DOMAIN="doctor_appointment_ai.metered.live",
        METERED_API_KEY="mock-metered-key",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(_mock_metered_handler)) as mock_client:
        fetched = await fetch_ice_servers(metered_cfg, http_client=mock_client)
    assert [s for s in fetched if str(s["urls"]).startswith("turn")][0]["urls"] == (
        "turns:global.relay.metered.ca:443?transport=tcp"
    )

    # Configured Metered failure raises MeteredTurnError instead of silent STUN fallback
    def _mock_metered_fail(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "upstream failure"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(_mock_metered_fail)) as fail_client:
        try:
            await fetch_ice_servers(metered_cfg, http_client=fail_client)
            raise AssertionError("Expected MeteredTurnError on configured Metered failure")
        except MeteredTurnError as exc:
            assert "mock-metered-key" not in str(exc)

    # Endpoint returns 503 when MeteredTurnError occurs
    async def _raise_metered_err():
        raise MeteredTurnError("Unable to retrieve TURN relay credentials.")

    orig_fetch_ice = voice_endpoint.fetch_ice_servers
    voice_endpoint.fetch_ice_servers = _raise_metered_err
    try:
        with TestClient(fastapi_app) as client:
            res_503 = client.get("/api/v1/voice/ice-config")
            assert res_503.status_code == 503
    finally:
        voice_endpoint.fetch_ice_servers = orig_fetch_ice

    # Unconfigured fallback to Google STUN
    fallback = await fetch_ice_servers(Settings(_env_file=None, METERED_DOMAIN=None, METERED_API_KEY=None))
    assert fallback == [{"urls": "stun:stun.l.google.com:19302"}]
    print("  ✓ Metered turns:443?transport=tcp prioritized, 503 on failure, and STUN fallback verified")

    # 11. Verify all DB sessions created were closed
    print("\n[TEST 11] Database Session Cleanup")
    assert session_tracker.created_count > 0
    assert session_tracker.created_count == session_tracker.closed_count, (
        f"Session leak: created={session_tracker.created_count}, closed={session_tracker.closed_count}"
    )
    print(f"  ✓ All {session_tracker.created_count} SQLAlchemy sessions were properly closed")

    if dev_db_initial_stat is not None:
        dev_db_final_stat = (os.path.getmtime(dev_db_path), os.path.getsize(dev_db_path))
        assert dev_db_initial_stat == dev_db_final_stat
        print("  ✓ Development database (appointments.db) was NOT modified")

    print("\n" + "=" * 70)
    print("ALL VOICE TOOL & PIPELINE TESTS PASSED SUCCESSFULLY! (10/10 TEST SUITES VERIFIED)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_voice_tool_tests())
