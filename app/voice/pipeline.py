"""Real-time Pipecat 1.17.0 voice pipeline for MediVoice AI."""

from typing import Tuple

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_start import (
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from app.config import Settings, settings
from app.services.voice_tool_service import VoiceCallState
from app.voice.phone_confirmation import (
    AssistantReadbackProcessor,
    PhoneConfirmationTracker,
    UserPhoneCaptureProcessor,
)
from app.voice.speech_timing import attach_speech_timing_logs
from app.voice.tools import register_appointment_tools

MEDIVOICE_SYSTEM_INSTRUCTION = (
    "You are MediVoice AI, a polite, helpful, and concise medical clinic voice assistant. "
    "You help patients check doctor availability, book appointments, and cancel appointments. "
    "Keep responses natural and conversational for spoken audio—avoid markdown, bullet lists, "
    "or overly long monologues. "
    "Before booking or cancelling any appointment, you MUST ask for the patient's phone number "
    "(and full name when booking), repeat the phone number digits back to the patient clearly, "
    "and wait for their explicit verbal confirmation before calling book_appointment or "
    "cancel_appointment. Also read back the doctor, date, and time before finalizing."
)


class VoiceConfigurationError(RuntimeError):
    """Raised when required voice service credentials or IDs are missing."""


def validate_voice_config(cfg: Settings = settings) -> None:
    """Validate that required voice API credentials and voice ID are configured."""
    missing = []
    if not cfg.GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not cfg.ELEVENLABS_API_KEY:
        missing.append("ELEVENLABS_API_KEY")
    if not cfg.ELEVENLABS_VOICE_ID:
        missing.append("ELEVENLABS_VOICE_ID")
    if missing:
        raise VoiceConfigurationError(
            f"Voice assistant is not configured. Missing: {', '.join(missing)}."
        )


def build_voice_pipeline(
    webrtc_connection: SmallWebRTCConnection,
    cfg: Settings = settings,
) -> Tuple[PipelineTask, SmallWebRTCTransport, VoiceCallState, PhoneConfirmationTracker]:
    """Construct a per-call Pipecat pipeline, transport, and isolated call state."""
    validate_voice_config(cfg)

    call_state = VoiceCallState()
    tracker = PhoneConfirmationTracker(call_state)

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_10ms_chunks=2,
        ),
    )

    stt = ElevenLabsRealtimeSTTService(
        api_key=cfg.ELEVENLABS_API_KEY,
        settings=ElevenLabsRealtimeSTTService.Settings(
            model=cfg.ELEVENLABS_STT_MODEL,
            filter_background_audio=True,
            no_verbatim=True,
        ),
    )

    llm = GroqLLMService(
        api_key=cfg.GROQ_API_KEY,
        settings=GroqLLMService.Settings(
            model=cfg.GROQ_MODEL,
            system_instruction=MEDIVOICE_SYSTEM_INSTRUCTION,
            reasoning_effort="low",
        ),
    )

    tts = ElevenLabsTTSService(
        api_key=cfg.ELEVENLABS_API_KEY,
        sample_rate=24000,
        auto_mode=True,
        # ElevenLabs' observed TTFB can exceed 9 s on Render; the default 3 s
        # context timeout fires first and declares the context audio-less before
        # ElevenLabs returns its final message.  12 s gives enough headroom for
        # slow responses while still bounding stuck contexts.
        stop_frame_timeout_s=12.0,
        settings=ElevenLabsTTSService.Settings(
            voice=cfg.ELEVENLABS_VOICE_ID,
            model=cfg.ELEVENLABS_TTS_MODEL,
        ),
    )

    tools = register_appointment_tools(llm, call_state)
    attach_speech_timing_logs(llm, tts, transport)

    context = LLMContext(
        messages=[{"role": "system", "content": MEDIVOICE_SYSTEM_INSTRUCTION}],
        tools=tools,
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=0.8,
                    start_secs=0.3,
                    stop_secs=0.3,
                    min_volume=0.65,
                )
            ),
            user_turn_strategies=UserTurnStrategies(
                start=[
                    VADUserTurnStartStrategy(),
                    TranscriptionUserTurnStartStrategy(use_interim=False),
                ],
                stop=[
                    SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.8),
                ],
            ),
        ),
    )

    user_phone_capture = UserPhoneCaptureProcessor(tracker)
    assistant_readback = AssistantReadbackProcessor(tracker)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_phone_capture,
            user_aggregator,
            llm,
            assistant_readback,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport_obj, client):
        logger.info("MediVoice WebRTC client connected")
        await task.queue_frames(
            [
                LLMMessagesAppendFrame(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "A patient has just joined the call. Greet them warmly as "
                                "MediVoice AI and ask how you can help with their appointment today."
                            ),
                        }
                    ],
                    run_llm=True,
                )
            ]
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport_obj, client):
        logger.info("MediVoice WebRTC client disconnected; cancelling pipeline task")
        await task.cancel()

    return task, transport, call_state, tracker


async def run_voice_pipeline(
    webrtc_connection: SmallWebRTCConnection,
    cfg: Settings = settings,
) -> None:
    """Run an isolated voice pipeline for a single WebRTC call."""
    task, _transport, _call_state, _tracker = build_voice_pipeline(webrtc_connection, cfg)
    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(task)
    except Exception as exc:
        logger.exception(f"Voice pipeline error: {exc}")
    finally:
        if not task.has_finished():
            await task.cancel()
