"""Temporary, privacy-safe timing instrumentation for diagnosing mid-speech pauses.

Measures per-turn millisecond deltas across the outgoing speech path without
logging patient data, transcript text, or API credentials:
1. Consecutive Groq LLM text chunks.
2. Consecutive LLMTextFrames reaching TTS.
3. Consecutive ElevenLabs audio chunks arriving from WebSocket.
4. Consecutive audio frames leaving SmallWebRTCTransport output.
5. TTS text-send buffering, flush_audio, and context-finalization delays.
"""

import time
from dataclasses import dataclass

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

# Thresholds (ms) for emitting per-event gap warnings so normal 20ms streaming stays quiet.
LLM_CHUNK_GAP_WARN_MS = 250.0
TTS_AUDIO_GAP_WARN_MS = 250.0
TRANSPORT_OUT_GAP_WARN_MS = 150.0


def _short_ctx(context_id: str | None) -> str:
    """Safely format a nullable context ID for log correlation."""
    return (str(context_id) if context_id is not None else "none")[:8]


@dataclass
class SpeechTimingStats:
    """Per-call / per-turn timing counters and max observed gaps (ms)."""

    llm_chunks: int = 0
    max_llm_chunk_gap_ms: float = 0.0
    tts_in_frames: int = 0
    max_tts_in_gap_ms: float = 0.0
    tts_audio_chunks: int = 0
    max_tts_audio_gap_ms: float = 0.0
    transport_out_frames: int = 0
    max_transport_out_gap_ms: float = 0.0
    last_flush_ms: float = 0.0
    last_close_context_ms: float = 0.0


def attach_speech_timing_logs(
    llm: GroqLLMService,
    tts: ElevenLabsTTSService,
    transport: SmallWebRTCTransport,
) -> SpeechTimingStats:
    """Attach non-intrusive, privacy-safe timing hooks to LLM, TTS, and transport output."""
    stats = SpeechTimingStats()

    state: dict[str, float | None] = {
        "llm_start_t": None,
        "last_llm_chunk_t": None,
        "first_tts_in_t": None,
        "last_tts_in_t": None,
        "last_tts_send_t": None,
        "last_tts_audio_t": None,
        "last_transport_out_t": None,
    }

    # 1. Consecutive Groq text chunks
    orig_push_llm_text = llm._push_llm_text

    async def tracked_push_llm_text(text: str):
        try:
            now = time.perf_counter()
            stats.llm_chunks += 1
            last_t = state["last_llm_chunk_t"]
            if last_t is not None:
                dt_ms = (now - last_t) * 1000.0
                if dt_ms > stats.max_llm_chunk_gap_ms:
                    stats.max_llm_chunk_gap_ms = dt_ms
                if dt_ms >= LLM_CHUNK_GAP_WARN_MS:
                    logger.warning(
                        f"[SpeechTiming] Groq chunk gap dt={dt_ms:.1f}ms "
                        f"(chunk={stats.llm_chunks}, chars={len(text)})"
                    )
            state["last_llm_chunk_t"] = now
        except Exception:
            pass
        return await orig_push_llm_text(text)

    llm._push_llm_text = tracked_push_llm_text  # type: ignore[method-assign]

    # 2. LLMTextFrames reaching TTS + turn boundary resets
    orig_tts_process_frame = tts.process_frame

    async def tracked_tts_process_frame(frame: Frame, direction: FrameDirection):
        try:
            now = time.perf_counter()
            if isinstance(frame, LLMFullResponseStartFrame):
                state["llm_start_t"] = now
                state["last_llm_chunk_t"] = None
                state["first_tts_in_t"] = None
                state["last_tts_in_t"] = None
                state["last_tts_send_t"] = None
            elif isinstance(frame, LLMTextFrame):
                stats.tts_in_frames += 1
                if state["first_tts_in_t"] is None:
                    state["first_tts_in_t"] = now
                last_t = state["last_tts_in_t"]
                if last_t is not None:
                    dt_ms = (now - last_t) * 1000.0
                    if dt_ms > stats.max_tts_in_gap_ms:
                        stats.max_tts_in_gap_ms = dt_ms
                    if dt_ms >= LLM_CHUNK_GAP_WARN_MS:
                        logger.warning(
                            f"[SpeechTiming] TTS input LLMTextFrame gap dt={dt_ms:.1f}ms "
                            f"(frame={stats.tts_in_frames}, chars={len(frame.text)})"
                        )
                state["last_tts_in_t"] = now
            elif isinstance(frame, LLMFullResponseEndFrame):
                since_last_in = (
                    (now - state["last_tts_in_t"]) * 1000.0
                    if state["last_tts_in_t"] is not None
                    else 0.0
                )
                logger.info(
                    f"[SpeechTiming] LLM response end -> TTS: "
                    f"since_last_text={since_last_in:.1f}ms, "
                    f"max_llm_gap={stats.max_llm_chunk_gap_ms:.1f}ms, "
                    f"max_tts_in_gap={stats.max_tts_in_gap_ms:.1f}ms"
                )
        except Exception:
            pass
        return await orig_tts_process_frame(frame, direction)

    tts.process_frame = tracked_tts_process_frame  # type: ignore[method-assign]

    # 3 & 5. ElevenLabs _send_text buffering, flush_audio, _close_context, and audio chunk arrival
    orig_send_text = tts._send_text
    orig_flush_audio = tts.flush_audio
    orig_close_context = tts._close_context
    orig_append_audio = tts.append_to_audio_context

    async def tracked_send_text(text: str, context_id: str):
        try:
            now = time.perf_counter()
            since_first_in = (
                (now - state["first_tts_in_t"]) * 1000.0
                if state["first_tts_in_t"] is not None
                else 0.0
            )
            since_prev_send = (
                (now - state["last_tts_send_t"]) * 1000.0
                if state["last_tts_send_t"] is not None
                else 0.0
            )
            state["last_tts_send_t"] = now
            logger.debug(
                f"[SpeechTiming] ElevenLabs send_text ctx={_short_ctx(context_id)} "
                f"chars={len(text)} since_first_text={since_first_in:.1f}ms "
                f"since_prev_send={since_prev_send:.1f}ms"
            )
        except Exception:
            pass
        return await orig_send_text(text, context_id)

    async def tracked_flush_audio(context_id: str | None = None):
        t_start = time.perf_counter()
        res = await orig_flush_audio(context_id)
        try:
            stats.last_flush_ms = (time.perf_counter() - t_start) * 1000.0
            logger.debug(
                f"[SpeechTiming] ElevenLabs flush_audio ctx={_short_ctx(context_id)} "
                f"took={stats.last_flush_ms:.1f}ms"
            )
        except Exception:
            pass
        return res

    async def tracked_close_context(context_id: str):
        t_start = time.perf_counter()
        res = await orig_close_context(context_id)
        try:
            stats.last_close_context_ms = (time.perf_counter() - t_start) * 1000.0
            logger.debug(
                f"[SpeechTiming] ElevenLabs close_context ctx={_short_ctx(context_id)} "
                f"took={stats.last_close_context_ms:.1f}ms"
            )
        except Exception:
            pass
        return res

    async def tracked_append_audio(context_id: str | None, frame: Frame):
        try:
            now = time.perf_counter()
            if isinstance(frame, TTSAudioRawFrame):
                stats.tts_audio_chunks += 1
                last_t = state["last_tts_audio_t"]
                if last_t is not None:
                    dt_ms = (now - last_t) * 1000.0
                    if dt_ms > stats.max_tts_audio_gap_ms:
                        stats.max_tts_audio_gap_ms = dt_ms
                    if dt_ms >= TTS_AUDIO_GAP_WARN_MS:
                        dur_ms = len(frame.audio) / max(frame.sample_rate * 2, 1) * 1000.0
                        logger.warning(
                            f"[SpeechTiming] ElevenLabs audio chunk gap dt={dt_ms:.1f}ms "
                            f"ctx={_short_ctx(context_id)} chunk={stats.tts_audio_chunks} dur={dur_ms:.1f}ms"
                        )
                state["last_tts_audio_t"] = now
            elif isinstance(frame, TTSStoppedFrame):
                logger.info(
                    f"[SpeechTiming] ElevenLabs context stopped ctx={_short_ctx(context_id)} "
                    f"max_audio_gap={stats.max_tts_audio_gap_ms:.1f}ms "
                    f"max_transport_gap={stats.max_transport_out_gap_ms:.1f}ms "
                    f"flush={stats.last_flush_ms:.1f}ms close={stats.last_close_context_ms:.1f}ms"
                )
                state["last_tts_audio_t"] = None
                state["last_transport_out_t"] = None
        except Exception:
            pass
        return await orig_append_audio(context_id, frame)

    tts._send_text = tracked_send_text  # type: ignore[method-assign]
    tts.flush_audio = tracked_flush_audio  # type: ignore[method-assign]
    tts._close_context = tracked_close_context  # type: ignore[method-assign]
    tts.append_to_audio_context = tracked_append_audio  # type: ignore[method-assign]

    # 4. Audio frames leaving SmallWebRTCTransport output
    output_transport = transport.output()
    orig_write_audio_frame = output_transport.write_audio_frame

    async def tracked_write_audio_frame(frame) -> bool:
        try:
            now = time.perf_counter()
            stats.transport_out_frames += 1
            last_t = state["last_transport_out_t"]
            if last_t is not None:
                dt_ms = (now - last_t) * 1000.0
                if dt_ms > stats.max_transport_out_gap_ms:
                    stats.max_transport_out_gap_ms = dt_ms
                if dt_ms >= TRANSPORT_OUT_GAP_WARN_MS:
                    logger.warning(
                        f"[SpeechTiming] Transport audio_out gap dt={dt_ms:.1f}ms "
                        f"(frame={stats.transport_out_frames}, bytes={len(frame.audio)})"
                    )
            state["last_transport_out_t"] = now
        except Exception:
            pass
        return await orig_write_audio_frame(frame)

    output_transport.write_audio_frame = tracked_write_audio_frame  # type: ignore[method-assign]

    return stats
