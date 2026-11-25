"""Temporary, privacy-safe timing instrumentation and TTS WS diagnostics.

Measures per-turn millisecond deltas and inspects ElevenLabs WebSocket
responses across the outgoing speech path without logging patient data,
transcript text, base64 audio payloads, or API credentials:
1. Consecutive Groq LLM text chunks.
2. Consecutive LLMTextFrames reaching TTS.
3. Consecutive ElevenLabs audio chunks arriving from WebSocket.
4. Consecutive audio frames leaving SmallWebRTCTransport output.
5. TTS text-send buffering, flush_audio, and context-finalization delays.
6. Per-context audio chunk counts, final-message structure, and provider errors.
"""

import base64
import json
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

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
from pipecat.services.elevenlabs.tts_base import (
    _select_alignment,
    _strip_utterance_leading_spaces,
    _word_timestamps_include_inter_frame_spaces,
)
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.utils.types import is_given

# Thresholds (ms) for emitting per-event gap warnings so normal 20ms streaming stays quiet.
LLM_CHUNK_GAP_WARN_MS = 250.0
TTS_AUDIO_GAP_WARN_MS = 250.0
TRANSPORT_OUT_GAP_WARN_MS = 150.0
MAX_TRACKED_CONTEXTS = 32

_SECRET_KEY_RE = re.compile(r"sk_[A-Za-z0-9_-]{8,}")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]+")
_HEADER_SECRET_RE = re.compile(
    r"(?i)\b(xi-api-key|api[_-]?key|authorization|secret|token)\s*[:=]\s*[\"']?[^\"'\s,}]+[\"']?"
)
_B64_BLOB_RE = re.compile(r"[A-Za-z0-9+/=]{64,}")
_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\-\(\)\s]{6,}\d(?!\w)")


def _short_ctx(context_id: str | None) -> str:
    """Safely format a nullable context ID for log correlation."""
    return (str(context_id) if context_id is not None else "none")[:8]


def sanitize_provider_text(value: Any, max_len: int = 240) -> str:
    """Collapse whitespace and redact secrets, payloads, and phone digits from provider text."""
    text = " ".join(str(value).split())
    text = _SECRET_KEY_RE.sub("sk_[REDACTED]", text)
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    text = _HEADER_SECRET_RE.sub(r"\1=[REDACTED]", text)
    text = _B64_BLOB_RE.sub("[REDACTED_PAYLOAD]", text)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    if len(text) > max_len:
        return f"{text[: max_len - 3]}..."
    return text


def extract_elevenlabs_error(msg: Mapping[str, Any]) -> tuple[bool, str | None, str | None]:
    """Detect and extract sanitized provider error code and message from a WS payload."""
    raw_code: Any = msg.get("code")
    raw_msg: Any = None
    has_error = False

    err_val = msg.get("error")
    if err_val is not None and err_val != "":
        has_error = True
        if isinstance(err_val, Mapping):
            raw_code = err_val.get("code") or err_val.get("status") or raw_code
            raw_msg = (
                err_val.get("message")
                or err_val.get("detail")
                or err_val.get("error")
                or err_val.get("type")
                or err_val
            )
        else:
            msg_field = msg.get("message")
            if msg_field and str(msg_field) != str(err_val):
                raw_msg = f"{err_val}: {msg_field}"
            else:
                raw_msg = msg_field or err_val
            if raw_code is None:
                raw_code = msg.get("error_type") or (
                    err_val if isinstance(err_val, str) and len(err_val) <= 64 and " " not in err_val else None
                )

    detail_val = msg.get("detail")
    if not has_error and detail_val is not None and detail_val != "":
        has_error = True
        if isinstance(detail_val, Mapping):
            raw_code = detail_val.get("code") or detail_val.get("status") or raw_code
            raw_msg = detail_val.get("message") or detail_val.get("error") or detail_val
        else:
            raw_msg = msg.get("message") or detail_val

    if not has_error and msg.get("error_type") is not None:
        has_error = True
        raw_code = msg.get("error_type") or raw_code
        raw_msg = msg.get("message") or raw_code

    status_val = str(msg.get("status", "")).lower()
    is_error_code = raw_code is not None and (
        isinstance(raw_code, str)
        or (isinstance(raw_code, int) and not isinstance(raw_code, bool) and raw_code >= 400)
    )
    if not has_error and (status_val in {"error", "failed"} or (is_error_code and not msg.get("audio"))):
        has_error = True
        raw_msg = msg.get("message") or status_val or f"code={raw_code}"

    # Standalone 'message' without audio/alignment on an ElevenLabs WS packet indicates status/error.
    if (
        not has_error
        and msg.get("message")
        and not msg.get("audio")
        and not msg.get("alignment")
        and not msg.get("normalizedAlignment")
    ):
        has_error = True
        raw_msg = msg.get("message")

    if not has_error:
        return False, None, None

    sanitized_code = sanitize_provider_text(raw_code, max_len=64) if raw_code is not None else None
    sanitized_msg = sanitize_provider_text(raw_msg, max_len=240) if raw_msg is not None else None
    return True, sanitized_code, sanitized_msg


def summarize_tts_ws_message(msg: Mapping[str, Any]) -> dict[str, Any]:
    """Build a privacy-safe structural summary of an incoming ElevenLabs WebSocket message."""
    context_id = msg.get("contextId") if "contextId" in msg else msg.get("context_id")
    is_final_raw = msg.get("isFinal") if "isFinal" in msg else msg.get("is_final")
    is_final = is_final_raw is True

    has_audio_field = "audio" in msg
    audio_val = msg.get("audio")
    audio_is_null = has_audio_field and audio_val is None
    audio_is_empty = has_audio_field and isinstance(audio_val, str) and len(audio_val) == 0
    has_audio_payload = isinstance(audio_val, str) and len(audio_val) > 0

    has_alignment = bool(msg.get("alignment"))
    has_normalized_alignment = bool(msg.get("normalizedAlignment"))
    alignment_present = has_alignment or has_normalized_alignment

    has_error_field, error_code, error_message = extract_elevenlabs_error(msg)

    return {
        "context_id": context_id,
        "is_final_raw": is_final_raw,
        "is_final": is_final,
        "has_audio_field": has_audio_field,
        "audio_is_null": audio_is_null,
        "audio_is_empty": audio_is_empty,
        "has_audio_payload": has_audio_payload,
        "has_alignment": has_alignment,
        "has_normalized_alignment": has_normalized_alignment,
        "alignment_present": alignment_present,
        "has_error_field": has_error_field,
        "error_code": error_code,
        "error_message": error_message,
    }


@dataclass
class _ContextAudioStats:
    """Per-context audio tracking for no-audio outcome diagnostics."""

    audio_chunks: int = 0
    first_text_t: float = 0.0
    flush_t: float = 0.0  # time when flush_audio was called; 0 = not yet called
    close_t: float = 0.0  # time when close_context was called; 0 = not yet called
    provider_error: str | None = None
    final_received: bool = False


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
    tts_final_messages: int = 0
    tts_provider_errors: int = 0
    last_tts_error: str | None = None


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

    # Per-context audio-chunk tracking: context_id -> _ContextAudioStats.
    # Bounded by MAX_TRACKED_CONTEXTS and cleaned on TTSStoppedFrame / isFinal.
    _ctx_stats: dict[str, _ContextAudioStats] = {}

    def _get_or_create_ctx_stats(ctx_key: str) -> _ContextAudioStats | None:
        if not ctx_key:
            return None
        cs = _ctx_stats.get(ctx_key)
        if cs is None:
            while len(_ctx_stats) >= MAX_TRACKED_CONTEXTS:
                oldest_key = next(iter(_ctx_stats))
                _ctx_stats.pop(oldest_key, None)
            cs = _ContextAudioStats(
                first_text_t=(
                    state["first_tts_in_t"]
                    if state["first_tts_in_t"] is not None
                    else time.perf_counter()
                )
            )
            _ctx_stats[ctx_key] = cs
        return cs

    def _format_ws_diag(ctx_id: Any, summary: dict[str, Any]) -> str:
        return (
            f"ctx={_short_ctx(ctx_id)} "
            f"isFinal={summary['is_final_raw']} "
            f"has_audio_field={summary['has_audio_field']} "
            f"audio_null={summary['audio_is_null']} "
            f"audio_empty={summary['audio_is_empty']} "
            f"alignment_present={summary['alignment_present']} "
            f"has_error_field={summary['has_error_field']}"
        )

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
            ctx_key = str(context_id) if context_id is not None else ""
            _get_or_create_ctx_stats(ctx_key)
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
        target_ctx = context_id
        try:
            if target_ctx is None and hasattr(tts, "get_active_audio_context_id"):
                target_ctx = tts.get_active_audio_context_id()
            ctx_key = str(target_ctx) if target_ctx is not None else ""
            if ctx_key and ctx_key in _ctx_stats and _ctx_stats[ctx_key].flush_t == 0.0:
                _ctx_stats[ctx_key].flush_t = t_start
        except Exception:
            pass
        res = await orig_flush_audio(context_id)
        try:
            stats.last_flush_ms = (time.perf_counter() - t_start) * 1000.0
            logger.debug(
                f"[SpeechTiming] ElevenLabs flush_audio ctx={_short_ctx(target_ctx)} "
                f"took={stats.last_flush_ms:.1f}ms"
            )
        except Exception:
            pass
        return res

    async def tracked_close_context(context_id: str):
        t_start = time.perf_counter()
        try:
            # Record close start before awaiting WS send so an immediate isFinal
            # response during send() does not observe close_t == 0.0.
            ctx_key = str(context_id) if context_id is not None else ""
            if ctx_key and ctx_key in _ctx_stats and _ctx_stats[ctx_key].close_t == 0.0:
                _ctx_stats[ctx_key].close_t = t_start
        except Exception:
            pass
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
            ctx_key = str(context_id) if context_id is not None else ""
            if isinstance(frame, TTSAudioRawFrame):
                stats.tts_audio_chunks += 1
                cs = _get_or_create_ctx_stats(ctx_key)
                if cs is not None:
                    cs.audio_chunks += 1
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
                if ctx_key and ctx_key in _ctx_stats:
                    cs = _ctx_stats.pop(ctx_key)
                    close_to_final_ms = (
                        max((now - cs.close_t) * 1000.0, 0.0) if cs.close_t > 0.0 else 0.0
                    )
                    flush_to_final_ms = (
                        max((now - cs.flush_t) * 1000.0, 0.0) if cs.flush_t > 0.0 else 0.0
                    )
                    if cs.provider_error and cs.audio_chunks == 0:
                        logger.error(
                            f"[SpeechTiming] TTS context failed with provider error "
                            f"ctx={_short_ctx(context_id)} "
                            f"close_to_stopped={close_to_final_ms:.1f}ms "
                            f"flush_to_stopped={flush_to_final_ms:.1f}ms "
                            f"error={cs.provider_error}"
                        )
                    elif cs.audio_chunks == 0:
                        logger.warning(
                            f"[SpeechTiming] TTS context no-audio outcome "
                            f"ctx={_short_ctx(context_id)} "
                            f"close_to_stopped={close_to_final_ms:.1f}ms "
                            f"flush_to_stopped={flush_to_final_ms:.1f}ms "
                            f"(close_called={cs.close_t > 0.0}, flush_called={cs.flush_t > 0.0})"
                        )
                    else:
                        ttfa_ms = (
                            (now - cs.first_text_t) * 1000.0 if cs.first_text_t > 0.0 else 0.0
                        )
                        logger.debug(
                            f"[SpeechTiming] TTS context complete "
                            f"ctx={_short_ctx(context_id)} "
                            f"audio_chunks={cs.audio_chunks} ttfa={ttfa_ms:.1f}ms"
                        )
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

    async def tracked_receive_messages():
        """Inspect incoming ElevenLabs WS messages, report errors, and preserve audio in final frames."""
        async for message in tts._get_websocket():
            try:
                msg = json.loads(message)
            except Exception as exc:
                msg_len = len(message) if hasattr(message, "__len__") else 0
                logger.warning(
                    f"[SpeechTiming] ElevenLabs WS non-JSON message len={msg_len} err={type(exc).__name__}"
                )
                continue

            if not isinstance(msg, dict):
                logger.warning(f"[SpeechTiming] ElevenLabs WS unexpected payload type={type(msg).__name__}")
                continue

            try:
                summary = summarize_tts_ws_message(msg)
            except Exception:
                summary = {
                    "context_id": msg.get("contextId"),
                    "is_final_raw": msg.get("isFinal"),
                    "is_final": msg.get("isFinal") is True,
                    "has_audio_field": "audio" in msg,
                    "audio_is_null": msg.get("audio") is None and "audio" in msg,
                    "audio_is_empty": msg.get("audio") == "",
                    "has_audio_payload": isinstance(msg.get("audio"), str) and bool(msg.get("audio")),
                    "alignment_present": False,
                    "has_error_field": False,
                    "error_code": None,
                    "error_message": None,
                }

            received_ctx_id = summary["context_id"]
            ctx_key = str(received_ctx_id) if received_ctx_id is not None else ""

            # 1. Process audio payload BEFORE isFinal so audio on final frames is never dropped.
            if summary["has_audio_payload"]:
                try:
                    audio = base64.b64decode(msg["audio"])
                    sample_rate = tts.sample_rate or getattr(tts, "_init_sample_rate", 0) or 24000
                    frame = TTSAudioRawFrame(audio, sample_rate, 1, context_id=received_ctx_id)
                    await tts.append_to_audio_context(received_ctx_id, frame)
                except Exception as exc:
                    err_text = sanitize_provider_text(f"audio decode failed: {type(exc).__name__}")
                    stats.tts_provider_errors += 1
                    stats.last_tts_error = err_text
                    logger.error(
                        f"[SpeechTiming] ElevenLabs WS audio decode error ctx={_short_ctx(received_ctx_id)} {err_text}"
                    )
                    await tts.push_error(error_msg=f"ElevenLabs TTS {err_text}")

            # 2. Capture and report provider error fields instead of treating failure as silence.
            if summary["has_error_field"]:
                err_code = summary["error_code"]
                err_msg = summary["error_message"] or "Unknown provider error"
                formatted_err = f"[{err_code}] {err_msg}" if err_code is not None else err_msg
                try:
                    stats.tts_provider_errors += 1
                    stats.last_tts_error = formatted_err
                    cs_err = _get_or_create_ctx_stats(ctx_key) if ctx_key else None
                    if cs_err is not None:
                        cs_err.provider_error = formatted_err
                    logger.error(
                        f"[SpeechTiming] ElevenLabs WS provider error "
                        f"({_format_ws_diag(received_ctx_id, summary)} "
                        f"error_code={err_code} error_msg={err_msg})"
                    )
                except Exception:
                    pass
                await tts.push_error(
                    error_msg=f"ElevenLabs TTS error (ctx={_short_ctx(received_ctx_id)}): {formatted_err}"
                )

            # 3. Process alignment metadata while context is still open (never log text/chars).
            ctx_available = bool(received_ctx_id) and tts.audio_context_available(received_ctx_id)
            if ctx_available:
                raw_alignment = _select_alignment(
                    msg,
                    normalized_key="normalizedAlignment",
                    alignment_key="alignment",
                    prefer_normalized=bool(tts._pronunciation_dictionary_locators),
                )
                if raw_alignment and "chars" in raw_alignment and "charStartTimesMs" in raw_alignment:
                    alignment = _strip_utterance_leading_spaces(
                        raw_alignment,
                        ("chars", "charStartTimesMs", "charDurationsMs"),
                        received_ctx_id not in tts._alignment_started_context_ids,
                    )
                    tts._alignment_started_context_ids.add(received_ctx_id)
                    word_times = tts._word_times_from_alignment(received_ctx_id, alignment)

                    if word_times:
                        lang = tts._settings.language if is_given(tts._settings.language) else None
                        await tts.add_word_timestamps(
                            word_times,
                            received_ctx_id,
                            includes_inter_frame_spaces=(
                                True if _word_timestamps_include_inter_frame_spaces(lang) else None
                            ),
                        )

            # 4. Handle final response diagnostics and context teardown.
            if summary["is_final"]:
                try:
                    stats.tts_final_messages += 1
                    cs = _ctx_stats.get(ctx_key) if ctx_key else None
                    if cs is not None:
                        cs.final_received = True
                    context_had_audio = summary["has_audio_payload"] or (
                        cs is not None and cs.audio_chunks > 0
                    )
                    had_error = summary["has_error_field"] or (
                        cs is not None and cs.provider_error is not None
                    )
                    diag = _format_ws_diag(received_ctx_id, summary)
                    if not context_had_audio and not had_error:
                        logger.warning(
                            f"[SpeechTiming] ElevenLabs final response contained no audio "
                            f"and no provider error ({diag})"
                        )
                    elif had_error:
                        err_summary = summary["error_message"] or (cs.provider_error if cs else None)
                        logger.warning(
                            f"[SpeechTiming] ElevenLabs final response completed with provider error "
                            f"({diag} error_code={summary['error_code']} error_msg={err_summary})"
                        )
                    else:
                        logger.info(f"[SpeechTiming] ElevenLabs WS final ({diag})")
                except Exception:
                    pass

                logger.debug(f"Received final message for context {received_ctx_id}")
                if received_ctx_id and tts.audio_context_available(received_ctx_id):
                    await tts.append_to_audio_context(
                        received_ctx_id, TTSStoppedFrame(context_id=received_ctx_id)
                    )
                    await tts.remove_audio_context(received_ctx_id)
                if ctx_key:
                    _ctx_stats.pop(ctx_key, None)
                continue

            # 5. Surface unexpected non-final messages lacking audio, alignment, and errors.
            if (
                not summary["has_audio_payload"]
                and not summary["alignment_present"]
                and not summary["has_error_field"]
            ):
                try:
                    logger.warning(
                        f"[SpeechTiming] ElevenLabs WS unexpected non-final message "
                        f"({_format_ws_diag(received_ctx_id, summary)} keys={sorted(str(k) for k in msg.keys())})"
                    )
                except Exception:
                    pass

    tts._send_text = tracked_send_text  # type: ignore[method-assign]
    tts.flush_audio = tracked_flush_audio  # type: ignore[method-assign]
    tts._close_context = tracked_close_context  # type: ignore[method-assign]
    tts.append_to_audio_context = tracked_append_audio  # type: ignore[method-assign]
    tts._receive_messages = tracked_receive_messages  # type: ignore[method-assign]

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
