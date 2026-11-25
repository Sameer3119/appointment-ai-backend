"""Per-call phone number capture and explicit verbal confirmation processors."""

import re

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from app.services.voice_tool_service import VoiceCallState, normalize_phone

# Matches ISO dates like 2026-11-26 so they are not mistaken for phone numbers
_DATE_PATTERN = re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b")

# Matches phone-like digit sequences (at least 7 digits, optional leading '+')
_PHONE_CANDIDATE_PATTERN = re.compile(r"(?:\+?\d[\d\s\-\(\)\.]{5,}\d)")

# Negation / correction cues that must always block confirmation
_REJECTION_PATTERN = re.compile(
    r"\b(?:no|nope|nah|not|wrong|incorrect|isn't|isnt|wasn't|wasnt|"
    r"don't|dont|didn't|didnt|change|actually|wait|stop|mistake|typo)\b",
    re.IGNORECASE,
)

# Strict, unambiguous confirmation utterances (checked only after rejection filter)
_AFFIRMATIVE_EXACT = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "yup",
        "correct",
        "right",
        "confirmed",
        "confirm",
        "affirmative",
        "sure",
        "ok",
        "okay",
        "yes it is",
        "yes that is",
        "yes that's right",
        "yes thats right",
        "yes that is right",
        "yes that's correct",
        "yes thats correct",
        "yes that is correct",
        "yes correct",
        "yes please",
        "that's right",
        "thats right",
        "that is right",
        "that's correct",
        "thats correct",
        "that is correct",
        "it is",
    }
)


def extract_phone_candidate(text: str) -> str | None:
    """Extract a normalized phone number candidate from user speech, or return None."""
    if not text:
        return None
    cleaned_for_dates = _DATE_PATTERN.sub(" ", text)
    for match in _PHONE_CANDIDATE_PATTERN.findall(cleaned_for_dates):
        digits_only = re.sub(r"\D", "", match)
        if len(digits_only) >= 7:
            try:
                return normalize_phone(match)
            except ValueError:
                continue
    return None


def is_explicit_confirmation(text: str) -> bool:
    """Return True only when the utterance is an unambiguous confirmation without negation."""
    if not text or not text.strip():
        return False
    if _REJECTION_PATTERN.search(text):
        return False
    normalized = re.sub(r"[^\w\s']", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized in _AFFIRMATIVE_EXACT


def is_explicit_rejection(text: str) -> bool:
    """Return True when the utterance contains a rejection or correction cue."""
    if not text:
        return False
    return bool(_REJECTION_PATTERN.search(text))


class PhoneConfirmationTracker:
    """Tracks the per-call sequence: capture -> assistant readback -> user confirmation.

    Only after the assistant reads back the pending phone number and the user
    explicitly confirms it does this tracker promote ``pending_phone`` to
    ``confirmed_phone`` on the call's ``VoiceCallState``.
    """

    def __init__(self, call_state: VoiceCallState) -> None:
        self.call_state = call_state
        self.awaiting_confirmation: bool = False
        self.readback_completed: bool = False
        self._assistant_buffer: list[str] = []

    def process_user_transcript(self, text: str) -> None:
        """Process a finalized user transcript for phone capture or confirmation."""
        candidate = extract_phone_candidate(text)
        if candidate is not None:
            self.call_state.set_pending_phone(candidate)
            self.awaiting_confirmation = True
            self.readback_completed = False
            return

        if not self.awaiting_confirmation:
            return

        if is_explicit_rejection(text):
            self.call_state.clear()
            self.awaiting_confirmation = False
            self.readback_completed = False
            return

        if self.readback_completed and is_explicit_confirmation(text):
            self.call_state.confirm_pending_phone()
            self.awaiting_confirmation = False

    def append_assistant_text(self, text: str) -> None:
        """Buffer assistant response text for readback verification."""
        if text:
            self._assistant_buffer.append(text)

    def complete_assistant_turn(self) -> None:
        """Check whether the completed assistant turn read back the pending phone."""
        full_text = "".join(self._assistant_buffer)
        self._assistant_buffer.clear()

        if not self.awaiting_confirmation or not self.call_state.pending_phone:
            return

        pending_digits = re.sub(r"\D", "", self.call_state.pending_phone)
        spoken_digits = re.sub(r"\D", "", full_text)
        if pending_digits and pending_digits in spoken_digits:
            self.readback_completed = True


class UserPhoneCaptureProcessor(FrameProcessor):
    """Captures candidate phone numbers and confirmations from finalized STT frames."""

    def __init__(self, tracker: PhoneConfirmationTracker) -> None:
        super().__init__()
        self._tracker = tracker

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text:
            self._tracker.process_user_transcript(frame.text)
        await self.push_frame(frame, direction)


class AssistantReadbackProcessor(FrameProcessor):
    """Verifies the assistant actually read back the pending phone number."""

    def __init__(self, tracker: PhoneConfirmationTracker) -> None:
        super().__init__()
        self._tracker = tracker

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame) and frame.text:
            self._tracker.append_assistant_text(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._tracker.complete_assistant_turn()
        await self.push_frame(frame, direction)
