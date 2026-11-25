"""SmallWebRTC signaling endpoints for MediVoice AI real-time voice calls."""

import asyncio
from typing import Set

from aiortc import RTCIceServer
from fastapi import APIRouter, HTTPException, status
from loguru import logger
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from app.voice.ice import MeteredTurnError, fetch_ice_servers, to_rtc_ice_servers
from app.voice.pipeline import (
    VoiceConfigurationError,
    run_voice_pipeline,
    validate_voice_config,
)

router = APIRouter(prefix="/voice", tags=["Voice"])

webrtc_handler = SmallWebRTCRequestHandler(
    ice_servers=[RTCIceServer(urls="stun:stun.l.google.com:19302")]
)
_active_voice_tasks: Set[asyncio.Task] = set()


@router.get("/ice-config", summary="Get WebRTC ICE/TURN server configuration")
async def get_ice_config():
    """Return short-lived ICE/TURN server configuration without exposing API keys."""
    try:
        ice_servers = await fetch_ice_servers()
    except MeteredTurnError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return {"iceServers": ice_servers}


@router.post("/offer", summary="WebRTC SDP offer for voice session")
async def voice_offer(request: SmallWebRTCRequest):
    """Handle an incoming SmallWebRTC SDP offer and start an isolated voice pipeline."""
    try:
        validate_voice_config()
    except VoiceConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    try:
        ice_servers = await fetch_ice_servers()
    except MeteredTurnError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    webrtc_handler._ice_servers = to_rtc_ice_servers(ice_servers)

    async def on_webrtc_connection(webrtc_connection: SmallWebRTCConnection) -> None:
        task = asyncio.create_task(run_voice_pipeline(webrtc_connection))
        _active_voice_tasks.add(task)
        task.add_done_callback(_active_voice_tasks.discard)

    return await webrtc_handler.handle_web_request(
        request=request,
        webrtc_connection_callback=on_webrtc_connection,
    )


@router.patch("/offer", summary="WebRTC ICE candidate patch")
async def voice_ice_patch(request: SmallWebRTCPatchRequest):
    """Handle trickle ICE candidates for an active SmallWebRTC connection."""
    await webrtc_handler.handle_patch_request(request)
    return {"status": "ok"}


async def close_voice_resources() -> None:
    """Cancel active voice call tasks and close the WebRTC request handler."""
    if _active_voice_tasks:
        tasks = list(_active_voice_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        _active_voice_tasks.clear()

    try:
        await webrtc_handler.close()
    except Exception as exc:
        logger.warning(f"Error closing SmallWebRTCRequestHandler: {exc}")
