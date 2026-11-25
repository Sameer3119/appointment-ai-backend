"""ICE / TURN server resolution for SmallWebRTC with Metered.ca support."""

from typing import Any, Dict, List, Optional

from aiortc import RTCIceServer
import httpx
from loguru import logger

from app.config import Settings, settings

DEFAULT_ICE_SERVERS: List[Dict[str, Any]] = [
    {"urls": "stun:stun.l.google.com:19302"},
]


class MeteredTurnError(RuntimeError):
    """Raised when configured Metered.ca TURN credential retrieval fails."""


def _turn_priority(entry: Dict[str, Any]) -> int:
    """Rank ICE server entries so aiortc picks turns: over TCP 443 first.

    aiortc uses only the first TURN server in iceServers. Render blocks inbound
    UDP, so ``turns:...:443?transport=tcp`` must precede ``turn:...:80`` UDP.
    """
    urls = entry.get("urls", "")
    url_str = urls[0] if isinstance(urls, list) and urls else str(urls)
    lower = url_str.lower()

    if lower.startswith("stun:") or lower.startswith("stuns:"):
        return 0
    if lower.startswith("turns:") and ":443" in lower and "transport=tcp" in lower:
        return 1
    if lower.startswith("turns:"):
        return 2
    if lower.startswith("turn:") and "transport=tcp" in lower and ":443" in lower:
        return 3
    if lower.startswith("turn:") and "transport=tcp" in lower:
        return 4
    return 5


def prioritize_ice_servers(servers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a copy of ``servers`` ordered so ``turns:`` TCP/443 precedes UDP TURN."""
    valid = [dict(s) for s in servers if isinstance(s, dict) and s.get("urls")]
    return sorted(valid, key=_turn_priority)


async def fetch_ice_servers(
    cfg: Settings = settings,
    *,
    http_client: Optional[httpx.AsyncClient] = None,
) -> List[Dict[str, Any]]:
    """Fetch ICE servers from Metered.ca when configured, else return Google STUN.

    Raises:
        MeteredTurnError: If Metered is configured (or partially configured) and
            credentials cannot be retrieved or are invalid.
    """
    domain = (cfg.METERED_DOMAIN or "").strip().rstrip("/")
    api_key = (cfg.METERED_API_KEY or "").strip()

    if not domain and not api_key:
        return [dict(s) for s in DEFAULT_ICE_SERVERS]

    if not domain or not api_key:
        raise MeteredTurnError(
            "Metered TURN configuration is incomplete; both METERED_DOMAIN and METERED_API_KEY are required."
        )

    endpoint = f"https://{domain}/api/v1/turn/credentials"
    try:
        if http_client is not None:
            response = await http_client.get(endpoint, params={"apiKey": api_key})
        else:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(endpoint, params={"apiKey": api_key})

        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise MeteredTurnError("Metered TURN service returned an unexpected response format.")

        prioritized = prioritize_ice_servers(payload)
        if not prioritized:
            raise MeteredTurnError("Metered TURN service returned no usable ICE servers.")

        return prioritized
    except MeteredTurnError:
        raise
    except Exception as exc:
        logger.warning(f"Failed to fetch Metered TURN credentials ({type(exc).__name__}).")
        raise MeteredTurnError("Unable to retrieve TURN relay credentials.") from exc


def to_rtc_ice_servers(servers: List[Dict[str, Any]]) -> List[RTCIceServer]:
    """Convert ICE server dictionaries into aiortc ``RTCIceServer`` objects."""
    rtc_servers: List[RTCIceServer] = []
    for entry in servers:
        urls = entry.get("urls")
        if not urls:
            continue
        kwargs: Dict[str, Any] = {"urls": urls}
        if entry.get("username"):
            kwargs["username"] = entry["username"]
        if entry.get("credential"):
            kwargs["credential"] = entry["credential"]
        rtc_servers.append(RTCIceServer(**kwargs))
    return rtc_servers or [RTCIceServer(urls="stun:stun.l.google.com:19302")]
