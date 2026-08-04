"""OAuth 2.0 device authorization grant (RFC 8628).

The `gh auth login` pattern: the CLI shows a short code, the human approves it in
a browser on any device, and the CLI polls until a token comes back. No local
web server and no client secret, which is what makes it right for a CLI.

Architecture.md line 11 questioned whether auth belongs in a CLI-first MVP at
all. It's here to attach an identity to runs and escalations, which is the
groundwork for the hosted multi-user mode that question anticipated.
"""

import json
import os
import stat
import time
from pathlib import Path
from typing import Optional

import requests
from pydantic import BaseModel

# GitHub by default: device flow needs only a public client id, no secret.
DEVICE_CODE_URL = os.environ.get(
    "OAUTH_DEVICE_CODE_URL", "https://github.com/login/device/code"
)
TOKEN_URL = os.environ.get(
    "OAUTH_TOKEN_URL", "https://github.com/login/oauth/access_token"
)
IDENTITY_URL = os.environ.get("OAUTH_IDENTITY_URL", "https://api.github.com/user")
CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID")
SCOPE = os.environ.get("OAUTH_SCOPE", "read:user")

TOKEN_PATH = Path(
    os.environ.get("ORCHESTRATION_TOKEN_PATH", Path.home() / ".orchestration" / "token.json")
)


class AuthError(Exception):
    """Raised when authentication cannot complete."""


class Credentials(BaseModel):
    access_token: str
    user_id: str


class DeviceCode(BaseModel):
    device_code: str
    user_code: str
    verification_uri: str
    interval: int = 5
    expires_in: int = 900


def configured() -> bool:
    """Whether a provider is set up.

    Without a client id there is nothing to authenticate against, so the CLI
    runs unauthenticated rather than becoming unusable.
    """
    return bool(CLIENT_ID)


def request_device_code(session: Optional[requests.Session] = None) -> DeviceCode:
    session = session or requests
    response = session.post(
        DEVICE_CODE_URL,
        data={"client_id": CLIENT_ID, "scope": SCOPE},
        headers={"Accept": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    return DeviceCode(**response.json())


def poll_for_token(
    device: DeviceCode,
    session: Optional[requests.Session] = None,
    sleep=time.sleep,
    deadline: Optional[float] = None,
) -> str:
    """Poll until the human approves, honouring the RFC's backoff signals."""
    session = session or requests
    interval = device.interval
    waited = 0.0
    limit = deadline if deadline is not None else device.expires_in

    while waited < limit:
        response = session.post(
            TOKEN_URL,
            data={
                "client_id": CLIENT_ID,
                "device_code": device.device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            headers={"Accept": "application/json"},
            timeout=30,
        )
        payload = response.json()

        if "access_token" in payload:
            return payload["access_token"]

        error = payload.get("error")
        if error == "authorization_pending":
            pass
        elif error == "slow_down":
            # The server is telling us to back off; ignoring it risks a ban.
            interval += 5
        elif error == "expired_token":
            raise AuthError("The device code expired. Run `login` again.")
        elif error == "access_denied":
            raise AuthError("Authorization was denied.")
        else:
            raise AuthError(f"Authorization failed: {error or payload}")

        sleep(interval)
        waited += interval

    raise AuthError("Timed out waiting for authorization.")


def fetch_identity(token: str, session: Optional[requests.Session] = None) -> str:
    session = session or requests
    response = session.get(
        IDENTITY_URL,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload.get("login") or payload.get("id") or payload.get("sub"))


def save_credentials(credentials: Credentials, path: Optional[Path] = None) -> Path:
    """Cache the token, readable only by its owner."""
    path = path or TOKEN_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(credentials.model_dump_json())
    # A bearer token is a credential; 0600 keeps it off other accounts on the box.
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return path


def load_credentials(path: Optional[Path] = None) -> Optional[Credentials]:
    path = path or TOKEN_PATH
    if not path.exists():
        return None
    try:
        return Credentials(**json.loads(path.read_text()))
    except (ValueError, TypeError):
        # A corrupt cache should send you back to `login`, not crash the run.
        return None


def clear_credentials(path: Optional[Path] = None) -> bool:
    path = path or TOKEN_PATH
    if path.exists():
        path.unlink()
        return True
    return False
