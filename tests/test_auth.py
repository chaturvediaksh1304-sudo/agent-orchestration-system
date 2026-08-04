"""Phase 6 criterion 1: the OAuth device flow.

Driven against a stub provider — no network, no real client id. The live
handshake needs an OAuth app registered by the user; that part is flagged in
Memory.md rather than faked here.
"""

import json
import stat

import pytest

from agent_orchestration import auth
from agent_orchestration.auth import (
    AuthError,
    Credentials,
    DeviceCode,
    clear_credentials,
    fetch_identity,
    load_credentials,
    poll_for_token,
    request_device_code,
    save_credentials,
)


class StubResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class StubSession:
    """Replays scripted provider responses and records what was sent."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append((url, data))
        return self.responses.pop(0)

    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, headers))
        return self.responses.pop(0)


DEVICE = DeviceCode(
    device_code="dev-123",
    user_code="ABCD-1234",
    verification_uri="https://example.test/device",
    interval=1,
    expires_in=100,
)


@pytest.fixture(autouse=True)
def client_id(monkeypatch):
    monkeypatch.setattr(auth, "CLIENT_ID", "test-client-id")


def test_device_code_request_returns_what_the_human_needs():
    session = StubSession(
        StubResponse(
            {
                "device_code": "dev-123",
                "user_code": "ABCD-1234",
                "verification_uri": "https://example.test/device",
                "interval": 5,
                "expires_in": 900,
            }
        )
    )

    device = request_device_code(session)

    assert device.user_code == "ABCD-1234"
    assert device.verification_uri == "https://example.test/device"


def test_the_client_id_is_sent_but_no_secret():
    """Device flow exists precisely so a CLI need not ship a client secret."""
    session = StubSession(
        StubResponse(
            {
                "device_code": "d",
                "user_code": "u",
                "verification_uri": "https://example.test/device",
            }
        )
    )

    request_device_code(session)

    _, data = session.calls[0]
    assert data["client_id"] == "test-client-id"
    assert "client_secret" not in data


def test_polling_returns_the_token_once_approved():
    session = StubSession(StubResponse({"access_token": "tok-abc"}))

    assert poll_for_token(DEVICE, session, sleep=lambda s: None) == "tok-abc"


def test_polling_waits_through_authorization_pending():
    session = StubSession(
        StubResponse({"error": "authorization_pending"}),
        StubResponse({"error": "authorization_pending"}),
        StubResponse({"access_token": "tok-abc"}),
    )

    assert poll_for_token(DEVICE, session, sleep=lambda s: None) == "tok-abc"
    assert len(session.calls) == 3


def test_slow_down_increases_the_interval():
    """Ignoring the server's backoff signal risks being rate-limited or banned."""
    slept = []
    session = StubSession(
        StubResponse({"error": "slow_down"}),
        StubResponse({"access_token": "tok-abc"}),
    )

    poll_for_token(DEVICE, session, sleep=slept.append)

    assert slept[0] == DEVICE.interval + 5


def test_expired_code_is_reported_not_retried_forever():
    session = StubSession(StubResponse({"error": "expired_token"}))

    with pytest.raises(AuthError, match="expired"):
        poll_for_token(DEVICE, session, sleep=lambda s: None)


def test_denied_authorization_is_reported():
    session = StubSession(StubResponse({"error": "access_denied"}))

    with pytest.raises(AuthError, match="denied"):
        poll_for_token(DEVICE, session, sleep=lambda s: None)


def test_polling_gives_up_at_the_deadline():
    session = StubSession(*[StubResponse({"error": "authorization_pending"})] * 50)

    with pytest.raises(AuthError, match="Timed out"):
        poll_for_token(DEVICE, session, sleep=lambda s: None, deadline=3)


def test_identity_is_read_from_the_provider():
    session = StubSession(StubResponse({"login": "octocat", "id": 583231}))

    assert fetch_identity("tok-abc", session) == "octocat"


def test_identity_request_sends_the_bearer_token():
    session = StubSession(StubResponse({"login": "octocat"}))

    fetch_identity("tok-abc", session)

    _, headers = session.calls[0]
    assert headers["Authorization"] == "Bearer tok-abc"


def test_saved_credentials_round_trip(tmp_path):
    path = tmp_path / "token.json"

    save_credentials(Credentials(access_token="tok", user_id="octocat"), path)

    assert load_credentials(path).user_id == "octocat"


def test_the_token_file_is_not_world_readable(tmp_path):
    """It is a bearer credential; other accounts must not be able to read it."""
    path = tmp_path / "token.json"

    save_credentials(Credentials(access_token="tok", user_id="octocat"), path)

    mode = stat.S_IMODE(path.stat().st_mode)
    assert not mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH)


def test_missing_credentials_read_as_none(tmp_path):
    assert load_credentials(tmp_path / "absent.json") is None


def test_corrupt_credentials_read_as_none_rather_than_crashing(tmp_path):
    path = tmp_path / "token.json"
    path.write_text("{not json")

    assert load_credentials(path) is None


def test_credentials_missing_fields_read_as_none(tmp_path):
    path = tmp_path / "token.json"
    path.write_text(json.dumps({"access_token": "tok"}))

    assert load_credentials(path) is None


def test_logout_removes_the_token(tmp_path):
    path = tmp_path / "token.json"
    save_credentials(Credentials(access_token="tok", user_id="octocat"), path)

    assert clear_credentials(path) is True
    assert not path.exists()


def test_logout_when_not_logged_in_is_not_an_error(tmp_path):
    assert clear_credentials(tmp_path / "absent.json") is False


def test_auth_is_inactive_without_a_client_id(monkeypatch):
    """No provider configured must leave the CLI usable, not broken."""
    monkeypatch.setattr(auth, "CLIENT_ID", None)

    assert auth.configured() is False


# --- CLI integration of auth ------------------------------------------------


def test_whoami_reports_the_logged_in_user(tmp_path, monkeypatch, capsys):
    from agent_orchestration import cli

    path = tmp_path / "token.json"
    save_credentials(Credentials(access_token="tok", user_id="octocat"), path)
    monkeypatch.setattr(auth, "TOKEN_PATH", path)

    assert cli.main(["whoami"]) == 0
    assert "octocat" in capsys.readouterr().out


def test_whoami_when_logged_out(tmp_path, monkeypatch, capsys):
    from agent_orchestration import cli

    monkeypatch.setattr(auth, "TOKEN_PATH", tmp_path / "absent.json")

    assert cli.main(["whoami"]) == 1
    assert "Not logged in" in capsys.readouterr().out


def test_logout_clears_the_session(tmp_path, monkeypatch, capsys):
    from agent_orchestration import cli

    path = tmp_path / "token.json"
    save_credentials(Credentials(access_token="tok", user_id="octocat"), path)
    monkeypatch.setattr(auth, "TOKEN_PATH", path)

    assert cli.main(["logout"]) == 0
    assert not path.exists()


def test_login_without_a_provider_explains_itself(monkeypatch, capsys):
    from agent_orchestration import cli

    monkeypatch.setattr(auth, "CLIENT_ID", None)

    assert cli.main(["login"]) == 1
    assert "OAUTH_CLIENT_ID" in capsys.readouterr().err


def test_a_run_is_blocked_when_configured_but_logged_out(tmp_path, monkeypatch, capsys):
    """Auth only gates once a provider exists, but then it really gates."""
    from agent_orchestration import cli

    monkeypatch.setattr(auth, "TOKEN_PATH", tmp_path / "absent.json")

    exit_code = cli.main(["a goal", "--workspace", str(tmp_path), "--no-memory"])

    assert exit_code == 1
    assert "login" in capsys.readouterr().err


def test_a_run_is_allowed_when_no_provider_is_configured(
    stub_model, two_subtasks, tmp_path, monkeypatch
):
    """CLI-first means usable with no auth provider at all."""
    from langchain_core.messages import AIMessage

    from agent_orchestration import cli

    monkeypatch.setattr(auth, "CLIENT_ID", None)
    monkeypatch.setattr(auth, "TOKEN_PATH", tmp_path / "absent.json")
    llm = stub_model(
        [two_subtasks, AIMessage(content="a"), AIMessage(content="b"), AIMessage(content="done")]
    )
    monkeypatch.setattr(cli, "build_llm", lambda provider, model: llm)

    assert cli.main(["a goal", "--workspace", str(tmp_path), "--no-memory"]) == 0
