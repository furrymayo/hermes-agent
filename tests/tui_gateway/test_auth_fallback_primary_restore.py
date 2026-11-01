"""A TUI/Desktop chat built on the pre-agent auth fallback (the route it intended to run on raised
AuthError at agent build, a ``fallback_providers`` entry answered) returns to that route once its login
works again, instead of running on the fallback until a manual ``/model``. The intended route is the
session's pinned ``model_override`` (Desktop manual pick, resumed row) else the configured model; the
attach sites record it as ``auth_fallback_primary``, a turn-start step probes it quietly each turn and
switches back when it resolves, and a resumed outage row is rebuilt ON the intended route.

Drives the real ``_make_agent`` → ``_attach_built_agent`` → ``_restore_auth_fallback_primary`` →
``_apply_model_switch`` chain; only credential resolution, ``hermes_cli.model_switch.switch_model`` and
the agent's in-place swap are stubbed. No inference.
"""
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.auth import AuthError

PRIMARY_MODEL, PRIMARY_PROVIDER = "gpt-5.6-sol", "openai-codex"
FALLBACK_MODEL, FALLBACK_PROVIDER = "claude-sonnet-5", "anthropic"
PINNED_MODEL, PINNED_PROVIDER = "claude-opus-5-5", "anthropic"
SID, KEY = "sid-outage", "session-key-outage"


class _FakeDb:
    def __init__(self):
        self.rows: dict = {}

    def get_session(self, key):
        return self.rows.get(key)

    def update_session_meta(self, key, model_config, model):
        self.rows[key] = {"model_config": model_config, "model": model}


class _StubAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model, self.provider = kwargs["model"], kwargs["provider"]
        self.base_url = kwargs.get("base_url") or ""
        self.api_key = kwargs.get("api_key") or ""
        self.api_mode = kwargs.get("api_mode") or ""
        self._session_db = _FakeDb()
        self.switches: list = []

    def switch_model(self, *, new_model, new_provider, **_kw):
        self.switches.append((new_model, new_provider))
        self.model, self.provider = new_model, new_provider


@pytest.fixture
def outage(monkeypatch, tmp_path):
    from tui_gateway import server

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_IGNORE_RULES", "1")
    (tmp_path / "config.yaml").write_text(
        f"model:\n  default: {PRIMARY_MODEL}\n  provider: {PRIMARY_PROVIDER}\n", encoding="utf-8")
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_load_fallback_model",
                        lambda: [{"provider": FALLBACK_PROVIDER, "model": FALLBACK_MODEL, "api_key": "fb"}])
    state = {"primary_down": True}

    def _resolve(**kwargs):
        # Only the fallback entry's own model resolves while the login is down; the pinned pick shares
        # the fallback's provider, so the outage is keyed on the model like a per-model login would be.
        if state["primary_down"] and kwargs.get("target_model") != FALLBACK_MODEL:
            raise AuthError("token expired", provider=kwargs.get("requested") or PRIMARY_PROVIDER)
        return {"api_key": "k", "base_url": "https://example.invalid/v1",
                "provider": kwargs.get("requested") or PRIMARY_PROVIDER, "api_mode": "chat_completions"}

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _resolve)

    def _switch(*, raw_input, explicit_provider, **_kw):
        return SimpleNamespace(
            success=True, new_model=raw_input, target_provider=explicit_provider, base_url="", api_key="key",
            api_mode="chat_completions", warning_message="", model_info=None, error_message="",
            runtime_capabilities=None, usage_notice="")

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _switch)
    for name in ("_restart_slash_worker", "_persist_live_session_system_prompt",
                 "_append_model_switch_marker", "_emit_session_info"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    emits: list = []
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emits.append((event, payload)))
    monkeypatch.setattr(server, "_sessions", {})
    return SimpleNamespace(server=server, state=state, emits=emits)


def _open_outage_session(o, model_override: dict | None = None):
    """Build during the outage and attach, as the first prompt's deferred build does; ``model_override``
    is the pin ``session.create`` records for a Desktop manual pick (and the deferred build receives)."""
    session = {"session_key": KEY, "history": [], "history_lock": threading.Lock()}
    if model_override is not None:
        session["model_override"] = model_override
    with patch("run_agent.AIAgent", _StubAgent):
        agent = o.server._make_agent(SID, KEY, context_cwd_is_launch_artifact=False, model_override=model_override)
    assert (agent.model, agent.provider) == (FALLBACK_MODEL, FALLBACK_PROVIDER)
    o.server._sessions[SID] = session
    assert o.server._attach_built_agent(SID, session, agent)
    return session, agent


def _turn_start(o, session):
    """The turn-start model steps in production order (``_prepare_turn_input``)."""
    o.server._restore_auth_fallback_primary(SID, session)
    o.server._sync_agent_model_with_config(SID, session)


def _persisted_model_config(agent) -> dict:
    return json.loads(agent._session_db.rows[KEY]["model_config"])


def _notices(o) -> list:
    return [p["message"] for e, p in o.emits if e == "notice"]


def test_outage_session_returns_to_the_configured_model_once_its_login_works(outage):
    session, agent = _open_outage_session(outage)
    assert session["auth_fallback_primary"] == {"model": PRIMARY_MODEL, "provider": PRIMARY_PROVIDER}
    assert session["config_model_seen"] == (PRIMARY_MODEL, PRIMARY_PROVIDER)

    # Login still down at turn start: no switch attempt, nothing emitted, marker kept -> next turn retries.
    for _turn in range(2):
        _turn_start(outage, session)
        assert agent.switches == []
        assert [e for e, _ in outage.emits if e in ("error", "notice")] == []
        assert session["auth_fallback_primary"] == {"model": PRIMARY_MODEL, "provider": PRIMARY_PROVIDER}

    outage.state["primary_down"] = False
    _turn_start(outage, session)
    assert agent.switches == [(PRIMARY_MODEL, PRIMARY_PROVIDER)]
    assert (agent.model, agent.provider) == (PRIMARY_MODEL, PRIMARY_PROVIDER)
    assert "auth_fallback_primary" not in session
    assert "auth_fallback_primary" not in _persisted_model_config(agent)
    assert "model_override" not in session  # a restore, not a session pin
    assert len(_notices(outage)) == 1
    assert f"{PRIMARY_MODEL} via {PRIMARY_PROVIDER}" in _notices(outage)[0]
    assert f"{FALLBACK_MODEL} via {FALLBACK_PROVIDER}" in _notices(outage)[0]

    # Settled: the following turn neither switches again nor repeats the notice.
    _turn_start(outage, session)
    assert len(agent.switches) == 1
    assert len(_notices(outage)) == 1


def test_pinned_outage_session_returns_to_its_pick_not_the_configured_model(outage):
    pin = {"model": PINNED_MODEL, "provider": PINNED_PROVIDER}
    session, agent = _open_outage_session(outage, model_override=dict(pin))
    assert session["auth_fallback_primary"] == pin

    _turn_start(outage, session)
    assert agent.switches == []

    outage.state["primary_down"] = False
    _turn_start(outage, session)
    assert agent.switches == [(PINNED_MODEL, PINNED_PROVIDER)]
    assert (agent.model, agent.provider) == (PINNED_MODEL, PINNED_PROVIDER)
    assert session["model_override"] == pin
    assert "auth_fallback_primary" not in session
    assert "auth_fallback_primary" not in _persisted_model_config(agent)
    assert len(_notices(outage)) == 1
    assert f"{PINNED_MODEL} via {PINNED_PROVIDER}" in _notices(outage)[0]

    _turn_start(outage, session)
    assert len(agent.switches) == 1


def test_outage_session_persists_the_marker_until_restored(outage):
    session, agent = _open_outage_session(outage)
    outage.server._persist_live_session_runtime(session)
    persisted = _persisted_model_config(agent)
    assert persisted["auth_fallback_primary"] == {"model": PRIMARY_MODEL, "provider": PRIMARY_PROVIDER}
    assert persisted["model"] == FALLBACK_MODEL


def test_explicit_model_pick_during_the_outage_is_kept_after_the_login_recovers(outage):
    session, agent = _open_outage_session(outage)

    outage.server._apply_model_switch(SID, session, f"{PINNED_MODEL} --provider {PINNED_PROVIDER}", confirm_expensive_model=True)
    assert session["model_override"]["model"] == PINNED_MODEL
    assert "auth_fallback_primary" not in session
    assert "auth_fallback_primary" not in _persisted_model_config(agent)

    outage.state["primary_down"] = False
    _turn_start(outage, session)
    assert agent.switches == [(PINNED_MODEL, PINNED_PROVIDER)]
    assert (agent.model, agent.provider) == (PINNED_MODEL, PINNED_PROVIDER)
    assert _notices(outage) == []


def test_model_once_does_not_end_the_wait_for_the_primary(outage):
    session, _agent = _open_outage_session(outage)

    outage.server._apply_model_switch(SID, session, f"{PINNED_MODEL} --provider {PINNED_PROVIDER} --once", confirm_expensive_model=True)
    assert "model_override" not in session
    assert session["auth_fallback_primary"] == {"model": PRIMARY_MODEL, "provider": PRIMARY_PROVIDER}


def test_resumed_outage_row_is_pinned_to_the_intended_route_not_the_fallback(outage):
    stored = outage.server._stored_session_runtime_overrides
    outage_row = {"model": FALLBACK_MODEL, "model_config": json.dumps({
        "model": FALLBACK_MODEL, "provider": FALLBACK_PROVIDER, "base_url": "https://fallback.invalid/v1",
        "api_mode": "chat_completions", "reasoning_config": {"enabled": True, "effort": "high"},
        "service_tier": "priority", "auth_fallback_primary": {"model": PRIMARY_MODEL, "provider": PRIMARY_PROVIDER}})}
    overrides = stored(outage_row)
    assert overrides["model_override"] == {
        "model": PRIMARY_MODEL, "provider": PRIMARY_PROVIDER, "base_url": None, "api_mode": None}
    assert overrides["provider_override"] == PRIMARY_PROVIDER
    assert overrides["reasoning_config_override"] == {"enabled": True, "effort": "high"}
    assert overrides["service_tier_override"] == "priority"

    chosen_row = {"model": FALLBACK_MODEL, "model_config": json.dumps({"model": FALLBACK_MODEL, "provider": FALLBACK_PROVIDER})}
    chosen = stored(chosen_row)
    assert chosen["model_override"]["model"] == FALLBACK_MODEL
    assert chosen["provider_override"] == FALLBACK_PROVIDER


class _StopTurn(Exception):
    pass


def _admit_turn(monkeypatch, one_turn_restore):
    """Run ``_prepare_turn_input`` up to the fallback-chain sync, recording which model steps ran, in order."""
    from tui_gateway import server

    ran: list = []
    monkeypatch.setattr(server, "_profile_runtime_scope_tokens", lambda profile_home: None)
    monkeypatch.setattr(server, "_set_session_context", lambda *a, **k: [])
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    for name in ("_apply_pending_model_switch", "_restore_auth_fallback_primary",
                 "_sync_agent_model_with_config", "_sync_agent_compression_with_config"):
        monkeypatch.setattr(server, name, lambda sid, session, _name=name: ran.append(_name))

    def stop(sid, session):
        raise _StopTurn()

    monkeypatch.setattr(server, "_sync_agent_fallback_with_config", stop)
    st = server._TurnRun(agent=None, one_turn_restore=one_turn_restore, terminal_callback=None, receipt_committed=False)
    with pytest.raises(_StopTurn):
        server._prepare_turn_input("sid", {"session_key": "key"}, st, "hello", [])
    return ran


def test_turn_admission_restores_the_primary_after_the_pending_pick_and_before_the_config_sync(monkeypatch):
    assert _admit_turn(monkeypatch, one_turn_restore=None) == [
        "_apply_pending_model_switch", "_restore_auth_fallback_primary",
        "_sync_agent_model_with_config", "_sync_agent_compression_with_config"]
    # A /model --once turn runs on its one-shot model: no restore (nor sync) may clobber it before the turn.
    assert _admit_turn(monkeypatch, one_turn_restore={"model": "once-model"}) == []
