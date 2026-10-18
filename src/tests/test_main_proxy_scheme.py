from typing import Any

import pytest
from flask import Flask, request
from waitress.server import BaseWSGIServer, create_server
from werkzeug.test import Client
from werkzeug.wrappers import Response

import main
from app.feeds import _get_base_url


@pytest.fixture
def serve_kwargs(monkeypatch):
    def run(trusted_proxy: str | None) -> dict[str, Any]:
        if trusted_proxy is None:
            monkeypatch.delenv("PODLY_TRUSTED_PROXY", raising=False)
        else:
            monkeypatch.setenv("PODLY_TRUSTED_PROXY", trusted_proxy)
        captured: dict[str, Any] = {}
        monkeypatch.setattr(main, "create_web_app", lambda: Flask(__name__))
        monkeypatch.setattr(
            main, "serve", lambda _app, **kwargs: captured.update(kwargs)
        )
        main.main()
        return captured

    return run


@pytest.fixture
def proxy_client(serve_kwargs):
    """Client for the WSGI app Waitress builds from main()'s server options."""
    servers = []

    def create(trusted_proxy: str | None) -> Client:
        kwargs = serve_kwargs(trusted_proxy) | {"host": "127.0.0.1", "port": 0}
        app = Flask(__name__)

        @app.get("/feed")
        def feed():
            return {"base_url": _get_base_url(), "peer": request.remote_addr}

        server = create_server(app, map={}, _start=False, **kwargs)
        assert isinstance(server, BaseWSGIServer)
        servers.append(server)
        return Client(server.application, Response)

    yield create
    for server in servers:
        server.close()


def _get_feed(client: Client, peer: str, **headers: str) -> dict[str, Any]:
    response = client.get(
        "/feed", headers=headers, environ_overrides={"REMOTE_ADDR": peer}
    )
    assert response.status_code == 200
    body = response.json
    assert isinstance(body, dict)
    return body


def test_trusted_proxy_generates_https_feed_urls(proxy_client):
    body = _get_feed(
        proxy_client("172.20.0.1"), "172.20.0.1", **{"X-Forwarded-Proto": "https"}
    )
    assert body == {"base_url": "https://localhost", "peer": "172.20.0.1"}


def test_untrusted_peer_cannot_set_forwarded_proto(proxy_client):
    body = _get_feed(
        proxy_client("172.20.0.1"), "203.0.113.7", **{"X-Forwarded-Proto": "https"}
    )
    assert body == {"base_url": "http://localhost", "peer": "203.0.113.7"}


@pytest.mark.parametrize("setting", [None, "", "   "])
def test_default_does_not_trust_forwarded_proto(proxy_client, setting):
    body = _get_feed(
        proxy_client(setting), "172.20.0.1", **{"X-Forwarded-Proto": "https"}
    )
    assert body["base_url"] == "http://localhost"


def test_trusted_proxy_without_scheme_keeps_http(proxy_client):
    body = _get_feed(proxy_client("172.20.0.1"), "172.20.0.1")
    assert body["base_url"] == "http://localhost"


def test_trusted_proxy_ignores_forwarded_address_and_host(proxy_client):
    body = _get_feed(
        proxy_client("172.20.0.1"),
        "172.20.0.1",
        **{
            "X-Forwarded-Proto": "https",
            "X-Forwarded-For": "203.0.113.9",
            "X-Forwarded-Host": "attacker.example",
        },
    )
    assert body == {"base_url": "https://localhost", "peer": "172.20.0.1"}


def test_trusted_proxy_logs_untrusted_headers(serve_kwargs):
    assert serve_kwargs("172.20.0.1")["log_untrusted_proxy_headers"] is True
    assert serve_kwargs(None)["log_untrusted_proxy_headers"] is False


@pytest.mark.parametrize(
    "setting",
    ["*", "172.20.0.0/16", "127.0.0.1,::1", "localhost", "::1", "::ffff:172.20.0.1"],
)
def test_invalid_proxy_trust_is_rejected(serve_kwargs, setting):
    with pytest.raises(ValueError, match="PODLY_TRUSTED_PROXY must be the IPv4"):
        serve_kwargs(setting)
