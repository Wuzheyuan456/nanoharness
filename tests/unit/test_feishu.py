"""
飞书通道插件单元测试 / Feishu channel plugin unit tests.

测试策略 / Test strategy:
  - parse_message 用原始 dict 构造 webhook 事件，不需要真实 HTTP / parse_message uses raw dicts to build webhook events, no real HTTP
  - send / _get_token 用 FakeHttpxClient 替换 self._http / send / _get_token uses FakeHttpxClient to replace self._http
  - 行为指纹风格：断言字段值和调用参数，不断言文本 / behavior-fingerprint style: assert field values and call args, not text
"""
from __future__ import annotations

import json
import time
import pytest

from nanoharness.channels.base import ChatType, OutboundEnvelope, SendStatus
from nanoharness.channels.feishu import FeishuChannel, _parse_content


# ─── Fakes ────────────────────────────────────────────────────────────────────

class FakeResponse:
    def __init__(self, data: dict, status_code: int = 200) -> None:
        self._data       = data
        self.status_code = status_code

    def json(self) -> dict:
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class FakeHttpxClient:
    """轻量级 httpx.AsyncClient 替身，按顺序返回预设 response。 / Lightweight httpx.AsyncClient stand-in that returns preset responses in order."""

    def __init__(self, *responses: dict | tuple) -> None:
        # 支持 dict（默认 200）或 (dict, status_code) 元组 / accept dict (default 200) or (dict, status_code) tuple
        self._responses: list[tuple[dict, int]] = []
        for r in responses:
            if isinstance(r, tuple):
                self._responses.append((r[0], r[1]))
            else:
                self._responses.append((r, 200))
        self._idx  = 0
        self.calls: list[dict] = []

    async def post(self, url: str, **kwargs) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        data, code = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return FakeResponse(data, code)

    async def aclose(self) -> None:
        pass


def _token_resp(token: str = "t-test", expire: int = 7200) -> dict:
    return {"code": 0, "tenant_access_token": token, "expire": expire}


def _send_ok(msg_id: str = "om_001") -> dict:
    return {"code": 0, "data": {"message_id": msg_id}}


def _feishu_with_token(token: str = "t-test", bot_open_id: str = "ou_bot") -> FeishuChannel:
    """Helper: FeishuChannel with a valid pre-seeded token so tests don't hit the token endpoint. / 辅助函数：预置有效 token，避免测试触发 token 端点。"""
    ch = FeishuChannel("app_id", "app_secret", bot_open_id=bot_open_id)
    ch._token     = token
    ch._token_exp = time.time() + 3600
    return ch


# ─── Webhook 事件构造器 / Webhook event builders ──────────────────────────────

def _p2p_event(text: str, sender_open_id: str = "ou_user", chat_id: str = "ou_user") -> dict:
    """构造私聊消息事件。 / Build a private chat message event."""
    return {
        "schema": "2.0",
        "event": {
            "message": {
                "chat_id":    chat_id,
                "chat_type":  "p2p",
                "message_id": "om_test",
                "content":    json.dumps({"text": text}),
                "mentions":   [],
            },
            "sender": {
                "sender_id": {"open_id": sender_open_id},
                "sender_type": "user",
            },
        },
    }


def _group_event(
    text: str,
    sender_open_id: str = "ou_user",
    chat_id: str = "oc_group",
    bot_open_id: str | None = None,
) -> dict:
    """构造群聊消息事件，可选 @bot。 / Build a group chat message event, optionally @bot."""
    mentions = []
    if bot_open_id:
        mentions.append({
            "key": "@_user_xxx",
            "id": {"open_id": bot_open_id},
            "name": "Nano",
        })
    return {
        "schema": "2.0",
        "event": {
            "message": {
                "chat_id":    chat_id,
                "chat_type":  "group",
                "message_id": "om_test",
                "content":    json.dumps({"text": text}),
                "mentions":   mentions,
            },
            "sender": {
                "sender_id": {"open_id": sender_open_id},
            },
        },
    }


# ─── extract_challenge ─────────────────────────────────────────────────────────

def test_extract_challenge_returns_value():
    data = {"type": "url_verification", "challenge": "abc123", "token": "tok"}
    assert FeishuChannel.extract_challenge(data) == "abc123"


def test_extract_challenge_returns_none_for_normal_event():
    data = {"schema": "2.0", "event": {}}
    assert FeishuChannel.extract_challenge(data) is None


# ─── parse_message: 私聊 / parse_message: private chat ───────────────────────

def test_parse_p2p_chat_type_is_direct():
    ch  = FeishuChannel("app_id", "app_secret")
    env = ch.parse_message(_p2p_event("hello", sender_open_id="ou_alice"))
    assert env.chat_type == ChatType.DIRECT


def test_parse_p2p_mentions_bot_always_true():
    """私聊消息不论是否 @，mentions_bot 都应为 True。 / p2p messages always have mentions_bot=True regardless of @."""
    ch  = FeishuChannel("app_id", "app_secret")
    env = ch.parse_message(_p2p_event("just a message"))
    assert env.mentions_bot is True


def test_parse_p2p_sender_id_extracted():
    ch  = FeishuChannel("app_id", "app_secret")
    env = ch.parse_message(_p2p_event("hi", sender_open_id="ou_alice"))
    assert env.sender_id == "ou_alice"


def test_parse_p2p_content_extracted():
    ch  = FeishuChannel("app_id", "app_secret")
    env = ch.parse_message(_p2p_event("what time is it"))
    assert env.content == "what time is it"


def test_parse_p2p_channel_id_is_feishu():
    ch  = FeishuChannel("app_id", "app_secret")
    env = ch.parse_message(_p2p_event("hi"))
    assert env.channel_id == "feishu"


# ─── parse_message: 群聊 / parse_message: group chat ─────────────────────────

def test_parse_group_chat_type_is_group():
    ch  = FeishuChannel("app_id", "app_secret")
    env = ch.parse_message(_group_event("hello"))
    assert env.chat_type == ChatType.GROUP


def test_parse_group_mentions_bot_true_when_mentioned():
    ch  = FeishuChannel("app_id", "app_secret", bot_open_id="ou_bot")
    env = ch.parse_message(_group_event("@Nano hi", bot_open_id="ou_bot"))
    assert env.mentions_bot is True


def test_parse_group_mentions_bot_false_when_not_mentioned():
    ch  = FeishuChannel("app_id", "app_secret", bot_open_id="ou_bot")
    env = ch.parse_message(_group_event("general discussion"))
    assert env.mentions_bot is False


def test_parse_group_mentions_bot_false_when_bot_open_id_not_set():
    """bot_open_id 未配置时，群聊 @mention 判定保守地返回 False。 / When bot_open_id is not configured, group mention detection conservatively returns False."""
    ch  = FeishuChannel("app_id", "app_secret", bot_open_id="")
    env = ch.parse_message(_group_event("hi", bot_open_id="ou_bot"))
    assert env.mentions_bot is False


def test_parse_group_chat_id_extracted():
    ch  = FeishuChannel("app_id", "app_secret")
    env = ch.parse_message(_group_event("hi", chat_id="oc_abc123"))
    assert env.chat_id == "oc_abc123"


# ─── _parse_content ───────────────────────────────────────────────────────────

def test_parse_content_text_message():
    assert _parse_content('{"text": "hello"}') == "hello"


def test_parse_content_empty_string():
    assert _parse_content("") == ""


def test_parse_content_invalid_json_returns_raw():
    raw = "not json"
    assert _parse_content(raw) == raw


def test_parse_content_post_rich_text():
    """post 格式：提取所有段落的 text 标签。 / post format: extract text tags from all paragraphs."""
    post = {
        "title": "Report",
        "content": [
            [{"tag": "text", "text": "Hello "}, {"tag": "a", "href": "http://x.com", "text": "link"}],
            [{"tag": "text", "text": "world"}],
        ],
    }
    result = _parse_content(json.dumps(post))
    assert "Hello" in result
    assert "world" in result


# ─── send: receive_id_type 自动推断 / send: receive_id_type auto-detection ─────

@pytest.mark.asyncio
async def test_send_user_open_id_uses_open_id_type():
    """"ou_" 开头的 target_peer 应使用 receive_id_type=open_id。 / target_peer starting with "ou_" should use receive_id_type=open_id."""
    ch = _feishu_with_token()
    ch._http = FakeHttpxClient(_send_ok())
    await ch.send(OutboundEnvelope(target_channel="feishu", target_peer="ou_user123", content="hi"))
    call = ch._http.calls[0]
    assert call["params"]["receive_id_type"] == "open_id"


@pytest.mark.asyncio
async def test_send_group_chat_id_uses_chat_id_type():
    """"oc_" 开头的 target_peer 应使用 receive_id_type=chat_id。 / target_peer starting with "oc_" should use receive_id_type=chat_id."""
    ch = _feishu_with_token()
    ch._http = FakeHttpxClient(_send_ok())
    await ch.send(OutboundEnvelope(target_channel="feishu", target_peer="oc_group123", content="hello"))
    call = ch._http.calls[0]
    assert call["params"]["receive_id_type"] == "chat_id"


@pytest.mark.asyncio
async def test_send_returns_sent_on_success():
    ch = _feishu_with_token()
    ch._http = FakeHttpxClient(_send_ok("om_abc"))
    result = await ch.send(OutboundEnvelope(target_channel="feishu", target_peer="ou_x", content="ok"))
    assert result.status == SendStatus.SENT
    assert "om_abc" in result.message_ids


@pytest.mark.asyncio
async def test_send_returns_rate_limited_on_429():
    ch = _feishu_with_token()
    ch._http = FakeHttpxClient(({"code": 99991429, "msg": "rate"}, 429))
    result = await ch.send(OutboundEnvelope(target_channel="feishu", target_peer="ou_x", content="hi"))
    assert result.status == SendStatus.RATE_LIMITED
    assert result.retryable is True


@pytest.mark.asyncio
async def test_send_returns_failed_on_api_error():
    ch = _feishu_with_token()
    ch._http = FakeHttpxClient({"code": 99991400, "msg": "invalid app_id"})
    # 99991400 is not in our rate-limit list → FAILED
    result = await ch.send(OutboundEnvelope(target_channel="feishu", target_peer="ou_x", content="hi"))
    assert result.status == SendStatus.FAILED


@pytest.mark.asyncio
async def test_send_splits_long_message():
    """超长消息应分多次调用 API。 / Long messages should result in multiple API calls."""
    long_text = ("a" * 4001)
    ch = _feishu_with_token()
    ch._http = FakeHttpxClient(_send_ok("om_1"), _send_ok("om_2"))
    result = await ch.send(OutboundEnvelope(target_channel="feishu", target_peer="ou_x", content=long_text))
    assert result.status == SendStatus.SENT
    assert len(ch._http.calls) == 2


@pytest.mark.asyncio
async def test_send_http_exception_returns_failed():
    class ErrorClient:
        async def post(self, *a, **kw):
            raise OSError("connection refused")
        async def aclose(self): pass

    ch = _feishu_with_token()
    ch._http = ErrorClient()
    result = await ch.send(OutboundEnvelope(target_channel="feishu", target_peer="ou_x", content="hi"))
    assert result.status == SendStatus.FAILED


# ─── _get_token ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_token_fetches_and_caches():
    ch = FeishuChannel("app_id", "app_secret")
    ch._http = FakeHttpxClient(_token_resp("t-fresh", expire=7200))
    token = await ch._get_token()
    assert token == "t-fresh"
    # Second call should be served from cache, not hitting HTTP again
    token2 = await ch._get_token()
    assert token2 == "t-fresh"
    assert len(ch._http.calls) == 1   # only 1 HTTP call total


@pytest.mark.asyncio
async def test_get_token_refreshes_near_expiry():
    ch = FeishuChannel("app_id", "app_secret")
    # Seed an expired token
    ch._token     = "t-old"
    ch._token_exp = time.time() + 30   # expires in 30 s < 60 s buffer → should refresh
    ch._http = FakeHttpxClient(_token_resp("t-new"))
    token = await ch._get_token()
    assert token == "t-new"


@pytest.mark.asyncio
async def test_get_token_returns_cached_when_valid():
    ch = FeishuChannel("app_id", "app_secret")
    ch._token     = "t-valid"
    ch._token_exp = time.time() + 3600   # expires in 1 hour → no refresh needed
    ch._http = FakeHttpxClient()   # empty: any call would fail
    token = await ch._get_token()
    assert token == "t-valid"
    assert len(ch._http.calls) == 0


@pytest.mark.asyncio
async def test_get_token_raises_on_api_error():
    ch = FeishuChannel("app_id", "app_secret")
    ch._http = FakeHttpxClient({"code": 10003, "msg": "invalid app secret"})
    with pytest.raises(RuntimeError, match="飞书 token"):
        await ch._get_token()


# ─── 生命周期 / Lifecycle ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_start_warms_token(monkeypatch):
    """start() 应预热 token（允许失败，不 crash）。 / start() should warm up the token (failure is allowed, no crash)."""
    import httpx

    class MockAsyncClient:
        def __init__(self, **kwargs): pass
        async def post(self, *a, **kw):
            return FakeResponse(_token_resp())
        async def aclose(self): pass

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)
    ch = FeishuChannel("app_id", "app_secret")
    await ch.start()
    assert ch._token == "t-test"
    await ch.stop()
    assert ch._http is None
