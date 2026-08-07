"""
飞书（Feishu / Lark）通道插件。 / Feishu (Lark) channel plugin.

职责 / Responsibilities:
  - parse_message(): 飞书 webhook 事件 (v2 格式) → InboundEnvelope
  - send(): OutboundEnvelope → 飞书 im/v1/messages API，自动分块
  - extract_challenge(): 识别 URL 验证请求，返回 challenge 串供 webhook 注册
  - _get_token(): 自动获取并缓存 tenant_access_token（2 小时有效，临近过期刷新）

实现要点 / Design notes:
  - 不依赖 lark-oapi，用 httpx（已有依赖）直接调飞书 REST API
  - httpx.AsyncClient 生命周期与 start()/stop() 绑定，复用连接，不每次新建
  - open_id（"ou_" 开头）→ receive_id_type=open_id；chat_id（"oc_" 开头）→ receive_id_type=chat_id
  - 私聊（chat_type=="p2p"）自动 mentions_bot=True；群聊检查 mentions 列表里是否有 bot open_id
  - 超长消息分块复用 telegram._split_long_text（换行优先，硬切兜底）

面试话术：
"飞书适配器和 Telegram/Discord 结构一模一样——parse_message + send + start/stop。
差异在三处：tenant_access_token 需要用 app_id+app_secret 换，有效期 2 小时，
临近过期（留 60 秒 buffer）自动刷新，发送时带 Bearer；
发送时根据 receive_id 前缀（ou_ 是用户 open_id，oc_ 是群聊 chat_id）
自动选不同的 receive_id_type，调用方不需要关心；
webhook 注册时飞书发一个 challenge 请求，extract_challenge() 识别并原样返回。
Gateway 层完全不感知这些——注册进去就能用，和 Telegram/Discord 对等。"
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from nanoharness.channels.base import (
    ChannelSendResult, ChatType, InboundEnvelope, OutboundEnvelope, SendStatus,
)
from nanoharness.channels.telegram import _split_long_text

log = logging.getLogger(__name__)

FEISHU_BASE_URL = "https://open.feishu.cn/open-apis"
FEISHU_MAX_LEN  = 4000     # 保守上限，飞书文本实际限制约 150 000 字符，分块避免超大回复 / conservative limit; Feishu text cap ~150k chars, chunking avoids oversized replies
_TOKEN_REFRESH_BUFFER = 60  # 提前 60 秒刷新 token / refresh token 60 s before expiry


class FeishuChannel:
    """
    飞书通道插件。外部注入 app_id / app_secret，便于测试替换。 / Feishu channel plugin. app_id/app_secret injected from outside for test substitution.
    """

    channel_id = "feishu"

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        bot_open_id: str = "",
    ) -> None:
        """
        Parameters
        ----------
        app_id / app_secret
            飞书应用凭证，用于换取 tenant_access_token。 / Feishu app credentials for obtaining tenant_access_token.
        bot_open_id
            机器人自己的 open_id（"ou_" 开头），群聊 @mention 检测用。留空则跳过检测。 / Bot's own open_id (starts with "ou_"), used for @mention detection in group chats. Leave empty to skip detection.
        """
        self._app_id       = app_id
        self._app_secret   = app_secret
        self._bot_open_id  = bot_open_id
        self._token        = ""
        self._token_exp    = 0.0   # unix timestamp when current token expires
        self._http: Any    = None  # httpx.AsyncClient，start() 时创建 / created in start()

    # ── 生命周期 / Lifecycle ──────────────────────────────────────────────────

    async def start(self) -> None:
        """创建 HTTP 连接池并预热 token。 / Create HTTP connection pool and warm up token."""
        import httpx
        self._http = httpx.AsyncClient(timeout=15.0)
        try:
            await self._get_token()
        except Exception as exc:
            log.warning("飞书 token 预热失败（将在首次发送时重试）: %s", exc)
        log.info("飞书通道已启动（app_id=%s）", self._app_id)

    async def stop(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        log.info("飞书通道已停止")

    # ── Webhook 工具 / Webhook utilities ─────────────────────────────────────

    @staticmethod
    def extract_challenge(event_data: dict) -> str | None:
        """
        如果 event_data 是飞书 URL 验证请求，返回 challenge 字符串（调用方原样回包即可）；否则返回 None。 / If event_data is a Feishu URL-verification request, return the challenge string (caller echoes it back); otherwise return None.

        Usage in webhook handler::

            data = await request.json()
            if (ch := FeishuChannel.extract_challenge(data)):
                return {"challenge": ch}
            envelope = feishu_channel.parse_message(data)
            ...
        """
        if event_data.get("type") == "url_verification":
            return event_data.get("challenge")
        return None

    # ── 消息解析 / Message parsing ────────────────────────────────────────────

    def parse_message(self, raw: Any) -> InboundEnvelope:
        """
        飞书 v2 事件字典 → InboundEnvelope。 / Feishu v2 event dict → InboundEnvelope.

        raw 是 webhook POST body 解析后的 dict（非加密模式）。 / raw is the parsed dict from the webhook POST body (non-encrypted mode).
        """
        event   = raw.get("event", {})
        message = event.get("message", {})
        sender  = event.get("sender", {})

        # 发送者 open_id / sender open_id
        sender_ids = sender.get("sender_id", {})
        open_id = sender_ids.get("open_id") or sender_ids.get("union_id") or ""

        # 会话信息 / chat info
        chat_id      = message.get("chat_id", "")
        chat_type_raw = message.get("chat_type", "p2p")
        chat_type = ChatType.DIRECT if chat_type_raw == "p2p" else ChatType.GROUP

        # 消息内容（content 是 JSON 字符串）/ message content (content is a JSON-encoded string)
        content_text = _parse_content(message.get("content", "{}"))

        # @机器人判定 / @bot detection
        # 私聊一定是对机器人说话；群聊检查 mentions 列表 / p2p always targets the bot; group checks mentions list
        mentions_bot = (chat_type == ChatType.DIRECT)
        if not mentions_bot and self._bot_open_id:
            for m in message.get("mentions", []):
                if m.get("id", {}).get("open_id") == self._bot_open_id:
                    mentions_bot = True
                    break

        return InboundEnvelope(
            channel_id=self.channel_id,
            sender_id=open_id,
            chat_id=chat_id,
            chat_type=chat_type,
            content=content_text,
            mentions_bot=mentions_bot,
            raw=raw,
        )

    # ── 消息发送 / Message sending ────────────────────────────────────────────

    async def send(self, envelope: OutboundEnvelope) -> ChannelSendResult:
        """
        发送出站消息，自动分块（FEISHU_MAX_LEN 上限）。 / Send outbound message, auto-chunked to FEISHU_MAX_LEN.

        receive_id_type 根据前缀自动判定：
          "oc_" → chat_id（群聊）/ "oc_" → chat_id (group chat)
          其余  → open_id（用户私聊）/ others → open_id (user DM)
        """
        chunks = _split_long_text(envelope.content, FEISHU_MAX_LEN)
        target = envelope.target_peer
        receive_id_type = "chat_id" if target.startswith("oc_") else "open_id"

        try:
            token = await self._get_token()
        except Exception as exc:
            return ChannelSendResult(
                status=SendStatus.FAILED, retryable=True,
                reason=f"token 获取失败: {exc}", message_ids=[],
            )

        if self._http is None:
            import httpx
            self._http = httpx.AsyncClient(timeout=15.0)

        headers   = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        sent_ids: list[str] = []

        for chunk in chunks:
            body = {
                "receive_id": target,
                "content":    json.dumps({"text": chunk}, ensure_ascii=False),
                "msg_type":   "text",
            }
            try:
                resp = await self._http.post(
                    f"{FEISHU_BASE_URL}/im/v1/messages",
                    params={"receive_id_type": receive_id_type},
                    headers=headers,
                    content=json.dumps(body, ensure_ascii=False).encode(),
                )
                data = resp.json()
                if data.get("code") == 0:
                    sent_ids.append(data.get("data", {}).get("message_id", ""))
                elif resp.status_code == 429 or data.get("code") == 99991429:
                    return ChannelSendResult(
                        status=SendStatus.RATE_LIMITED, retryable=True,
                        reason=f"限流: {data}", message_ids=sent_ids,
                    )
                else:
                    log.warning("飞书发送失败: %s", data)
                    return ChannelSendResult(
                        status=SendStatus.FAILED, retryable=False,
                        reason=str(data), message_ids=sent_ids,
                    )
            except Exception as exc:
                log.warning("飞书 HTTP 异常: %s", exc)
                return ChannelSendResult(
                    status=SendStatus.FAILED, retryable=False,
                    reason=str(exc), message_ids=sent_ids,
                )

        return ChannelSendResult(
            status=SendStatus.SENT, retryable=False,
            reason="", message_ids=sent_ids,
        )

    # ── Token 管理 / Token management ────────────────────────────────────────

    async def _get_token(self) -> str:
        """
        获取 tenant_access_token，临近过期（60 秒内）自动刷新。 / Obtain tenant_access_token; auto-refresh within 60 s of expiry.

        飞书 token 有效期 7200 秒（2 小时），提前 60 秒刷新避免请求途中过期。 / Feishu tokens expire in 7200 s (2 h); refresh 60 s early to avoid mid-request expiry.
        """
        if self._token and time.time() < self._token_exp - _TOKEN_REFRESH_BUFFER:
            return self._token

        if self._http is None:
            import httpx
            self._http = httpx.AsyncClient(timeout=15.0)

        resp = await self._http.post(
            f"{FEISHU_BASE_URL}/auth/v3/tenant_access_token/internal",
            json={"app_id": self._app_id, "app_secret": self._app_secret},
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"飞书 token 接口返回错误: {data}")

        self._token     = data["tenant_access_token"]
        self._token_exp = time.time() + int(data.get("expire", 7200))
        log.debug("飞书 token 已刷新，有效期 %s 秒", data.get("expire", 7200))
        return self._token


# ── 辅助函数 / Helpers ────────────────────────────────────────────────────────

def _parse_content(content_json: str) -> str:
    """
    把飞书消息 content 字段（JSON 字符串）转成纯文本。 / Convert Feishu message content field (JSON string) to plain text.

    飞书文本消息 content 格式：'{"text": "hello @bot"}' / Feishu text message content format: '{"text": "hello @bot"}'
    富文本（post）格式更复杂，这里只提取 text 字段，够面试 demo 用。 / Rich text (post) format is more complex; only extracts the text field, sufficient for demo.
    """
    if not content_json:
        return ""
    try:
        obj = json.loads(content_json)
    except (json.JSONDecodeError, TypeError):
        return content_json  # 解析失败当纯文本 / treat as plain text on parse failure
    # 优先取 text 字段；如果是 post 富文本则拼接所有 paragraph 的文字 / prefer text field; for post rich text, concatenate all paragraph texts
    if "text" in obj:
        return str(obj["text"])
    # post 格式: {"title": "...", "content": [[{"tag":"text","text":"..."}]]}
    parts: list[str] = []
    for para in obj.get("content", []):
        for seg in (para if isinstance(para, list) else []):
            if isinstance(seg, dict) and seg.get("tag") == "text":
                parts.append(seg.get("text", ""))
    return "".join(parts)
