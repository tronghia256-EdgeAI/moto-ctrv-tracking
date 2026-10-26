"""
code/alert/telegram_alert.py
──────────────────────────────
Telegram bot alert for accident detection.

Only called when a track passes the THRESHOLD_ACCIDENT reliability
threshold AND EfficientNet classifies severity as MINOR or SEVERE.

Rate limiting (config: telegram.rate_limit_seconds) suppresses repeat
alerts from the same camera. dry_run=True logs instead of sending —
every ablation experiment runs dry so a metrics sweep never spams the
real chat.

Author  : Trọng Nghĩa
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class TelegramAlert:
    """
    Sends formatted alerts to a Telegram chat.

    Usage:
        alert = TelegramAlert.from_config("configs/default.yaml" config dict)
        await alert.send(severity, frame, location)
    """

    def __init__(self, token: str, chat_id: str,
                 rate_limit_seconds: float = 10.0,
                 dry_run: bool = False) -> None:
        self.chat_id = chat_id
        self.rate_limit_seconds = rate_limit_seconds
        self.dry_run = dry_run
        self._last_sent: dict[str, float] = {}   # location → timestamp

        if dry_run:
            self.bot = None
        else:
            from telegram import Bot  # lazy import
            self.bot = Bot(token=token)

    @classmethod
    def from_config(cls, config: dict, dry_run: bool = False) -> "TelegramAlert":
        """Build from the main config dict; credentials live in the
        gitignored file referenced by telegram.credentials_file."""
        tele_cfg = config.get("telegram", {})
        rate = tele_cfg.get("rate_limit_seconds", 10)
        cred_file = tele_cfg.get("credentials_file", "configs/tele_api.yaml")

        token, chat_id = "", ""
        if Path(cred_file).exists():
            with open(cred_file, encoding="utf-8") as f:
                creds = yaml.safe_load(f)["telegram"]
            token, chat_id = creds["token"], str(creds["chat_id"])
        elif not dry_run:
            logger.warning("Credentials file %s missing — forcing dry_run.",
                           cred_file)
            dry_run = True

        return cls(token=token, chat_id=chat_id,
                   rate_limit_seconds=rate, dry_run=dry_run)

    async def send(
        self,
        severity,          # Severity enum
        frame: np.ndarray, # frame to attach as photo
        location: str = "Camera 01",
    ) -> bool:
        """
        Send an alert message with a snapshot. Returns True if the alert
        was sent (or would have been sent in dry_run), False if it was
        rate-limited.
        """
        now = time.monotonic()
        last = self._last_sent.get(location)
        if last is not None and now - last < self.rate_limit_seconds:
            return False
        self._last_sent[location] = now

        caption = (f"🚨 TAI NẠN — mức độ: {severity.name}\n"
                   f"📍 {location}\n"
                   f"🕐 {time.strftime('%Y-%m-%d %H:%M:%S')}")

        if self.dry_run:
            logger.info("[dry_run] Telegram alert: %s", caption.replace("\n", " | "))
            return True

        import cv2
        ok, jpeg = cv2.imencode(".jpg", frame)
        assert ok, "JPEG encoding failed"
        await self.bot.send_photo(chat_id=self.chat_id,
                                  photo=bytes(jpeg.tobytes()),
                                  caption=caption)
        return True
