from __future__ import annotations

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand

from allgoodpy.config import Settings

from .handlers import router
from .middleware import SettingsMiddleware


async def _run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    settings = Settings()
    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    await bot.set_my_commands(
        [
            BotCommand(command="start", description="Справка"),
            BotCommand(command="topic_id", description="Chat и topic id для .env"),
            BotCommand(command="scan", description="Режим сканирования"),
            BotCommand(command="cancel", description="Отмена"),
        ]
    )

    dp = Dispatcher(storage=MemoryStorage())
    dp.update.outer_middleware(SettingsMiddleware(settings))
    dp.include_router(router)

    await dp.start_polling(bot)


def main() -> None:
    asyncio.run(_run())
