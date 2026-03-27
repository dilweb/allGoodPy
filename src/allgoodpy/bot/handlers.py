from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from io import BytesIO

from aiogram import F, Router
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import KeyboardButton, Message, ReplyKeyboardMarkup

from allgoodpy.config import Settings
from allgoodpy.recognition.pipeline import recognize_image_bytes
from allgoodpy.sheets.writer import append_processing_row

from .states import ScanStates

logger = logging.getLogger(__name__)

router = Router(name="scan")


def _main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="Сканировать")]],
        resize_keyboard=True,
    )


def _topic_ok(settings: Settings, message: Message) -> bool:
    if settings.forum_topic_id is None:
        return True
    tid = message.message_thread_id
    if tid is None:
        return False
    return tid == settings.forum_topic_id


def _chat_ok(settings: Settings, message: Message) -> bool:
    if settings.allowed_chat_id is None:
        return True
    return message.chat.id == settings.allowed_chat_id


def _public_message_link(chat_id: int, message_id: int) -> str:
    s = str(chat_id)
    if s.startswith("-100"):
        return f"https://t.me/c/{s[4:]}/{message_id}"
    return ""


async def _download_largest_photo(message: Message) -> bytes | None:
    photos = message.photo
    if not photos:
        return None
    bio = BytesIO()
    await message.bot.download(file=photos[-1], destination=bio)
    return bio.getvalue()


async def _download_document_image(message: Message) -> bytes | None:
    doc = message.document
    if doc is None or doc.mime_type is None:
        return None
    if not doc.mime_type.startswith("image/"):
        return None
    bio = BytesIO()
    await message.bot.download(file=doc, destination=bio)
    return bio.getvalue()


@router.message(Command("topic_id"))
async def cmd_topic_id(message: Message, settings: Settings) -> None:
    """Показывает chat id и message_thread_id текущей ветки (для .env)."""
    if not _chat_ok(settings, message):
        return
    chat_id = message.chat.id
    tid = message.message_thread_id
    lines = [
        f"<b>chat id</b>: <code>{chat_id}</code>",
        "Для .env: <code>ALLOWED_CHAT_ID={cid}</code>".format(cid=chat_id),
        "",
    ]
    if tid is None:
        lines.append(
            "<b>topic id</b>: Telegram не прислал номер ветки.\n\n"
            "Частая причина: ты в теме <b>«Основная» / General</b> — даже если её "
            "переименовали в «ШК», для API это всё ещё «главная» ветка, и "
            "<code>message_thread_id</code> там пустой.\n\n"
            "<b>Что сделать:</b> в меню группы открой список тем → "
            "<b>Создать тему</b> (новую, не переименовывай только первую) — например «ШК» — "
            "зайди в неё и снова отправь /topic_id.\n\n"
            "Если бот должен работать во <b>всех</b> ветках группы — в .env "
            "не задавай <code>FORUM_TOPIC_ID</code> (оставь пустым)."
        )
    else:
        lines.extend(
            [
                f"<b>topic id</b> (FORUM_TOPIC_ID): <code>{tid}</code>",
                f"Для .env: <code>FORUM_TOPIC_ID={tid}</code>",
            ]
        )
    await message.answer("\n".join(lines))


@router.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext, settings: Settings) -> None:
    if not _chat_ok(settings, message):
        return
    if not _topic_ok(settings, message):
        return
    await state.clear()
    await message.answer(
        "Бот распознаёт штрихкоды/QR и номера заказов на фото.\n"
        "Нажми «Сканировать» или отправь /scan, затем пришли фото в этот чат.",
        reply_markup=_main_keyboard(),
    )


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext, settings: Settings) -> None:
    if not _chat_ok(settings, message):
        return
    if not _topic_ok(settings, message):
        return
    await state.clear()
    await message.answer("Сброшено. Снова: /scan или «Сканировать».")


@router.message(Command("scan"))
@router.message(F.text == "Сканировать")
async def cmd_scan(message: Message, state: FSMContext, settings: Settings) -> None:
    if not _chat_ok(settings, message):
        return
    if not _topic_ok(settings, message):
        return
    await state.set_state(ScanStates.waiting_photo)
    await message.answer("Пришли фото этикетки (как фото или файлом-картинкой).")


async def _process_photo_message(
    message: Message,
    state: FSMContext,
    settings: Settings,
) -> None:
    if not _chat_ok(settings, message):
        return
    if not _topic_ok(settings, message):
        return

    raw: bytes | None = None
    if message.photo:
        raw = await _download_largest_photo(message)
    elif message.document:
        raw = await _download_document_image(message)

    if not raw:
        await message.reply("Не удалось скачать изображение. Попробуй ещё раз.")
        return

    try:
        result = await asyncio.to_thread(recognize_image_bytes, raw)
    except Exception:
        logger.exception("recognition failed")
        await message.reply("Ошибка при разборе изображения. Попробуй другое фото.")
        return

    lines = []
    if result.barcodes:
        lines.append("Штрихкоды / QR:")
        lines.extend(f"• {b}" for b in result.barcodes)
    else:
        lines.append("Штрихкоды / QR: не найдены")
    lines.append("")
    if result.order_numbers:
        lines.append("Номера заказов:")
        lines.extend(f"• {o}" for o in result.order_numbers)
    else:
        lines.append("Номера заказов: не найдены")

    reply = await message.reply("\n".join(lines))
    await state.clear()

    link = _public_message_link(message.chat.id, message.message_id)
    if not link and message.chat and message.chat.username:
        link = f"https://t.me/{message.chat.username}/{message.message_id}"

    user = message.from_user
    user_label = ""
    if user:
        parts = []
        if user.username:
            parts.append(f"@{user.username}")
        name = " ".join(x for x in (user.first_name, user.last_name) if x).strip()
        if name:
            parts.append(name)
        user_label = " ".join(parts) or str(user.id)

    try:
        await append_processing_row(
            spreadsheet_id=settings.google_sheets_spreadsheet_id,
            service_account_path=settings.google_service_account_file,
            sheet_name=settings.google_sheet_name,
            processed_at=datetime.now(timezone.utc),
            user_id=user.id if user else 0,
            user_label=user_label,
            order_numbers=result.order_numbers,
            barcodes=result.barcodes,
            message_link=link,
        )
    except Exception:
        logger.exception("google sheets append failed")
        await reply.reply("Распознавание готово, но запись в таблицу не удалась. Проверь доступ сервисного аккаунта.")


@router.message(StateFilter(ScanStates.waiting_photo), F.photo)
async def on_photo(
    message: Message,
    state: FSMContext,
    settings: Settings,
) -> None:
    await _process_photo_message(message, state, settings)


@router.message(StateFilter(ScanStates.waiting_photo), F.document)
async def on_document(
    message: Message,
    state: FSMContext,
    settings: Settings,
) -> None:
    doc = message.document
    if doc is None or doc.mime_type is None or not doc.mime_type.startswith("image/"):
        await message.answer("Пришли изображение (фото или файл с типом image/…).")
        return
    await _process_photo_message(message, state, settings)


@router.message(StateFilter(ScanStates.waiting_photo))
async def waiting_but_wrong(message: Message, settings: Settings) -> None:
    if not _chat_ok(settings, message):
        return
    if not _topic_ok(settings, message):
        return
    await message.answer("Нужно отправить фото или картинку файлом. /cancel — отмена.")
