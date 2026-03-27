from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials

_SCOPES = ("https://www.googleapis.com/auth/spreadsheets",)


def _worksheet(
    spreadsheet_id: str,
    service_account_path: Path,
    sheet_name: str | None,
):
    creds = Credentials.from_service_account_file(
        str(service_account_path),
        scopes=_SCOPES,
    )
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(spreadsheet_id)
    if sheet_name:
        return sh.worksheet(sheet_name)
    return sh.sheet1


def append_processing_row_sync(
    *,
    spreadsheet_id: str,
    service_account_path: Path,
    sheet_name: str | None,
    processed_at: datetime,
    user_id: int,
    user_label: str,
    order_numbers: list[str],
    barcodes: list[str],
    message_link: str,
) -> None:
    ws = _worksheet(spreadsheet_id, service_account_path, sheet_name)
    ts = processed_at.astimezone(timezone.utc).isoformat()
    orders_cell = "; ".join(order_numbers)
    barcodes_cell = "; ".join(barcodes)
    ws.append_row(
        [ts, str(user_id), user_label, orders_cell, barcodes_cell, message_link],
        value_input_option="USER_ENTERED",
    )


async def append_processing_row(**kwargs) -> None:
    await asyncio.to_thread(append_processing_row_sync, **kwargs)
