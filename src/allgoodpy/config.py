from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    telegram_bot_token: str
    forum_topic_id: int | None = None
    allowed_chat_id: int | None = None

    google_sheets_spreadsheet_id: str
    google_service_account_file: Path
    google_sheet_name: str | None = None
