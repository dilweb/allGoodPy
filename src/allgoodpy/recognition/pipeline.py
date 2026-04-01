import logging
import re
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pyzbar.pyzbar import decode as zbar_decode

ORDER_PATTERN = re.compile(r"\b(\d{8,10}-\d{4}-\d{1})\b")

logger = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    barcodes: list[str] = field(default_factory=list)
    order_numbers: list[str] = field(default_factory=list)


def _np_from_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Не удалось прочитать изображение")
    return img


def _variants_bgr(bgr: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = [bgr]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )
    out.append(cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    out.append(cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR))
    return out


def _maybe_scale(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = min(h, w)
    if m >= 900:
        return bgr
    scale = 900 / m
    return cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _decode_barcodes(bgr: np.ndarray) -> set[str]:
    found: set[str] = set()
    for variant in _variants_bgr(bgr):
        for angle in (0, 90, 180, 270):
            img = variant if angle == 0 else _rotate(variant, angle)
            for sym in zbar_decode(img):
                raw = sym.data
                try:
                    found.add(raw.decode("utf-8"))
                except UnicodeDecodeError:
                    found.add(raw.decode("latin-1", errors="replace"))
    return found


def _rotate(bgr: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return bgr
    h, w = bgr.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        bgr, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def _ocr_pil_once(pil: Image.Image, cfg: str, lang: str | None) -> str:
    if lang is None:
        return pytesseract.image_to_string(pil, config=cfg) or ""
    return pytesseract.image_to_string(pil, lang=lang, config=cfg) or ""


def _ocr_texts(bgr: np.ndarray) -> str:
    chunks: list[str] = []
    cfg = "--oem 3 --psm 6"
    lang_chain: tuple[str | None, ...] = ("rus+eng", "eng", None)
    for variant in _variants_bgr(_maybe_scale(bgr))[:3]:
        rgb = cv2.cvtColor(variant, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        piece = ""
        for lang in lang_chain:
            try:
                piece = _ocr_pil_once(pil, cfg, lang)
                break
            except pytesseract.TesseractError as e:
                logger.debug("tesseract lang=%r failed: %s", lang, e)
                continue
        chunks.append(piece)
    return "\n".join(chunks)


def _extract_orders(text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for m in ORDER_PATTERN.finditer(text):
        v = m.group(1)
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered


def recognize_image_bytes(data: bytes) -> RecognitionResult:
    bgr = _np_from_bytes(data)
    barcodes = sorted(_decode_barcodes(bgr))
    try:
        text = _ocr_texts(bgr)
    except Exception as e:
        logger.warning("OCR skipped after error: %s", e)
        text = ""
    orders = _extract_orders(text)
    return RecognitionResult(barcodes=barcodes, order_numbers=orders)
