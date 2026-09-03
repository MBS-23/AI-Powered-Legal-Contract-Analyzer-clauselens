"""
security.py
-----------
Central, reusable security primitives for the app. Keeping them in one place
makes the app's defensive posture auditable and testable.

Threat model (see SECURITY.md for the full write-up):
  - The app renders text extracted from *untrusted* uploaded documents. Any
    place that emits raw HTML must escape that text first, or a booby-trapped
    contract (e.g. a clause containing `<img src=x onerror=alert(1)>`) becomes
    stored XSS. `safe()` / `clean_display()` handle that.
  - Uploaded files are attacker-controlled. `validate_upload()` enforces size,
    extension, and magic-byte checks before the file ever reaches a parser,
    defending against renamed executables, oversized decompression bombs, and
    empty/garbage input.
  - There is deliberately NO LLM in the inference path, so prompt-injection /
    LLM output-handling attacks have no surface here; the ML model only ever
    returns a class label + probability, never executable content.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

# Hard limits — tune in one place.
MAX_UPLOAD_BYTES = 15 * 1024 * 1024          # 15 MB uploaded file
MAX_TEXT_CHARS = 2_000_000                    # cap stored/processed text (~2 MB)
MAX_QUERY_CHARS = 500                         # search box input
ALLOWED_EXTENSIONS = {".pdf", ".docx"}

# Magic bytes: PDF starts "%PDF"; DOCX is a ZIP container ("PK\x03\x04").
_MAGIC = {
    ".pdf": (b"%PDF",),
    ".docx": (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"),
}

# Control characters (except tab/newline/carriage-return) have no place in
# contract text and can be used to smuggle payloads or corrupt output.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class UploadCheck:
    ok: bool
    reason: str = ""
    extension: str = ""


def _extension(filename: str) -> str:
    filename = filename or ""
    dot = filename.rfind(".")
    return filename[dot:].lower() if dot != -1 else ""


def validate_upload(filename: str, data: bytes) -> UploadCheck:
    """Validate an uploaded file by size, extension, and magic bytes.

    Returns an UploadCheck; callers should refuse to parse when ok is False.
    """
    ext = _extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        return UploadCheck(False, f"Unsupported file type '{ext or 'unknown'}'. Only PDF and DOCX are allowed.")
    if not data:
        return UploadCheck(False, "The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        return UploadCheck(False, f"File is too large (limit {mb:.0f} MB).")
    # Magic-byte sniffing: the real bytes must match the claimed extension, so a
    # renamed .exe/.html can't sneak in as a .pdf/.docx.
    head = data[:8]
    if not any(head.startswith(sig) for sig in _MAGIC[ext]):
        return UploadCheck(False, "File contents don't match its extension (possible spoofed file).")
    return UploadCheck(True, extension=ext)


def clean_text_input(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    """Normalize and bound free text before it is processed or stored.

    - Unicode-normalizes (NFKC) to fold look-alike/confusable forms.
    - Strips control characters.
    - Truncates to a hard length cap (DoS / storage bound).
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS_RE.sub("", text)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def safe(value) -> str:
    """HTML-escape a value for safe interpolation into an HTML string.

    Use this for EVERY dynamic value placed inside an `unsafe_allow_html`
    block. Escapes &, <, >, and quotes.
    """
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def clean_display(text: str, max_len: int | None = None) -> str:
    """Escape text and optionally clip it for display in an HTML snippet."""
    s = safe(text)
    if max_len is not None and len(s) > max_len:
        s = s[:max_len] + "&hellip;"
    return s


def clean_query(text: str) -> str:
    """Sanitize a search query: strip control chars, cap length. Not rendered
    as HTML, but bounded to keep the vectorizer input sane."""
    return clean_text_input(text, max_chars=MAX_QUERY_CHARS).strip()
