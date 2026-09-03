"""
Tests for the security layer (src/security.py) and injection-resistance of the
storage + rendering paths.

These assert the concrete attacker scenarios the app must survive:
  - XSS/HTML injection via booby-trapped contract text
  - SQL-injection-style strings passed as data
  - spoofed / oversized / empty file uploads
  - control-character and oversized-input handling
"""

import sys
import zipfile
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import security
from src.security import validate_upload, safe, clean_text_input, clean_query, clean_display
from src.storage import db
from src.report import build_html_report
from src.core import analyze_text


# ---------------- HTML escaping / XSS ----------------

def test_safe_escapes_script_tags():
    payload = '<script>alert("xss")</script>'
    out = safe(payload)
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_safe_escapes_event_handler_and_quotes():
    payload = '"><img src=x onerror=alert(1)>'
    out = safe(payload)
    assert "<img" not in out
    assert "onerror=alert(1)" not in out or "&gt;" in out
    assert "&quot;" in out or "&#x27;" in out or "&lt;" in out


def test_clean_display_escapes_and_truncates():
    out = clean_display("<b>" + "A" * 500, max_len=50)
    assert "<b>" not in out
    assert out.endswith("&hellip;")


def test_html_report_neutralizes_malicious_contract_text():
    # A contract whose clause text tries to inject a script.
    malicious = (
        "1. EVIL CLAUSE\n<script>document.location='http://evil'</script> "
        "and <img src=x onerror=alert(1)>. Governed by the laws of Delaware.\n"
    )
    result = analyze_text(malicious, filename="evil.docx")
    html = build_html_report(result)
    # No executable tag should survive into the report.
    assert "<script>" not in html
    assert "onerror=alert(1)>" not in html
    assert "&lt;script&gt;" in html or "&lt;img" in html


# ---------------- input sanitization ----------------

def test_clean_text_input_strips_control_chars():
    dirty = "hello\x00\x07world\x1b[31m"
    out = clean_text_input(dirty)
    assert "\x00" not in out and "\x07" not in out and "\x1b" not in out
    assert "hello" in out and "world" in out


def test_clean_text_input_caps_length():
    out = clean_text_input("A" * 10000, max_chars=100)
    assert len(out) == 100


def test_clean_query_bounds_length():
    out = clean_query("x" * 5000)
    assert len(out) <= security.MAX_QUERY_CHARS


# ---------------- file upload validation ----------------

def _docx_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<xml/>")
    return buf.getvalue()


def test_upload_rejects_wrong_extension():
    check = validate_upload("malware.exe", b"MZ\x90\x00")
    assert not check.ok


def test_upload_rejects_empty_file():
    check = validate_upload("empty.pdf", b"")
    assert not check.ok


def test_upload_rejects_spoofed_pdf():
    # .pdf extension but HTML/script contents — magic bytes don't match.
    check = validate_upload("fake.pdf", b"<html><script>alert(1)</script></html>")
    assert not check.ok
    assert "spoof" in check.reason.lower() or "match" in check.reason.lower()


def test_upload_rejects_oversized():
    big = b"%PDF" + b"0" * (security.MAX_UPLOAD_BYTES + 1)
    check = validate_upload("big.pdf", big)
    assert not check.ok


def test_upload_accepts_valid_pdf_magic():
    check = validate_upload("ok.pdf", b"%PDF-1.7\n...content...")
    assert check.ok
    assert check.extension == ".pdf"


def test_upload_accepts_valid_docx_zip():
    check = validate_upload("ok.docx", _docx_bytes())
    assert check.ok


# ---------------- SQL injection resistance ----------------

def test_sql_injection_string_stored_as_data(tmp_path):
    from dataclasses import dataclass

    @dataclass
    class C:
        section_number: str = "1"
        heading: str = "H"
        clause_type: str = "GOVERNING_LAW"
        confidence: float = 0.9
        method: str = "ml"
        text: str = "body"

    dbfile = tmp_path / "t.db"
    evil_name = "Robert'); DROP TABLE contracts;--.docx"
    cid = db.save_contract(evil_name, [C()], full_text="x", db_path=dbfile)
    # The table must still exist and hold the row (injection treated as data).
    assert db.count_contracts(db_path=dbfile) == 1
    got = db.get_contract(cid, db_path=dbfile)
    assert got is not None
    assert "DROP TABLE" in got["filename"]  # stored verbatim as data, not executed
