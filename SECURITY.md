# Security Posture & Threat Model

This document describes how the Legal Contract Analyzer defends against common
web and AI application attacks. Controls are centralized in
[`src/security.py`](src/security.py) and covered by
[`tests/test_security.py`](tests/test_security.py).

## Trust boundaries

| Input | Trust | Handling |
|---|---|---|
| Uploaded PDF/DOCX file | **Untrusted** | Validated (size, extension, magic bytes) before parsing |
| Text extracted from documents | **Untrusted** | Normalized, control-char stripped, length-capped; HTML-escaped on render |
| Search query | **Untrusted** | Sanitized + length-capped; used only as vectorizer input |
| Educational content (flashcards/articles) | Trusted (in-repo) | Still escaped on render (defense-in-depth) |
| ML model file | Trusted (in-repo, committed) | Loaded from local disk only |

## OWASP Web Top 10 (2021 & 2025) — mapping

| Risk | Mitigation in this app |
|---|---|
| **A01 Broken Access Control / IDOR / BOLA** | Single-user local tool with no authenticated multi-tenant data. Contract IDs are not authorization tokens — there are no per-user records to cross-access. For a shared public deployment, treat the library as a shared demo store (documented below). |
| **A02 Cryptographic Failures** | No secrets, passwords, or PII credentials are collected or stored. No secrets in the repo (`.streamlit/secrets.toml` is gitignored). |
| **A03 Injection (SQL/HTML/XSS)** | **SQL:** every query in `src/storage/db.py` is parameterized (`?` placeholders); no string-built SQL. **XSS/HTML injection:** all document-derived text is passed through `security.safe()` (HTML-escape) before any `unsafe_allow_html` render; raw HTML from documents is never emitted. Input is Unicode-normalized and control-char stripped. |
| **A04 Insecure Design** | Business logic isolated in `src/`; the model returns only a label + probability (no code path executes document content). Hard input caps by design. |
| **A05 Security Misconfiguration** | Streamlit XSRF protection enabled; usage stats disabled; upload size capped to 15 MB; no debug endpoints exposed. |
| **A06 Vulnerable Components** | Lean dependency set, pinned scikit-learn; no `torch`/transformer stack. `pip audit` friendly. |
| **A07 Identification/Auth Failures** | No authentication system to weaken; no session tokens or passwords handled. |
| **A08 Software/Data Integrity** | Model artifact is committed and loaded from local disk; no dynamic code download or `pickle` from untrusted sources (only the app's own trained model). |
| **A09 Logging/Monitoring** | Errors are caught and shown as sanitized messages; no sensitive data logged. |
| **A10 SSRF** | The app makes **no outbound network requests** at runtime; documents are parsed locally. (CUAD download happens only in the offline training script.) |

## OWASP LLM / AI Top 10 — mapping

| Risk | Status |
|---|---|
| **LLM01 Prompt Injection** | **Not applicable** — there is no LLM in the inference path. Classification is scikit-learn TF-IDF; extraction and risk are rule-based. Document text is never used as a model *instruction*. |
| **LLM02 Insecure Output Handling** | Model output is a bounded class label + float, rendered as data (escaped), never as executable content. |
| **LLM03 Training Data Poisoning** | The model is trained offline on the public CUAD dataset; end-user input never retrains the deployed model. |
| **LLM04 Model DoS** | Input length and upload size are hard-capped; inference is O(ms) with no unbounded loops. |
| **LLM05 Supply Chain** | No third-party model APIs; the only model is the app's own committed artifact. |
| **LLM06 Sensitive Info Disclosure** | Runs locally; uploaded contracts are not sent to any external service and persist only if the user saves them. |
| **LLM08 Excessive Agency** | The system has no tools/actions — it reads and reports. It cannot send email, execute code, or take actions. |

## File upload defense (`validate_upload`)

Before any parser runs, an upload must pass:
1. **Extension allowlist** — only `.pdf` / `.docx`.
2. **Non-empty + size cap** — ≤ 15 MB (blocks decompression/DoS attempts).
3. **Magic-byte check** — the real bytes must match the extension (`%PDF` for PDF, `PK\x03\x04` for DOCX), so a renamed `.exe`/`.html`/script cannot masquerade as a document.

Malformed-but-valid documents are additionally handled by `try/except` around parsing, so a crafted file yields a graceful error, not a crash.

## Known limitations / notes

- **Shared library on public deploy:** the SQLite library is process-wide. On a
  single-user local run this is expected; on a shared public deployment, saved
  contracts are visible to all visitors. For multi-user isolation, add
  authentication and per-user scoping in `src/storage/db.py` (the only module
  that would change).
- **No OCR:** scanned image PDFs are detected and flagged, not processed.
- This tool provides informational analysis only and is **not legal advice**.

## Reporting

Found an issue? Open a private security advisory on the repository rather than a
public issue.
