"""
Utils — NotebookLM MCP Bridge (Platform Layer)

Connects the Digital Twin to Google NotebookLM via MCP (Model Context
Protocol). This is the Platform Layer requirement from the PDF:

  > Engineer a direct connection between NotebookLM and Antigravity
  > using the NotebookLM MCP Server.

Modes:
  1. MCP mode  — queries notebooklm-mcp-server if running locally
  2. Manual mode — user pastes NotebookLM summary text, we parse it
  3. Fallback — returns instructions on how to set up MCP

Public API:
    extract_from_notebook(notebook_url=None, text=None) -> dict
"""

from __future__ import annotations
import os, json, re, subprocess
from typing import Optional


# ---------------------------------------------------------------------------
# MCP Server query (when notebooklm-mcp-server is running)
# ---------------------------------------------------------------------------

def _query_mcp_server(notebook_url: str) -> Optional[dict]:
    """
    Try to query the notebooklm-mcp-server via stdio/HTTP.
    Returns extracted profile dict or None if server is not available.
    """
    # Check if MCP server process is available
    # The notebooklm-mcp-server typically runs as a local process
    try:
        # Try HTTP endpoint (common MCP server port)
        import urllib.request
        import urllib.error

        mcp_host = os.getenv("NOTEBOOKLM_MCP_HOST", "localhost")
        mcp_port = os.getenv("NOTEBOOKLM_MCP_PORT", "3100")
        url = f"http://{mcp_host}:{mcp_port}/query"

        payload = json.dumps({
            "notebook_url": notebook_url,
            "query": (
                "Extract all technical skills, soft skills, education, "
                "certifications, and career interests from this notebook. "
                "Return as JSON with keys: hard_skills, soft_skills, "
                "interests, education, certifications."
            )
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Text-based extraction (manual paste from NotebookLM)
# ---------------------------------------------------------------------------

def _parse_notebook_text(text: str) -> dict:
    """Extract skills from NotebookLM summary text using LLM or regex."""

    # Try LLM first
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if api_key:
        try:
            from utils.llm_client import call_llm_json
            prompt = (
                "Extract structured data from this NotebookLM summary.\n\n"
                f"TEXT:\n{text[:4000]}\n\n"
                "Return JSON: {\"hard_skills\": [...], \"soft_skills\": [...], "
                "\"interests\": [...], \"education\": [...], "
                "\"certifications\": [...]}\nOnly JSON."
            )
            data = call_llm_json(prompt, max_tokens=1000)
            data["_meta"] = {"source": "notebooklm-llm"}
            return data
        except Exception:
            pass

    # Regex fallback — reuse linkedin_parser skill extraction
    try:
        from utils.linkedin_parser import _extract_skills_regex, _extract_interests
        hard, soft = _extract_skills_regex(text)
        interests = _extract_interests(text)
        return {
            "hard_skills": hard,
            "soft_skills": soft,
            "interests": interests,
            "education": [],
            "certifications": [],
            "_meta": {"source": "notebooklm-regex"},
        }
    except ImportError:
        return {
            "hard_skills": [], "soft_skills": [], "interests": [],
            "education": [], "certifications": [],
            "_meta": {"source": "notebooklm-fallback"},
        }


# ---------------------------------------------------------------------------
# MCP setup instructions
# ---------------------------------------------------------------------------

MCP_SETUP_INSTRUCTIONS = """
## NotebookLM MCP Setup (Platform Layer)

### Шаг 1: Создай NotebookLM Notebook
1. Открой [NotebookLM](https://notebooklm.google.com/)
2. Загрузи свои данные как источники:
   - CV / Resume (PDF)
   - LinkedIn профиль (скопированный текст)
   - Транскрипт оценок
3. Нажми **Share** и скопируй ссылку на notebook

### Шаг 2: Установи MCP Server
В терминале Antigravity:
```
npx -y notebooklm-mcp-server
```

### Шаг 3: Подключи
В Antigravity Settings → MCP Servers → добавь notebooklm-mcp-server.
Вставь ссылку на notebook в system prompt.

### Шаг 4: Настрой .env
```
NOTEBOOKLM_URL=https://notebooklm.google.com/notebook/YOUR_ID
NOTEBOOKLM_MCP_HOST=localhost
NOTEBOOKLM_MCP_PORT=3100
```

### Альтернатива (без MCP)
Скопируй текст из NotebookLM и вставь в поле ниже.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_from_notebook(
    notebook_url: Optional[str] = None,
    text: Optional[str] = None,
) -> dict:
    """
    Extract profile data from NotebookLM.

    Priority:
      1. MCP server query (if notebook_url provided and server running)
      2. Text parsing (if text provided)
      3. Returns empty profile with setup instructions

    Returns:
        dict with hard_skills, soft_skills, interests, etc.
    """
    # Try MCP server first
    if notebook_url:
        result = _query_mcp_server(notebook_url)
        if result:
            result.setdefault("_meta", {})["source"] = "notebooklm-mcp"
            return result

    # Parse text manually
    if text and text.strip():
        return _parse_notebook_text(text)

    # No input — return empty with instructions
    return {
        "hard_skills": [],
        "soft_skills": [],
        "interests": [],
        "education": [],
        "certifications": [],
        "_meta": {
            "source": "notebooklm-none",
            "instructions": MCP_SETUP_INSTRUCTIONS,
        },
    }


def is_mcp_available() -> bool:
    """Check if NotebookLM MCP server is reachable."""
    try:
        import urllib.request
        mcp_host = os.getenv("NOTEBOOKLM_MCP_HOST", "localhost")
        mcp_port = os.getenv("NOTEBOOKLM_MCP_PORT", "3100")
        url = f"http://{mcp_host}:{mcp_port}/health"
        with urllib.request.urlopen(url, timeout=2):
            return True
    except Exception:
        return False


def get_notebook_url() -> str:
    """Get configured NotebookLM URL from environment."""
    return os.getenv("NOTEBOOKLM_URL", "").strip()


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== NotebookLM MCP Bridge ===")
    print(f"MCP available: {is_mcp_available()}")
    print(f"Notebook URL: {get_notebook_url() or '(not configured)'}")

    # Test text parsing
    sample = """
    Based on the uploaded CV and transcript, the student has:
    Technical Skills: Python, Java, SQL, PostgreSQL, Docker, Git, Spring,
    REST API, Linux, CI/CD, RabbitMQ
    Soft Skills: Communication, Teamwork
    Education: AUCA, Computer Science, 2024
    Interests: Backend development, AI, Security
    """
    result = extract_from_notebook(text=sample)
    print(f"\nExtracted skills: {result.get('hard_skills', [])}")
    print(f"Source: {result.get('_meta', {}).get('source', 'unknown')}")
