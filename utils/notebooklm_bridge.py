"""
Platform Layer -- NotebookLM MCP bridge helpers.

Этот модуль не запускает Antigravity сам по себе, но даёт проекту
воспроизводимый конфиг для PDF-требования:
  - какой MCP server установить
  - какой notebook URL передать
  - какой system prompt отдать live-агенту
"""

from __future__ import annotations

import json
import re


NOTEBOOKLM_SERVER_PACKAGE = "notebooklm-mcp-server"

_NOTEBOOK_RE = re.compile(
    r"^https://notebooklm\.google\.com/(?:notebook|share)/[A-Za-z0-9_\-?=&/]+$"
)


def validate_notebooklm_url(url: str) -> bool:
    return bool(url and _NOTEBOOK_RE.match(url.strip()))


def build_antigravity_system_prompt(
    notebook_url: str,
    *,
    target_role: str = "Backend Engineer",
    missing_skill: str = "Kubernetes",
) -> str:
    if not validate_notebooklm_url(notebook_url):
        raise ValueError("NotebookLM URL is invalid")

    return (
        "You are the Digital Twin live agent. "
        "Use the NotebookLM MCP server first, query the notebook below, "
        "extract the user's hard skills, soft skills, projects, and academic signals, "
        "then combine that context with the ML prediction.\n\n"
        f"NotebookLM notebook URL: {notebook_url}\n"
        f"Predicted target role: {target_role}\n"
        f"Highest-priority missing skill: {missing_skill}\n\n"
        "After reading NotebookLM, browse the live web for current resources, "
        "GitHub repositories, events, or hackathons that help the user learn "
        "that missing skill. Return concise, actionable results."
    )


def build_notebooklm_bridge_config(
    notebook_url: str,
    *,
    target_role: str = "Backend Engineer",
    missing_skill: str = "Kubernetes",
) -> dict:
    if not validate_notebooklm_url(notebook_url):
        raise ValueError("NotebookLM URL is invalid")

    return {
        "platform_layer": "NotebookLM MCP Bridge",
        "mcp_server_package": NOTEBOOKLM_SERVER_PACKAGE,
        "antigravity_review_policy": "Always Proceed",
        "notebook_url": notebook_url,
        "system_prompt": build_antigravity_system_prompt(
            notebook_url,
            target_role=target_role,
            missing_skill=missing_skill,
        ),
        "installation_steps": [
            "Open Antigravity -> MCP Servers",
            f"Install {NOTEBOOKLM_SERVER_PACKAGE}",
            "Share your NotebookLM notebook and copy the direct URL",
            "Paste the URL into the agent system prompt/context",
        ],
    }


if __name__ == "__main__":
    sample_url = "https://notebooklm.google.com/notebook/example"
    config = build_notebooklm_bridge_config(sample_url)
    print(json.dumps(config, indent=2, ensure_ascii=False))
