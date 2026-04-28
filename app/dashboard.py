"""
Dashboard widgets for the Application Layer.

Implements three UI modules required by the Final Generative AI Project PDF:
  1. Balance Wheel    -- radar chart Hard vs Soft skills (by category)
  2. RPG Tech Tree    -- video-game style locked/unlocked skills map
  3. Semester Wrapped -- shareable PNG card (LinkedIn-ready)

All renderers return a matplotlib Figure (or PNG bytes) and are pure
functions of the user profile + ML prediction output.
"""

from __future__ import annotations

import io
from collections import Counter
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Skill taxonomy -- maps every skill to one or more categories
# ---------------------------------------------------------------------------

_CATEGORIES: dict[str, set[str]] = {
    "Backend": {
        "python", "java", "go", "golang", "node", "node.js", "nodejs",
        "fastapi", "django", "flask", "spring", "spring boot", "express",
        "rest api", "graphql", "microservices", "system design",
        "rabbitmq", "kafka", "grpc",
    },
    "Frontend": {
        "javascript", "typescript", "react", "vue", "angular", "next.js",
        "nextjs", "html", "css", "tailwind", "figma", "redux", "svelte",
    },
    "DevOps": {
        "docker", "kubernetes", "k8s", "linux", "bash", "terraform",
        "ansible", "ci/cd", "github actions", "gitlab ci", "jenkins",
        "aws", "gcp", "azure", "helm", "prometheus", "grafana", "nginx",
    },
    "Data/ML": {
        "pandas", "numpy", "scikit-learn", "sklearn", "pytorch",
        "tensorflow", "keras", "matplotlib", "jupyter", "spark",
        "machine learning", "ml", "statistics", "math", "mlops",
        "huggingface", "openai", "llm", "nlp", "computer vision",
    },
    "Database": {
        "sql", "postgresql", "postgres", "mysql", "mongodb", "redis",
        "sqlite", "elasticsearch", "clickhouse", "cassandra", "dynamodb",
    },
    "Tools": {
        "git", "github", "gitlab", "jira", "confluence", "notion",
        "vscode", "intellij", "postman", "swagger",
    },
}

_SOFT_KEYWORDS: set[str] = {
    "communication", "teamwork", "leadership", "problem solving",
    "ownership", "persistence", "creativity", "collaboration",
    "adaptability", "time management", "critical thinking", "empathy",
    "коммуникация", "лидерство", "командная работа",
}


def _bucket_skills(skills: list[str]) -> dict[str, int]:
    """Return {category: count_in_category}."""
    counts = {c: 0 for c in _CATEGORIES}
    for raw in skills:
        s = raw.strip().lower()
        for cat, vocab in _CATEGORIES.items():
            if s in vocab:
                counts[cat] += 1
                break
    return counts


# ---------------------------------------------------------------------------
# 1. Balance Wheel -- radar chart Hard vs Soft
# ---------------------------------------------------------------------------

def fig_balance_wheel(profile: dict, *, max_per_axis: int = 8):
    """Radar chart with one axis per skill category + an axis for soft skills.

    The polygon area visually shows whether the user is balanced or
    specialized. Filled polygon == hard skill coverage by category, with
    a separate Soft-skills axis included so the wheel is "Hard vs Soft".
    """
    hard = profile.get("hard_skills", []) or []
    soft = profile.get("soft_skills", []) or []

    hard_counts = _bucket_skills(hard)
    soft_count  = len(soft)

    axes_labels = list(hard_counts.keys()) + ["Soft"]
    raw_values  = list(hard_counts.values()) + [soft_count]

    # Normalize each axis to 0..10 scale (cap at max_per_axis)
    values = [min(v, max_per_axis) / max_per_axis * 10 for v in raw_values]

    n = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    vals = values + values[:1]
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("none"); ax.set_facecolor("none")

    ax.plot(angles_closed, vals, color="#4299e1", linewidth=2)
    ax.fill(angles_closed, vals, color="#4299e1", alpha=0.25)

    # styling
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_yticks([2.5, 5, 7.5, 10])
    ax.set_yticklabels(["2.5", "5", "7.5", "10"], color="#a0aec0", fontsize=8)
    ax.set_ylim(0, 10)
    ax.set_xticks(angles)
    composite = [f"{lab}  ·  {raw}" for lab, raw in zip(axes_labels, raw_values)]
    ax.set_xticklabels(composite, color="#e2e8f0", fontsize=10)
    ax.tick_params(axis="x", pad=14)
    ax.grid(color="#4a5568", alpha=0.4)

    ax.set_title("Balance Wheel — Hard vs Soft Skills",
                 color="#e2e8f0", fontsize=13, pad=22)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. RPG Tech Tree -- locked/unlocked skill nodes for a target role
# ---------------------------------------------------------------------------

def fig_tech_tree(target_role: str,
                  matched_skills: list[str],
                  missing_skills: list[str]):
    """Video-game style skill tree.

    Center node = target role (boss).
    Surrounding nodes = required skills, color-coded:
      - green/unlocked = user already has it
      - red/locked     = user is missing it
    Each ring outward is "1 level deeper" so unlocked are inner ring,
    locked are outer ring (the road still to walk).
    """
    matched = list(matched_skills or [])
    missing = list(missing_skills or [])

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("none"); ax.set_facecolor("none")
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal"); ax.axis("off")

    # central boss node
    boss = mpatches.FancyBboxPatch(
        (-0.32, -0.10), 0.64, 0.20,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=2, edgecolor="#f6ad55", facecolor="#7b341e",
    )
    ax.add_patch(boss)
    ax.text(0, 0, target_role.upper(), ha="center", va="center",
            color="#fefcbf", fontsize=12, fontweight="bold")
    ax.text(0, -0.18, "FINAL BOSS", ha="center", va="center",
            color="#f6ad55", fontsize=8, fontstyle="italic")

    def _ring(skills, radius, color_face, color_edge, color_text, label):
        n = len(skills)
        if n == 0:
            return
        for i, s in enumerate(skills):
            ang = np.pi / 2 - 2 * np.pi * i / n  # top -> clockwise
            x, y = radius * np.cos(ang), radius * np.sin(ang)
            node = mpatches.FancyBboxPatch(
                (x - 0.16, y - 0.05), 0.32, 0.10,
                boxstyle="round,pad=0.01,rounding_size=0.04",
                linewidth=1.5, edgecolor=color_edge, facecolor=color_face,
            )
            ax.add_patch(node)
            ax.text(x, y, s, ha="center", va="center",
                    color=color_text, fontsize=8, fontweight="bold")
            # connector line to center
            ax.plot([0, x * 0.62], [0, y * 0.62],
                    color=color_edge, linewidth=0.8, alpha=0.5,
                    linestyle="--" if label == "locked" else "-")

    _ring(matched, radius=0.55,
          color_face="#22543d", color_edge="#68d391",
          color_text="#c6f6d5", label="unlocked")
    _ring(missing, radius=1.05,
          color_face="#742a2a", color_edge="#fc8181",
          color_text="#fed7d7", label="locked")

    # legend
    legend_handles = [
        mpatches.Patch(facecolor="#22543d", edgecolor="#68d391",
                       label=f"UNLOCKED ({len(matched)})"),
        mpatches.Patch(facecolor="#742a2a", edgecolor="#fc8181",
                       label=f"LOCKED ({len(missing)})"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              ncol=2, frameon=False, fontsize=9,
              labelcolor="#e2e8f0",
              bbox_to_anchor=(0.5, -0.06))

    # progress %
    total = len(matched) + len(missing)
    pct = int(100 * len(matched) / total) if total else 0
    ax.set_title(f"RPG Tech Tree — Path to {target_role}   ·   "
                 f"Progress: {pct}%",
                 color="#e2e8f0", fontsize=13, pad=12)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Semester Wrapped -- shareable PNG (Spotify-style card)
# ---------------------------------------------------------------------------

def fig_semester_wrapped(profile: dict,
                          predictions: list[dict],
                          *,
                          highlights: dict | None = None):
    """Generate a square shareable card summarizing the semester.

    `predictions` is the output of predict_top_roles (list of dicts with
    keys: role, score_pct, matched_skills, missing_skills).

    `highlights` may provide custom strings:
        top_skill, top_role, focus_area, total_skills, ...

    Returns a matplotlib Figure (1080x1080-ish) ready to st.pyplot or
    convert to PNG bytes via fig_to_png_bytes().
    """
    hard = profile.get("hard_skills", []) or []
    soft = profile.get("soft_skills", []) or []
    name = (profile.get("name") or "Digital Twin").strip()

    highlights = highlights or {}

    # ---- derive headline numbers ----
    top_role     = highlights.get("top_role")     or (predictions[0]["role"] if predictions else "—")
    top_score    = highlights.get("top_score")    or (predictions[0]["score_pct"] if predictions else 0)
    top_skill    = highlights.get("top_skill")    or (hard[0] if hard else "—")
    focus_area   = highlights.get("focus_area")   or _detect_focus(hard)
    total_skills = highlights.get("total_skills") or len(hard) + len(soft)
    next_skill   = highlights.get("next_skill")
    if not next_skill and predictions and predictions[0].get("missing_skills"):
        next_skill = predictions[0]["missing_skills"][0]
    next_skill = next_skill or "—"

    # ---- canvas ----
    fig = plt.figure(figsize=(8, 8), facecolor="#0d1117")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # gradient background blobs
    for r, c, alpha in [(0.55, "#553c9a", 0.55),
                         (0.40, "#2c5282", 0.55),
                         (0.30, "#742a2a", 0.30)]:
        circ = mpatches.Circle((0.85, 0.95), r, color=c, alpha=alpha,
                               transform=ax.transAxes)
        ax.add_patch(circ)
    for r, c, alpha in [(0.50, "#22543d", 0.50),
                         (0.30, "#1a365d", 0.55)]:
        circ = mpatches.Circle((0.05, 0.05), r, color=c, alpha=alpha,
                               transform=ax.transAxes)
        ax.add_patch(circ)

    # header
    year = datetime.now().year
    ax.text(0.06, 0.93, "SEMESTER WRAPPED", color="#fbd38d",
            fontsize=15, fontweight="bold")
    ax.text(0.06, 0.89, f"{name}  ·  {year}",
            color="#e2e8f0", fontsize=11, alpha=0.85)

    # card rows
    rows = [
        ("TOP ROLE MATCH",   top_role,           f"{top_score}% fit"),
        ("FOCUS AREA",       focus_area,         f"{len(hard)} hard skills"),
        ("MOST USED SKILL",  top_skill,          "from your CV"),
        ("NEXT TO LEARN",    next_skill,         "biggest gap to close"),
        ("TOTAL ARSENAL",    f"{total_skills} skills",
                              f"{len(hard)} hard / {len(soft)} soft"),
    ]

    y0, dy = 0.78, 0.13
    for i, (label, value, sub) in enumerate(rows):
        y = y0 - i * dy
        ax.text(0.06, y,        label, color="#fbd38d",
                fontsize=10, fontweight="bold", alpha=0.85)
        ax.text(0.06, y - 0.04, str(value)[:34], color="#ffffff",
                fontsize=18, fontweight="bold")
        ax.text(0.06, y - 0.075, sub, color="#a0aec0", fontsize=9,
                fontstyle="italic")

    # footer
    ax.text(0.5, 0.04,
            "Digital Twin Career Engine  ·  Generated by GenAI",
            color="#718096", fontsize=9, ha="center",
            fontstyle="italic", alpha=0.85)

    return fig


def _detect_focus(hard_skills: list[str]) -> str:
    counts = _bucket_skills(hard_skills)
    if not counts or max(counts.values()) == 0:
        return "Generalist"
    return max(counts, key=counts.get)


def fig_to_png_bytes(fig, dpi: int = 150) -> bytes:
    """Convert any matplotlib Figure to PNG bytes (for st.download_button)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi,
                bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()
