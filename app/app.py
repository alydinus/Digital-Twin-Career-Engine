"""
Application Layer -- Streamlit UI
Digital Twin Career Engine v2

Tabs:
  1. CV Upload   -- PDF + text + LinkedIn + NotebookLM, auto-profile
  2. Predict     -- ML prediction + Balance Wheel + RPG Tech Tree
  3. CV Roast    -- AI critique of your CV
  4. Telegram    -- job search + infographics (connected to CV)
  5. Roadmap     -- GenAI career plan
  6. Interview   -- interview preparation
  7. Live Coach  -- chatbot with "Roast My Stack" toggle
  8. About       -- architecture

Run: streamlit run app/app.py
"""

import sys, json, os
from pathlib import Path
from collections import Counter

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
except ImportError:
    pass

sys.path.insert(0, str(ROOT))

from model.predictor        import load_profile, load_jobs, predict_top_roles, normalize_skills
from model.semantic_matcher import SKLEARN_AVAILABLE
from agent.resource_finder  import find_resources
from agent.career_coach     import generate_roadmap
from agent.interview_coach  import generate_interview_questions
from agent.live_coach       import coach_reply
from utils.resume_parser    import parse_resume
from utils.linkedin_parser  import parse_linkedin_pdf, parse_linkedin_text
from utils.notebooklm_bridge import (
    extract_from_notebook, is_mcp_available,
    get_notebook_url, MCP_SETUP_INSTRUCTIONS,
)
from app.dashboard          import (
    fig_balance_wheel,
    fig_tech_tree,
    fig_semester_wrapped,
    fig_to_png_bytes,
)

# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Digital Twin Career Engine",
    page_icon="\U0001f3af",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.skill-tag     { display:inline-block; background:#ebf8ff; border:1px solid #90cdf4;
                 border-radius:12px; padding:2px 10px; margin:2px; font-size:.82em; color:#2b6cb0; }
.skill-missing { background:#fff5f5; border-color:#fc8181; color:#c53030; }
.skill-match   { background:#f0fff4; border-color:#68d391; color:#276749; }
.phase-card  { border-left:4px solid #4299e1; padding:10px 14px; margin:6px 0;
               background:rgba(66,153,225,0.10); border-radius:0 8px 8px 0; }
.roast-issue { border-left:4px solid #fc8181; padding:10px 14px; margin:8px 0;
               background:rgba(252,129,129,0.12); border-radius:0 8px 8px 0; }
.roast-issue b { color:#fc8181; }
.roast-praise{ border-left:4px solid #68d391; padding:10px 14px; margin:8px 0;
               background:rgba(104,211,145,0.12); border-radius:0 8px 8px 0; }
.roast-praise b { color:#68d391; }
.q-card      { border-left:4px solid #9f7aea; padding:8px 12px; margin:6px 0;
               background:rgba(159,122,234,0.10); border-radius:0 8px 8px 0; }
.cv-banner   { border-left:4px solid #4299e1; border-radius:0 8px 8px 0;
               background:rgba(66,153,225,0.10); padding:10px 16px; margin:8px 0; }

/* ── Metric boxes: transparent, theme-aware ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
}
[data-testid="metric-container"] label {
    color: rgba(180,200,220,0.85) !important;
    font-size: 0.82em !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="stMetricValue"] {
    color: inherit !important;
    font-size: 1.9em !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tags(skills, cls="skill-tag"):
    return " ".join(f'<span class="{cls}">{s}</span>' for s in skills) if skills else "<i>&#8212;</i>"


def get_active_profile(sidebar_profile: dict) -> dict:
    cv = st.session_state.get("cv_profile")
    if cv and (cv.get("hard_skills") or cv.get("soft_skills")):
        return cv
    return sidebar_profile


def _enrich_profile(profile: dict) -> dict:
    hard = profile.get("hard_skills", [])
    soft = profile.get("soft_skills", [])
    profile.setdefault("_hard_skills_norm", normalize_skills(hard))
    profile.setdefault("_soft_skills_norm", normalize_skills(soft))
    profile.setdefault("_all_skills_norm",
                        normalize_skills(hard) | normalize_skills(soft))
    return profile


# ---------------------------------------------------------------------------
# CV Roast logic
# ---------------------------------------------------------------------------

_MARKET_DEMAND = {
    "python": 90, "docker": 85, "postgresql": 80, "git": 95,
    "linux": 75, "aws": 70, "kubernetes": 60, "fastapi": 55,
    "django": 50, "redis": 65, "ci/cd": 70, "github actions": 55,
    "rest api": 80, "sql": 85, "javascript": 75, "typescript": 60,
    "react": 65, "go": 45, "java": 55, "terraform": 40,
}


def _roast_rules(profile: dict, harshness: int) -> dict:
    hard      = [s.lower() for s in profile.get("hard_skills", [])]
    hard_orig = profile.get("hard_skills", [])
    soft      = profile.get("soft_skills", [])
    interests = profile.get("interests", [])
    name      = profile.get("name", "").strip()
    issues, praise, tips = [], [], []

    if not name or name in ("--", "User", ""):
        issues.append(("Нет имени", "Анонимные резюме отклоняют первыми. Укажи полное имя."))

    n_hard = len(hard_orig)
    if n_hard == 0:
        issues.append(("Ноль hard skills", "Нет навыков = нет резюме. Добавь всё, с чем работал."))
    elif n_hard < 5:
        issues.append(("Слишком мало навыков",
                        f"Только {n_hard} hard skill(ов). Рекрутеры ждут минимум 6-8. "
                        "Добавь фреймворки, инструменты, версии — всё, что использовал."))
    elif n_hard >= 15:
        praise.append(("Широкий стек", f"{n_hard} hard skills — отличное покрытие."))
    else:
        praise.append(("Нормальный стек", f"{n_hard} hard skills — хорошая отправная точка."))

    langs = {"python","java","go","golang","javascript","typescript",
             "rust","c++","c#","kotlin","swift","scala","php","ruby"}
    found_langs = [s for s in hard if s in langs]
    if not found_langs:
        issues.append(("Нет языка программирования",
                        "Рекрутер смотрит на язык первым делом. Укажи основной язык явно."))
    elif len(found_langs) == 1:
        praise.append(("Основной язык", f"Указан {found_langs[0].title()} — хорошо."))
    else:
        praise.append(("Несколько языков", f"{', '.join(s.title() for s in found_langs[:3])}."))

    dbs = {"postgresql","mysql","mongodb","redis","sqlite","sql",
           "cassandra","elasticsearch","clickhouse","dynamodb"}
    found_dbs = [s for s in hard if s in dbs]
    if not found_dbs:
        issues.append(("Нет баз данных",
                        "Почти каждая вакансия требует SQL или NoSQL. Добавь хотя бы PostgreSQL или MongoDB."))
        tips.append("Учи PostgreSQL: https://www.postgresqltutorial.com/")
    else:
        praise.append(("БД в стеке", f"{', '.join(s.upper() for s in found_dbs[:3])}."))

    devops = {"docker","kubernetes","k8s","terraform","ansible","helm"}
    found_devops = [s for s in hard if s in devops]
    if not found_devops:
        issues.append(("Нет контейнеризации",
                        "Docker — базовый навык для backend/devops в 2024. "
                        "Без него выпадаешь из 70%+ вакансий."))
        tips.append("Docker quickstart: https://docs.docker.com/get-started/")
    else:
        praise.append(("Контейнеры", f"Есть {', '.join(s.title() for s in found_devops[:3])}."))

    clouds = {"aws","gcp","azure","digitalocean","cloudflare"}
    found_cloud = [s for s in hard if s in clouds]
    if not found_cloud:
        issues.append(("Нет облачных платформ",
                        "AWS/GCP/Azure встречается в 60%+ вакансий. Даже базовый AWS — большой плюс."))
        tips.append("AWS Free Tier: https://aws.amazon.com/free/")
    else:
        praise.append(("Облако", f"{', '.join(s.upper() for s in found_cloud)}."))

    cicd = {"ci/cd","github actions","gitlab ci","jenkins","circleci"}
    if harshness >= 5 and not any(s in hard for s in cicd):
        issues.append(("Нет CI/CD",
                        "Автоматизация деплоя — стандарт. GitHub Actions можно освоить за день."))

    market_missing = []
    for skill, demand in sorted(_MARKET_DEMAND.items(), key=lambda x: -x[1]):
        if skill not in hard and demand >= 70:
            market_missing.append(f"{skill.title()} ({demand}%)")
        if len(market_missing) >= 3:
            break
    if market_missing:
        issues.append(("Высокоспросные навыки отсутствуют",
                        "Пропущены навыки с высоким рыночным спросом: " + ", ".join(market_missing) +
                        ". Приоритизируй их изучение."))

    frontend_s = {"react","vue","angular","html","css","javascript","typescript"}
    backend_s  = {"python","java","go","fastapi","django","flask","spring"}
    data_s     = {"pandas","numpy","scikit-learn","pytorch","tensorflow","spark"}
    has_back  = any(s in hard for s in backend_s)
    has_front = any(s in hard for s in frontend_s)
    has_data  = any(s in hard for s in data_s)

    if has_back and not has_front and not has_data:
        praise.append(("Чёткая backend-специализация", "Специализация видна — это плюс."))
    elif has_front and has_back:
        praise.append(("Fullstack стек", "Есть и Frontend, и Backend."))
    elif has_data:
        praise.append(("Data/ML стек", "Присутствуют навыки data science."))

    if not soft:
        issues.append(("Нет soft skills",
                        "HR и менеджеры смотрят на soft skills. Добавь teamwork, коммуникацию, ownership."))
    elif len(soft) >= 3:
        praise.append(("Soft skills есть", f"{len(soft)} указано — достаточно."))

    if not interests and harshness >= 4:
        issues.append(("Нет интересов",
                        "Интересы показывают мотивацию и помогают попасть в культуру компании."))

    n = len(issues)
    if n == 0:
        verdict, text, score = "Топовый профиль", "Почти нечего критиковать — сильная заявка.", 9
    elif n <= 2:
        verdict, text, score = "Хороший профиль с пробелами", "В целом solid, но пару вещей нужно поправить перед отправкой.", 7
    elif n <= 4:
        verdict, text, score = "Средний профиль — нужна работа", "Рекрутеры заметят эти пробелы. Исправь топ-2 проблемы для резкого роста.", 5
    elif n <= 6:
        verdict, text, score = "Слабый профиль — нужна переработка", "Много красных флагов. Авто-скрининг отфильтрует.", 3
    else:
        verdict, text, score = "Критическое состояние", "Серьёзно переработай перед отправкой — отклонят на первом этапе.", 1

    if harshness >= 8 and score < 7:
        text += f" (Жёсткость {harshness}/10 — без прикрас.)"

    return {
        "issues": issues, "praise": praise, "tips": tips,
        "verdict": verdict, "roast_text": text,
        "roast_score": score, "source": "rule-based",
    }


def generate_cv_roast(profile: dict, harshness: int = 5) -> dict:
    if os.environ.get("LLM_API_KEY", "").strip():
        try:
            from utils.llm_client import call_llm_json
            prompt = (
                f"Ты жёсткий карьерный ментор (жёсткость {harshness}/10, 10=без пощады).\n"
                "Дай глубокий критический разбор IT-профиля. Проанализируй:\n"
                "- Полноту и актуальность стека (рынок 2024)\n"
                "- Баланс навыков (backend/frontend/devops/data)\n"
                "- Соответствие рыночному спросу\n"
                "- Что конкретно добавить или убрать\n\n"
                "ВАЖНО: отвечай ТОЛЬКО на русском языке.\n\n"
                + json.dumps({k:v for k,v in profile.items() if not k.startswith("_")},
                             ensure_ascii=False, indent=2)
                + "\n\nВерни JSON:\n"
                + '{\n  "verdict": "краткий вердикт одной фразой",\n  "roast_text": "честное резюме в 2-3 предложениях",\n  "roast_score": <1-10, где 10=отлично>,\n  "issues": [["название проблемы", "объяснение + что делать"], ...],\n  "praise": [["название сильной стороны", "почему это хорошо"], ...],\n  "tips": ["конкретная рекомендация 1", "рекомендация 2"]\n}\nТолько JSON. Будь конкретным и полезным.'
            )
            data = call_llm_json(prompt, max_tokens=1500)
            data["source"] = "llm"
            return data
        except Exception:
            pass
    return _roast_rules(profile, harshness)


# ---------------------------------------------------------------------------
# Infographic helpers
# ---------------------------------------------------------------------------

def _fig_match_bars(jobs):
    n = min(len(jobs), 15)
    fig, ax = plt.subplots(figsize=(8, max(3, n * 0.48)))
    fig.patch.set_facecolor("none"); ax.set_facecolor("none")
    names  = [f"{j.role_name[:28]}  ({j.channel_name})" for j in jobs[:n]]
    scores = [j.match_pct for j in jobs[:n]]
    colors = ["#68d391" if s >= 70 else "#f6ad55" if s >= 40 else "#fc8181" for s in scores]
    bars = ax.barh(names, scores, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Match %", fontsize=10, color="#a0aec0")
    ax.set_title("Job relevance to your profile", fontsize=12, color="#e2e8f0", pad=10)
    ax.tick_params(axis="y", labelsize=8, colors="#a0aec0")
    ax.tick_params(axis="x", labelsize=9, colors="#a0aec0")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4a5568"); ax.spines["bottom"].set_color("#4a5568")
    ax.axvline(70, color="#68d391", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(40, color="#f6ad55", linestyle="--", linewidth=0.8, alpha=0.6)
    for bar, s in zip(bars, scores):
        ax.text(s + 1, bar.get_y() + bar.get_height() / 2,
                f"{s}%", va="center", fontsize=8, color="#a0aec0")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#68d391", label=">=70% Strong"),
        Patch(facecolor="#f6ad55", label="40-69% Partial"),
        Patch(facecolor="#fc8181", label="<40% Weak"),
    ], loc="lower right", fontsize=8, framealpha=0.3)
    plt.tight_layout()
    return fig


def _fig_skills_gap(jobs, top_n=12):
    cnt = Counter(s.lower() for j in jobs for s in j.you_need)
    if not cnt:
        return None
    skills, counts = zip(*cnt.most_common(top_n))
    colors = ["#fc8181" if c >= 5 else "#f6ad55" if c >= 3 else "#90cdf4" for c in counts]
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("none"); ax.set_facecolor("none")
    ax.bar(skills, counts, color=colors, edgecolor="white")
    ax.set_title("Skills you are missing (by job count)", fontsize=12, color="#e2e8f0", pad=10)
    ax.set_ylabel("Jobs requiring this", fontsize=10, color="#a0aec0")
    ax.tick_params(axis="x", rotation=38, labelsize=8, colors="#a0aec0")
    ax.tick_params(axis="y", labelsize=9, colors="#a0aec0")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4a5568"); ax.spines["bottom"].set_color("#4a5568")
    for i, (_, c) in enumerate(zip(skills, counts)):
        ax.text(i, c + 0.05, str(c), ha="center", fontsize=8, color="#a0aec0")
    plt.tight_layout()
    return fig


def _fig_seniority_pie(jobs):
    cnt = Counter(j.seniority for j in jobs)
    colors_map = {"senior":"#fc8181","middle":"#f6ad55","junior":"#68d391"}
    labels, vals = zip(*cnt.most_common())
    colors = [colors_map.get(l, "#90cdf4") for l in labels]
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor("none"); ax.set_facecolor("none")
    wedges, texts, autotexts = ax.pie(vals, labels=labels, colors=colors,
                                       autopct="%1.0f%%", startangle=90)
    for t in texts + autotexts:
        t.set_color("#e2e8f0")
    ax.set_title("Seniority breakdown", fontsize=11, color="#e2e8f0", pad=10)
    plt.tight_layout()
    return fig


def _fig_channels(jobs):
    cnt = Counter(j.channel_name for j in jobs)
    if len(cnt) < 2:
        return None
    channels, nums = zip(*cnt.most_common())
    fig, ax = plt.subplots(figsize=(5, max(2.5, len(channels) * 0.45)))
    fig.patch.set_facecolor("none"); ax.set_facecolor("none")
    ax.barh(channels, nums, color="#4299e1", edgecolor="white")
    ax.set_title("Jobs per channel", fontsize=11, color="#e2e8f0", pad=10)
    ax.tick_params(axis="y", labelsize=9, colors="#a0aec0")
    ax.tick_params(axis="x", labelsize=9, colors="#a0aec0")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4a5568"); ax.spines["bottom"].set_color("#4a5568")
    for i, n in enumerate(nums):
        ax.text(n + 0.05, i, str(n), va="center", fontsize=9, color="#a0aec0")
    plt.tight_layout()
    return fig


def _fig_match_histogram(jobs):
    scores = [j.match_pct for j in jobs]
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("none"); ax.set_facecolor("none")
    ax.hist(scores, bins=10, range=(0, 100), color="#4299e1", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(scores), color="#fc8181", linestyle="--",
               linewidth=1.5, label=f"Avg: {int(np.mean(scores))}%")
    ax.set_xlabel("Match %", fontsize=10, color="#a0aec0")
    ax.set_ylabel("Jobs", fontsize=10, color="#a0aec0")
    ax.set_title("Match score distribution", fontsize=11, color="#e2e8f0", pad=10)
    ax.legend(fontsize=9)
    ax.tick_params(colors="#a0aec0")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4a5568"); ax.spines["bottom"].set_color("#4a5568")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def sidebar_profile():
    st.sidebar.header("Profile")

    if "cv_profile" in st.session_state:
        cv = st.session_state["cv_profile"]
        st.sidebar.success(
            f"CV loaded: **{cv.get('name','--')}**\n"
            f"{len(cv.get('hard_skills',[]))} hard / "
            f"{len(cv.get('soft_skills',[]))} soft skills"
        )
        if st.sidebar.button("Clear CV", use_container_width=True):
            del st.session_state["cv_profile"]
            st.rerun()
        st.sidebar.divider()

    with open(ROOT / "data" / "profile.json", encoding="utf-8") as f:
        default = json.load(f)

    name      = st.sidebar.text_input("Name", value=default.get("name", ""))
    hard_raw  = st.sidebar.text_area("Hard Skills (comma-separated)",
                    value=", ".join(default.get("hard_skills", [])), height=80)
    soft_raw  = st.sidebar.text_area("Soft Skills (comma-separated)",
                    value=", ".join(default.get("soft_skills", [])), height=60)
    inter_raw = st.sidebar.text_area("Interests (comma-separated)",
                    value=", ".join(default.get("interests", [])), height=50)

    st.sidebar.divider()
    llm_key = st.sidebar.text_input("LLM API Key (optional)", type="password",
                                     placeholder="sk-ant-... / sk-...")
    if llm_key:
        os.environ["LLM_API_KEY"] = llm_key
        st.sidebar.success("LLM activated")

    parse = lambda raw: [s.strip() for s in raw.split(",") if s.strip()]
    ns = normalize_skills(parse(hard_raw)) | normalize_skills(parse(soft_raw))
    return {
        "name": name, "hard_skills": parse(hard_raw),
        "soft_skills": parse(soft_raw), "interests": parse(inter_raw),
        "_hard_skills_norm": normalize_skills(parse(hard_raw)),
        "_soft_skills_norm": normalize_skills(parse(soft_raw)),
        "_all_skills_norm":  ns,
    }


# ---------------------------------------------------------------------------
# Tab -- Predict & Dashboard
# (Balance Wheel + Top-3 ML prediction + RPG Tech Tree + Semester Wrapped)
# ---------------------------------------------------------------------------

def tab_predict_dashboard(profile: dict):
    st.subheader("Predict & Dashboard")
    st.caption("ML prediction · Balance Wheel · RPG Tech Tree · Semester Wrapped")

    if not profile.get("hard_skills") and not profile.get("soft_skills"):
        st.warning("Upload your resume in CV Upload or fill the sidebar profile first.")
        return

    # ---- run ML prediction ----
    jobs = load_jobs(ROOT / "data" / "jobs.csv")
    predictions = predict_top_roles(profile, jobs, top_n=len(jobs))
    top3 = predictions[:3]

    # ---- KPI strip ----
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Hard Skills", len(profile.get("hard_skills", [])))
    k2.metric("Soft Skills", len(profile.get("soft_skills", [])))
    k3.metric("Top match",   f"{top3[0]['score_pct']}%" if top3 else "--")
    k4.metric("Roles ≥40%",  sum(1 for r in predictions if r["score_pct"] >= 40))

    st.divider()

    # ---- Balance Wheel + Top-3 list ----
    st.markdown("### ⚖️ Balance Wheel — Hard vs Soft")
    col_wheel, col_top = st.columns([3, 2])
    with col_wheel:
        fig_w = fig_balance_wheel(profile)
        st.pyplot(fig_w, use_container_width=True); plt.close(fig_w)
    with col_top:
        st.markdown("**Top-3 ML predictions**")
        for i, r in enumerate(top3, 1):
            medal = ["🥇", "🥈", "🥉"][i - 1]
            st.markdown(
                f"<div class='phase-card'>{medal} <b>{r['role']}</b> · {r['score_pct']}%<br>"
                f"<small>have {len(r['matched_skills'])} · need {len(r['missing_skills'])}</small></div>",
                unsafe_allow_html=True)

        # JSON output (PDF requires JSON output from Model layer)
        with st.expander("Raw JSON output (Model Layer)"):
            st.json({
                "top_3_roles": [
                    {
                        "role":           r["role"],
                        "score":          r["score_pct"] / 100,
                        "matched_skills": r["matched_skills"],
                        "missing_skills": r["missing_skills"],
                    } for r in top3
                ]
            })

    st.divider()

    # ---- RPG Tech Tree ----
    st.markdown("### 🎮 RPG Tech Tree — path to your dream job")
    role_names = [r["role"] for r in predictions]
    target_role = st.selectbox("Choose target role:", role_names,
                               index=0, key="tree_role")
    target_data = next(r for r in predictions if r["role"] == target_role)

    fig_t = fig_tech_tree(
        target_role,
        target_data["matched_skills"],
        target_data["missing_skills"],
    )
    st.pyplot(fig_t, use_container_width=True); plt.close(fig_t)

    if target_data["missing_skills"]:
        st.markdown("**Locked skills — unlock with these resources:**")
        sk = st.selectbox("Skill:", target_data["missing_skills"], key="tree_skill")
        res = find_resources(sk)
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.markdown("**Docs**")
            for u in res.get("docs", []): st.markdown(f"- [{u[:40]}…]({u})")
        with rc2:
            st.markdown("**Courses**")
            for u in res.get("courses", []): st.markdown(f"- [{u[:40]}…]({u})")
        with rc3:
            st.markdown("**GitHub**")
            for u in res.get("github", []): st.markdown(f"- [{u[:40]}…]({u})")

    st.divider()

    # ---- Semester Wrapped export ----
    st.markdown("### 🎁 Semester Wrapped — share your year")
    st.caption("Generates a LinkedIn-ready PNG card summarizing your skills journey")

    col_btn, col_dl = st.columns([1, 1])
    with col_btn:
        if st.button("Generate Semester Wrapped", type="primary",
                     use_container_width=True):
            fig_s = fig_semester_wrapped(profile, predictions)
            png   = fig_to_png_bytes(fig_s, dpi=180)
            st.session_state["wrapped_png"] = png
            plt.close(fig_s)

    if "wrapped_png" in st.session_state:
        with col_dl:
            st.download_button(
                "Download PNG",
                data=st.session_state["wrapped_png"],
                file_name="semester_wrapped.png",
                mime="image/png",
                use_container_width=True,
            )
        st.image(st.session_state["wrapped_png"], use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 1 -- CV Upload
# ---------------------------------------------------------------------------

def tab_cv_upload():
    st.subheader("CV Upload")
    st.caption("Upload PDF or paste text -- skills will flow into Telegram Jobs and Roadmap automatically")

    has_llm = bool(os.environ.get("LLM_API_KEY", ""))

    from utils.pdf_parser import get_available_backend
    backend = get_available_backend()

    col_b, col_l = st.columns(2)
    with col_b:
        if "not installed" not in backend and "не установлена" not in backend:
            st.success(f"PDF backend: `{backend}`")
        else:
            st.warning("pdfplumber not installed  -->  `pip install pdfplumber`")
    with col_l:
        if has_llm:
            st.success("LLM mode active")
        else:
            st.info("Regex fallback (add LLM_API_KEY for AI parsing)")

    if "cv_profile" in st.session_state:
        cv = st.session_state["cv_profile"]
        st.markdown(
            f'<div class="cv-banner">CV loaded: <b>{cv.get("name","--")}</b> ',
            unsafe_allow_html=True
        )
        st.markdown(
            f'&nbsp;&middot; {len(cv.get("hard_skills",[]))} hard ',
            unsafe_allow_html=True
        )
        st.markdown(
            f'&nbsp;&middot; {len(cv.get("soft_skills",[]))} soft skills</div>',
            unsafe_allow_html=True
        )

    tab_pdf, tab_txt, tab_li, tab_nb = st.tabs(
        ["📄 PDF file", "📝 Text resume", "🔗 LinkedIn Import", "📓 NotebookLM"]
    )

    with tab_pdf:
        uploaded = st.file_uploader("Choose PDF:", type=["pdf"])
        if uploaded:
            st.info(f"`{uploaded.name}` -- {uploaded.size // 1024} KB")
            if st.button("Parse PDF", type="primary"):
                with st.spinner("Reading PDF..."):
                    try:
                        from utils.pdf_parser import parse_pdf_resume
                        parsed = parse_pdf_resume(uploaded.read(), use_llm=has_llm)
                        _store_cv(parsed)
                    except RuntimeError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Error: {e}")

    with tab_txt:
        sample = ("Ivan Petrov -- Senior Backend Developer\n"
                  "Stack: Python, FastAPI, PostgreSQL, Redis, Docker, Kubernetes, AWS, Kafka\n"
                  "Also used: Terraform, GitHub Actions, Grafana, Prometheus\n"
                  "Soft skills: problem solving, teamwork, leadership\n"
                  "Interests: backend, cloud, distributed systems")
        text = st.text_area("Paste resume text:", value=sample, height=200)
        if st.button("Parse text", type="primary"):
            with st.spinner("Analyzing..."):
                parsed = parse_resume(text, use_llm=has_llm)
                _store_cv(parsed)

    with tab_li:
        st.markdown("### LinkedIn Profile Import")
        st.caption(
            "Export your LinkedIn profile as PDF "
            "(Settings → Data Privacy → Get a copy) or paste the text."
        )
        li_pdf_tab, li_txt_tab = st.tabs(["LinkedIn PDF", "LinkedIn Text"])
        with li_pdf_tab:
            li_file = st.file_uploader("LinkedIn PDF export:", type=["pdf"],
                                        key="li_pdf")
            if li_file:
                st.info(f"`{li_file.name}` -- {li_file.size // 1024} KB")
                if st.button("Parse LinkedIn PDF", type="primary", key="li_pdf_btn"):
                    with st.spinner("Parsing LinkedIn PDF..."):
                        try:
                            parsed = parse_linkedin_pdf(li_file.read(), use_llm=has_llm)
                            _store_cv(parsed)
                        except Exception as e:
                            st.error(f"Error: {e}")
        with li_txt_tab:
            li_sample = (
                "Алыкулов Алыдин\n"
                "Backend Developer at Company\n\n"
                "Experience\n"
                "Company — Backend Developer — 2023-Present\n"
                "Java, Spring Boot, PostgreSQL, Docker, Kubernetes\n\n"
                "Skills\n"
                "Java, Go, SQL, Docker, Linux, Git, CI/CD, REST API"
            )
            li_text = st.text_area("Paste LinkedIn profile text:",
                                    value=li_sample, height=200,
                                    key="li_text")
            if st.button("Parse LinkedIn text", type="primary", key="li_txt_btn"):
                with st.spinner("Analyzing LinkedIn profile..."):
                    parsed = parse_linkedin_text(li_text, use_llm=has_llm)
                    _store_cv(parsed)

    with tab_nb:
        st.markdown("### NotebookLM Integration (Platform Layer)")
        mcp_ok = is_mcp_available()
        nb_url = get_notebook_url()
        if mcp_ok:
            st.success("MCP Server connected ✓")
        else:
            st.info("MCP Server not detected — use manual text input below")

        nb_url_input = st.text_input(
            "NotebookLM Notebook URL:",
            value=nb_url,
            placeholder="https://notebooklm.google.com/notebook/...",
            key="nb_url",
        )

        if nb_url_input and mcp_ok:
            if st.button("Extract via MCP", type="primary", key="nb_mcp_btn"):
                with st.spinner("Querying NotebookLM via MCP..."):
                    result = extract_from_notebook(notebook_url=nb_url_input)
                    if result.get("hard_skills"):
                        _store_cv(result)
                    else:
                        st.warning("No skills extracted. Check notebook link.")

        st.divider()
        st.markdown("**Manual: paste NotebookLM summary text**")
        nb_text = st.text_area(
            "NotebookLM output text:",
            height=180,
            placeholder=(
                "Paste the summary or audio overview text from NotebookLM.\n"
                "Example: 'Based on the uploaded CV, the student has skills in...'"
            ),
            key="nb_text",
        )
        if st.button("Parse NotebookLM text", type="primary", key="nb_txt_btn"):
            if nb_text and nb_text.strip():
                with st.spinner("Extracting skills from NotebookLM..."):
                    result = extract_from_notebook(text=nb_text)
                    if result.get("hard_skills"):
                        _store_cv(result)
                    else:
                        st.warning("No skills found in the text.")
            else:
                st.warning("Paste text from NotebookLM first.")

        with st.expander("Setup instructions (MCP)", expanded=False):
            st.markdown(MCP_SETUP_INSTRUCTIONS)

    if "cv_profile" in st.session_state:
        _render_cv_card(st.session_state["cv_profile"])


def _store_cv(parsed: dict):
    parsed = _enrich_profile(parsed)
    st.session_state["cv_profile"] = parsed
    src   = parsed.get("_meta", {}).get("source", "")
    chars = parsed.get("_meta", {}).get("pdf_chars", 0)
    msg   = f"Done -- {'LLM' if 'llm' in src else 'Regex'}"
    if chars:
        msg += f" -- {chars} chars"
    st.success(msg)


def _render_cv_card(profile: dict):
    st.divider()
    st.markdown("### 📋 Извлечённый профиль")

    hard      = profile.get("hard_skills", [])
    soft      = profile.get("soft_skills", [])
    interests = profile.get("interests",   [])

    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Hard Skills", len(hard))
    with m2: st.metric("Soft Skills", len(soft))
    with m3: st.metric("Интересы",    len(interests))

    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        name = profile.get("name", "")
        if name:
            st.markdown(f"**Имя:** {name}")
        st.markdown("**Hard Skills:**")
        if hard:
            st.markdown(tags(hard), unsafe_allow_html=True)
        else:
            st.caption("не найдено")
    with c2:
        st.markdown("**Soft Skills:**")
        if soft:
            st.markdown(tags(soft), unsafe_allow_html=True)
        else:
            st.caption("не найдено")
        st.markdown("**Интересы:**")
        if interests:
            st.markdown(tags(interests), unsafe_allow_html=True)
        else:
            st.caption("не найдено")

    st.divider()
    col_save, col_hint = st.columns(2)
    with col_save:
        if st.button("Save as profile.json", use_container_width=True):
            display = {k: v for k, v in profile.items() if not k.startswith("_")}
            with open(ROOT / "data" / "profile.json", "w", encoding="utf-8") as f:
                json.dump(display, f, indent=2, ensure_ascii=False)
            st.success("Saved!")
    with col_hint:
        st.info("Go to CV Roast or Telegram Jobs next")


# ---------------------------------------------------------------------------
# Tab 2 -- CV Roast
# ---------------------------------------------------------------------------

def tab_cv_roast(profile):
    st.subheader("CV Roast")
    st.caption("Honest critique of your profile -- what is blocking your next offer")

    has_llm = bool(os.environ.get("LLM_API_KEY", ""))

    if "cv_profile" in st.session_state:
        st.markdown('<div class="cv-banner">Analyzing profile from your CV</div>',
                    unsafe_allow_html=True)

    if not profile.get("hard_skills") and not profile.get("soft_skills"):
        st.warning("Upload your resume in CV Upload tab or fill the sidebar profile first.")
        return

    col_h, col_m = st.columns([2, 1])
    with col_h:
        harshness = st.slider("Harshness level:", 1, 10, 6,
                              help="1=gentle, 10=merciless")
    with col_m:
        st.caption(f"Mode: {'LLM' if has_llm else 'Rule-based'}")
        st.write("")
        run = st.button("Run Roast", type="primary", use_container_width=True)

    if run:
        with st.spinner("Analyzing..."):
            result = generate_cv_roast(profile, harshness)
            st.session_state["roast_result"] = result

    if "roast_result" not in st.session_state:
        return

    r = st.session_state["roast_result"]
    score = r.get("roast_score", 5)
    color = "#68d391" if score >= 7 else "#f6ad55" if score >= 4 else "#fc8181"

    col_v, col_s = st.columns([3, 1])
    with col_v:
        st.markdown(f"## {r.get('verdict', '--')}")
        st.markdown(f"*{r.get('roast_text', '')}*")
    with col_s:
        st.metric("CV Score", f"{score}/10")

    st.progress(score / 10)
    st.caption(f"Source: {'LLM' if r.get('source')=='llm' else 'Rule-based'}")
    st.divider()

    col_i, col_p = st.columns(2)
    with col_i:
        if r.get("issues"):
            st.markdown(f"### Issues ({len(r['issues'])})")
            for title, desc in r["issues"]:
                st.markdown(
                    f'<div class="roast-issue"><b>{title}</b><br><small>{desc}</small></div>',
                    unsafe_allow_html=True)
    with col_p:
        if r.get("praise"):
            st.markdown(f"### Strengths ({len(r['praise'])})")
            for title, desc in r["praise"]:
                st.markdown(
                    f'<div class="roast-praise"><b>{title}</b><br><small>{desc}</small></div>',
                    unsafe_allow_html=True)

    if r.get("tips"):
        st.divider()
        st.markdown("### Action steps")
        for i, tip in enumerate(r["tips"], 1):
            st.markdown(f"**{i}.** {tip}")


# ---------------------------------------------------------------------------
# Tab 3 -- Telegram Jobs
# ---------------------------------------------------------------------------

def tab_telegram_jobs(profile):
    st.subheader("Telegram Job Search")
    st.caption("IT vacancies from Telegram channels matched to your profile with full infographics")

    if "cv_profile" in st.session_state:
        cv = st.session_state["cv_profile"]
        st.markdown(
            f'<div class="cv-banner">Profile from CV: <b>{cv.get("name","--")}</b> ',
            unsafe_allow_html=True)
        n = len(cv.get("hard_skills", []))
        st.markdown(f'&nbsp;&middot; {n} hard skills</div>', unsafe_allow_html=True)

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=True)
    except ImportError:
        pass

    _id    = os.environ.get("TELEGRAM_API_ID",   "").strip()
    _hash  = os.environ.get("TELEGRAM_API_HASH",  "").strip()
    _phone = os.environ.get("TELEGRAM_PHONE",     "").strip()
    _sess  = ROOT / "data" / ".telegram_session.session"

    has_creds   = bool(_id and _hash and _phone)
    has_session = _sess.exists()
    has_tg      = has_creds and has_session

    if has_tg:
        st.success("Telegram authorized -- real search active")
    elif has_creds:
        st.warning("Keys found, no session yet --> run: `python telegram_bot/auth.py`")
    else:
        st.info("Demo mode (mock data). Configure .env for real search.")

    col1, col2, col3 = st.columns(3)
    with col1:
        days_back = st.slider("Search depth (days):", 1, 30, 7)
    with col2:
        min_match = st.slider("Min match (%):", 0, 80, 20)
    with col3:
        max_cards = st.slider("Cards to show:", 3, 20, 8)

    with st.expander("Channels and settings", expanded=False):
        with open(ROOT / "data" / "telegram_channels.json", encoding="utf-8") as ff:
            ch_cfg = json.load(ff)
        ch_opts = {c["name"]: c["username"] for c in ch_cfg["channels"]}
        default_ch = [
            n for n, u in ch_opts.items()
            if u in ("dev_kg","getmatch","python_jobs_ru","devops_jobs_ru","backend_jobs_ru")
        ][:5]
        sel_names = st.multiselect("Channels:", list(ch_opts.keys()),
                                   default=default_ch or list(ch_opts.keys())[:5])
        sel_ch = [ch_opts[n] for n in sel_names]

        st.markdown("**Add channel:**")
        cu_col, btn_col = st.columns([4, 1])
        with cu_col:
            custom_u = st.text_input("Username (no @):", key="tg_cu", placeholder="dev_kg")
        with btn_col:
            st.write("")
            if st.button("+", key="tg_add"):
                if custom_u.strip():
                    from telegram_bot.job_scraper import add_custom_channel
                    ok = add_custom_channel(custom_u.strip().lstrip("@"))
                    if ok:
                        st.success("Added!")
                    else:
                        st.info("Already in list")
                    st.rerun()

    st.divider()

    if st.button("Find vacancies", type="primary", use_container_width=True):
        with st.spinner("Searching..."):
            try:
                from telegram_bot.job_scraper import scrape_jobs
                all_found = scrape_jobs(
                    profile       = profile,
                    channels      = sel_ch if sel_ch else None,
                    days_back     = days_back,
                    min_match_pct = 0,
                    use_mock      = not has_tg,
                )
                st.session_state["tg_jobs"] = all_found
            except Exception as e:
                st.error(f"Error: {e}")
                return

    if "tg_jobs" not in st.session_state:
        st.info("Click Find vacancies to search")
        return

    all_jobs = st.session_state["tg_jobs"]
    jobs     = [j for j in all_jobs if j.match_pct >= min_match]

    if not jobs:
        st.warning(f"No vacancies with match >= {min_match}%. Lower the threshold.")
        return

    # KPI
    st.markdown("## Analytics")
    avg     = int(sum(j.match_pct for j in jobs) / len(jobs))
    best    = max(j.match_pct for j in jobs)
    seniors = sum(1 for j in jobs if j.seniority == "senior")
    juniors = sum(1 for j in jobs if j.seniority == "junior")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Found",       len(jobs))
    k2.metric("Avg match",   f"{avg}%")
    k3.metric("Best match",  f"{best}%")
    k4.metric("Senior",      seniors)
    k5.metric("Junior",      juniors)

    st.markdown("---")

    r1c1, r1c2 = st.columns([3, 1])
    with r1c1:
        st.markdown("**Match scores by vacancy**")
        fig = _fig_match_bars(jobs)
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    with r1c2:
        st.markdown("**Seniority**")
        fig2 = _fig_seniority_pie(jobs)
        st.pyplot(fig2, use_container_width=True); plt.close(fig2)

    st.markdown("---")

    r2c1, r2c2 = st.columns([3, 2])
    with r2c1:
        st.markdown("**Skills you are missing (across all jobs)**")
        fig3 = _fig_skills_gap(jobs)
        if fig3:
            st.pyplot(fig3, use_container_width=True); plt.close(fig3)
        else:
            st.success("You cover all skills in found vacancies!")
    with r2c2:
        st.markdown("**Match score distribution**")
        fig4 = _fig_match_histogram(jobs)
        st.pyplot(fig4, use_container_width=True); plt.close(fig4)

    fig5 = _fig_channels(jobs)
    if fig5:
        st.markdown("---")
        st.markdown("**Jobs per channel**")
        st.pyplot(fig5, use_container_width=True); plt.close(fig5)

    st.markdown(f"## Vacancies  ({min(max_cards, len(jobs))} of {len(jobs)})")

    seniority_emoji = {"senior": "R", "middle": "Y", "junior": "G"}
    for j in jobs[:max_cards]:
        emoji = {"senior": "🔴", "middle": "🟡", "junior": "🟢"}.get(j.seniority, "🟡")
        with st.expander(
            f"{emoji} **{j.role_name}** -- {j.match_pct}% | {j.channel_name} · {j.date.strftime('%d.%m')}",
            expanded=False,
        ):
            st.progress(j.match_pct / 100)
            ca, cb = st.columns(2)
            with ca:
                st.markdown("**Have:**")
                st.markdown(tags(j.you_have, "skill-match"), unsafe_allow_html=True)
            with cb:
                st.markdown("**Need:**")
                st.markdown(tags(j.you_need, "skill-missing"), unsafe_allow_html=True)

            with st.expander("Job text"):
                st.text(j.text[:600] + ("..." if len(j.text) > 600 else ""))

            st.markdown(f"[Open in Telegram]({j.url})")

            if j.you_need:
                sk = st.selectbox("Resources for skill:", j.you_need, key=f"r_{j.id}")
                res = find_resources(sk)
                rr1, rr2, rr3 = st.columns(3)
                with rr1:
                    for u in res.get("docs",    []): st.markdown(f"[Docs]({u})")
                with rr2:
                    for u in res.get("courses", []): st.markdown(f"[Course]({u})")
                with rr3:
                    for u in res.get("github",  []): st.markdown(f"[GitHub]({u})")


# ---------------------------------------------------------------------------
# Tab 4 -- Roadmap
# ---------------------------------------------------------------------------

def tab_roadmap(profile):
    st.subheader("Career Roadmap")
    if "cv_profile" in st.session_state:
        st.markdown('<div class="cv-banner">Using profile from your CV</div>',
                    unsafe_allow_html=True)

    jobs       = load_jobs(ROOT / "data" / "jobs.csv")
    role_names = [j["role"] for j in jobs]
    target     = st.selectbox("Target role:", role_names)
    results    = predict_top_roles(profile, jobs, top_n=len(jobs))
    role_data  = next((r for r in results if r["role"] == target), None)
    if not role_data:
        return

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Match",  f"{role_data['score_pct']}%")
    with c2: st.metric("Have",   len(role_data["matched_skills"]))
    with c3: st.metric("Need",   len(role_data["missing_skills"]))

    st.divider()
    if st.button("Generate Roadmap", type="primary"):
        with st.spinner("Building plan..."):
            rm = generate_roadmap(
                profile=profile, target_role=target,
                missing_skills=role_data["missing_skills"],
                matched_skills=role_data["matched_skills"],
                score_pct=role_data["score_pct"],
            )
            st.session_state["roadmap"] = rm

    if "roadmap" not in st.session_state:
        return
    rm = st.session_state["roadmap"]
    st.caption(f"Source: {'LLM' if rm.get('source')=='llm' else 'Template'}")
    lvl = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(rm.get("readiness_level", "medium"), "🟡")
    st.info(f"{lvl} {rm.get('readiness', '')}")
    if rm.get("total_months", 0) > 0:
        st.markdown(f"**Time:** {rm['total_months']} months ({rm['total_weeks']} weeks)")
    if rm.get("quick_wins"):
        st.markdown("**Quick wins:**")
        st.markdown(tags(rm["quick_wins"]), unsafe_allow_html=True)
    if rm.get("phases"):
        st.markdown("### Phases")
        for p in rm["phases"]:
            desc = f'<br><small>{p.get("description","")}</small>' if p.get("description") else ""
            st.markdown(
                f'<div class="phase-card"><b>{p["title"]}</b> &middot; {p["weeks"]} weeks',
                unsafe_allow_html=True)
            st.markdown(tags(p["skills"]), unsafe_allow_html=True)
            if desc:
                st.markdown(desc, unsafe_allow_html=True)
    if rm.get("tips"):
        st.markdown("### Tips")
        for t in rm["tips"]:
            st.markdown(f"- {t}")
    if rm.get("strengths"):
        st.markdown("### Strengths")
        st.markdown(tags(rm["strengths"], "skill-match"), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 5 -- Interview Coach
# ---------------------------------------------------------------------------

def tab_interview(profile):
    st.subheader("Interview Coach")
    st.caption("Questions focused on your skill gaps to prepare for interviews")
    if "cv_profile" in st.session_state:
        st.markdown('<div class="cv-banner">Using profile from your CV</div>',
                    unsafe_allow_html=True)

    jobs       = load_jobs(ROOT / "data" / "jobs.csv")
    role_names = [j["role"] for j in jobs]
    target     = st.selectbox("Target role:", role_names, key="interview_role")
    results    = predict_top_roles(profile, jobs, top_n=len(jobs))
    role_data  = next((r for r in results if r["role"] == target), None)
    if not role_data:
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Gaps (questions on these):**")
        st.markdown(tags(role_data["missing_skills"], "skill-missing"), unsafe_allow_html=True)
    with col2:
        st.markdown("**Strengths:**")
        st.markdown(tags(role_data["matched_skills"], "skill-match"), unsafe_allow_html=True)

    n_per = st.slider("Questions per skill:", 1, 4, 2)
    st.divider()

    if st.button("Generate questions", type="primary"):
        with st.spinner("Generating..."):
            qs = generate_interview_questions(
                target_role=target,
                missing_skills=role_data["missing_skills"],
                matched_skills=role_data["matched_skills"],
                n_per_skill=n_per,
            )
            st.session_state["interview_qs"] = qs

    if "interview_qs" not in st.session_state:
        return
    qs = st.session_state["interview_qs"]
    st.caption(f"Source: {'LLM' if qs.get('source')=='llm' else 'Static bank'}")

    if qs.get("missing"):
        st.markdown("### Gap questions")
        for q in qs["missing"]:
            st.markdown(
                f'<div class="q-card"><b>[{q["skill"]}]</b> {q["question"]}',
                unsafe_allow_html=True)
            st.markdown(f'<div class="q-card"><small>Hint: {q["hint"]}</small></div>',
                        unsafe_allow_html=True)

    if qs.get("strengths"):
        st.markdown("### Strength questions")
        for q in qs["strengths"]:
            st.markdown(
                f'<div class="q-card" style="border-color:#68d391"><b>[{q["skill"]}]</b> {q["question"]}',
                unsafe_allow_html=True)
            st.markdown(f'<div class="q-card" style="border-color:#68d391"><small>Hint: {q["hint"]}</small></div>',
                        unsafe_allow_html=True)

    if qs.get("behavioral"):
        st.markdown("### Behavioral questions (STAR)")
        for q in qs["behavioral"]:
            hint_html = f"<br><small>Hint: {q['hint']}</small>" if q.get("hint") else ""
            st.markdown(
                f'<div class="q-card" style="border-color:#f6ad55">{q["question"]}{hint_html}</div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 6 -- About
# ---------------------------------------------------------------------------

def tab_about():
    st.subheader("System Architecture")
    st.code("""
 PDF / Text CV         Telegram channels        Manual profile
      |                       |                       |
      v                       v                       v
 +------------------------------------------------------------+
 | Platform Layer                                             |
 | pdf_parser  resume_parser  job_scraper (Telethon)          |
 +------------------------+-----------------------------------+
                          v
 +------------------------------------------------------------+
 | Model Layer                                                |
 | predictor.py (overlap)   semantic_matcher (TF-IDF)         |
 +------------------------+-----------------------------------+
                          v
 +------------------------------------------------------------+
 | Agent Layer  (LLM + fallback)                              |
 | career_coach  interview_coach  resource_finder             |
 | cv_roaster (rule-based + LLM)                              |
 +------------------------+-----------------------------------+
                          v
 +------------------------------------------------------------+
 | Application Layer -- Streamlit 6 tabs                      |
 | CV Upload -> CV Roast -> Telegram Jobs                     |
 |          -> Roadmap   -> Interview                         |
 +------------------------------------------------------------+
""", language="text")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**GenAI / ML:**
- `pdf_parser` - pdfplumber + LLM
- `resume_parser` - LLM / regex (150+ skills)
- `career_coach` - LLM roadmap
- `interview_coach` - LLM questions
- `cv_roaster` - LLM + rule-based critique
- `resource_finder` - LLM + static dict
- `semantic_matcher` - TF-IDF cosine
- `job_scraper` - Telethon parser
""")
    with col2:
        st.markdown("""
**Fallback (нет API ключа):**
- Резюме -> словарь 150+ навыков
- Roadmap -> шаблон фаз
- Интервью -> банк 100+ вопросов
- CV Roast -> правила рынка
- Ресурсы -> статический словарь
- Telegram -> 11 mock-вакансий

**Стек:** Python 3.11 - Streamlit - Pandas
Matplotlib - scikit-learn - Telethon - Docker
""")
    st.divider()
    st.markdown("""
**Быстрый старт:**
```
pip install -r requirements.txt
streamlit run app/app.py
```
**Авторизация Telegram (один раз):**
```
python telegram_bot/auth.py
```
**Docker:**
```
docker compose up -d
```
""")


# ---------------------------------------------------------------------------
# Tab 7 -- Live Coach Chatbot
# ---------------------------------------------------------------------------

def tab_live_coach(profile):
    st.subheader("Live Coach")
    st.caption(
        "Chat with your Digital Twin — mentor mode or Roast My Stack mode"
    )

    # ---- persona toggle ----
    col_toggle, col_status = st.columns([1, 2])
    with col_toggle:
        roast_mode = st.toggle("🔥 Roast My Stack", value=False, key="roast_toggle")
    persona = "roast" if roast_mode else "mentor"
    with col_status:
        if roast_mode:
            st.error("ROAST MODE — aggressive tech lead persona active")
        else:
            st.info("Mentor mode — supportive career advisor")

    has_llm = bool(os.environ.get("LLM_API_KEY", ""))
    st.caption(f"Engine: {'LLM' if has_llm else 'Rule-based'}")

    # ---- ML predictions for context ----
    predictions = None
    try:
        jobs = load_jobs(ROOT / "data" / "jobs.csv")
        predictions = predict_top_roles(profile, jobs, top_n=3)
    except Exception:
        pass

    # ---- chat history ----
    if "coach_messages" not in st.session_state:
        st.session_state["coach_messages"] = []

    # Display existing messages
    for msg in st.session_state["coach_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---- user input ----
    user_input = st.chat_input(
        "Roast my stack!" if roast_mode else "Ask your Digital Twin..."
    )

    if user_input:
        # Append user message
        st.session_state["coach_messages"].append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate reply
        with st.chat_message("assistant"):
            with st.spinner("Thinking..." if not roast_mode else "Preparing roast..."):
                result = coach_reply(
                    messages=st.session_state["coach_messages"],
                    persona=persona,
                    profile=profile,
                    predictions=predictions,
                )
                reply = result["reply"]
                st.markdown(reply)
                st.caption(f"Source: {result['source']}")

        st.session_state["coach_messages"].append(
            {"role": "assistant", "content": reply}
        )

    # ---- clear chat button ----
    if st.session_state["coach_messages"]:
        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state["coach_messages"] = []
            st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sidebar_p = sidebar_profile()
    st.session_state["sidebar_profile"] = sidebar_p
    profile = get_active_profile(sidebar_p)

    t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
        "\U0001f4c4 CV Upload",
        "\U0001f3af Predict",
        "\U0001f525 CV Roast",
        "\U0001f4e8 Telegram Jobs",
        "\U0001f5fa\ufe0f Roadmap",
        "\U0001f3a4 Interview",
        "\U0001f4ac Live Coach",
        "\u2139\ufe0f About",
    ])

    with t1: tab_cv_upload()
    with t2: tab_predict_dashboard(profile)
    with t3: tab_cv_roast(profile)
    with t4: tab_telegram_jobs(profile)
    with t5: tab_roadmap(profile)
    with t6: tab_interview(profile)
    with t7: tab_live_coach(profile)
    with t8: tab_about()


if __name__ == "__main__":
    main()
