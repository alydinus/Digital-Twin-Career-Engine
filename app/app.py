"""
Application Layer — Streamlit UI
Digital Twin Career Engine

Вкладки:
  1. Predict    — матчинг навыков, bar chart, ресурсы
  2. Roadmap    — GenAI план развития по фазам
  3. Interview  — вопросы для подготовки к собеседованию
  4. Job Match  — вставь вакансию → анализ соответствия
  5. Resume     — парсинг текста резюме
  6. About      — архитектура проекта

Запуск: streamlit run app/app.py
"""

import sys, json, os
from pathlib import Path
from io import BytesIO

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent

# ── Загрузка переменных из .env ──────────────────────────────────────────────
# override=True: перезаписывает уже существующие переменные окружения,
# чтобы изменения в .env подхватились без перезапуска процесса
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
except ImportError:
    pass  # python-dotenv не установлен — переменные берутся из окружения
sys.path.insert(0, str(ROOT))

from model.predictor        import load_profile, load_jobs, predict_top_roles, normalize_skills
from model.semantic_matcher import semantic_match_score, hybrid_score, SKLEARN_AVAILABLE
from agent.resource_finder  import find_resources
from agent.career_coach     import generate_roadmap
from agent.interview_coach  import generate_interview_questions
from utils.resume_parser    import parse_resume
from utils.job_parser       import parse_job_description, gap_analysis

# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Digital Twin Career Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.skill-tag     { display:inline-block; background:#ebf8ff; border:1px solid #90cdf4;
                 border-radius:12px; padding:2px 10px; margin:2px; font-size:.82em; color:#2b6cb0; }
.skill-missing { background:#fff5f5; border-color:#fc8181; color:#c53030; }
.skill-match   { background:#f0fff4; border-color:#68d391; color:#276749; }
.phase-card    { border-left:4px solid #4299e1; padding:8px 12px; margin:6px 0;
                 background:#f7fafc; border-radius:0 8px 8px 0; }
.metric-box    { text-align:center; padding:12px; background:#f7fafc;
                 border-radius:10px; border:1px solid #e2e8f0; }
.q-card        { border-left:4px solid #9f7aea; padding:8px 12px; margin:6px 0;
                 background:#faf5ff; border-radius:0 8px 8px 0; }
.fit-strong    { color:#276749; font-weight:bold; }
.fit-good      { color:#744210; font-weight:bold; }
.fit-partial   { color:#c05621; font-weight:bold; }
.fit-weak      { color:#c53030; font-weight:bold; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tags(skills, cls="skill-tag"):
    return " ".join(f'<span class="{cls}">{s}</span>' for s in skills) if skills else "<i>—</i>"

def export_md(profile, top_roles):
    lines = [f"# Career Report — {profile.get('name','User')}", "",
             "## Профиль",
             f"**Hard skills:** {', '.join(profile.get('hard_skills',[]))}",
             f"**Soft skills:** {', '.join(profile.get('soft_skills',[]))}", "",
             "## Топ профессий"]
    for i, r in enumerate(top_roles, 1):
        lines += [f"### {i}. {r['role']} — {r['score_pct']}%",
                  f"**Есть:** {', '.join(r['matched_skills']) or '—'}",
                  f"**Нет:**  {', '.join(r['missing_skills']) or '—'}", ""]
    return "\n".join(lines)

def radar_chart(results, score_key="score_pct"):
    """Строит radar chart для всех ролей."""
    roles  = [r["role"].replace(" Engineer","").replace(" Scientist","") for r in results]
    scores = [r.get(score_key, r["score_pct"]) / 100 for r in results]
    N      = len(roles)
    if N < 3:
        return None

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    scores += scores[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.set_facecolor("#f7fafc")
    fig.patch.set_facecolor("#f7fafc")

    ax.plot(angles, scores, "o-", linewidth=2, color="#4299e1")
    ax.fill(angles, scores, alpha=0.25, color="#4299e1")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(roles, size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%","50%","75%","100%"], size=7, color="#718096")
    ax.grid(color="#e2e8f0", linestyle="--", linewidth=0.5)
    ax.set_title("Skill Coverage", size=11, pad=14, color="#2d3748")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar_profile():
    st.sidebar.header("👤 Профиль")
    with open(ROOT / "data" / "profile.json", encoding="utf-8") as f:
        default = json.load(f)

    name      = st.sidebar.text_input("Имя", value=default.get("name",""))
    hard_raw  = st.sidebar.text_area("Hard Skills (через запятую)",
                    value=", ".join(default.get("hard_skills",[])), height=80)
    soft_raw  = st.sidebar.text_area("Soft Skills (через запятую)",
                    value=", ".join(default.get("soft_skills",[])), height=60)
    inter_raw = st.sidebar.text_area("Интересы (через запятую)",
                    value=", ".join(default.get("interests",[])), height=60)
    top_n     = st.sidebar.slider("Топ-N профессий", 1, 5, 3)

    st.sidebar.divider()
    llm_key = st.sidebar.text_input("🔑 LLM API Key (опц.)", type="password",
                                    placeholder="sk-ant-... или sk-...")
    if llm_key:
        os.environ["LLM_API_KEY"] = llm_key
        st.sidebar.success("LLM активирован")

    use_sem = st.sidebar.toggle("🧠 Semantic Matching", value=False)
    st.sidebar.caption("💡 Вкладка **Resume** для авто-заполнения из резюме")

    parse = lambda raw: [s.strip() for s in raw.split(",") if s.strip()]
    ns    = normalize_skills(parse(hard_raw)) | normalize_skills(parse(soft_raw))
    return {
        "name": name, "hard_skills": parse(hard_raw),
        "soft_skills": parse(soft_raw), "interests": parse(inter_raw),
        "_hard_skills_norm": normalize_skills(parse(hard_raw)),
        "_soft_skills_norm": normalize_skills(parse(soft_raw)),
        "_all_skills_norm":  ns,
    }, top_n, use_sem


# ---------------------------------------------------------------------------
# Tab 1 — Predict
# ---------------------------------------------------------------------------
def tab_predict(profile, top_n, use_semantic):
    col_l, col_r = st.columns([1, 2], gap="large")

    with col_l:
        st.subheader(f"👤 {profile['name']}")
        st.markdown("**Hard Skills:**")
        st.markdown(tags(profile["hard_skills"]), unsafe_allow_html=True)
        st.markdown("**Soft Skills:**")
        st.markdown(tags(profile["soft_skills"]), unsafe_allow_html=True)
        st.markdown("**Интересы:**")
        st.markdown(tags(profile["interests"]), unsafe_allow_html=True)
        st.divider()
        predict_btn = st.button("🚀 Predict", type="primary", use_container_width=True)

    with col_r:
        if not predict_btn and "results" not in st.session_state:
            st.info("👈 Настрой профиль и нажми **Predict**")
            return

        if predict_btn:
            jobs    = load_jobs(ROOT / "data" / "jobs.csv")
            results = predict_top_roles(profile, jobs, top_n=top_n)
            if use_semantic:
                for r in results:
                    sem = semantic_match_score(
                        profile["hard_skills"],
                        r["missing_skills"] + r["matched_skills"])
                    r["hybrid_score"] = hybrid_score(r["score"], sem["score"])
                    r["hybrid_pct"]   = int(r["hybrid_score"] * 100)
                results.sort(key=lambda x: x.get("hybrid_score", x["score"]), reverse=True)
            st.session_state["results"] = results
            st.session_state["profile"] = profile

        results   = st.session_state["results"]
        score_key = "hybrid_pct" if use_semantic else "score_pct"

        # Charts: bar + radar
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("📊 Bar Chart")
            df = pd.DataFrame({
                "Роль":      [r["role"] for r in results],
                "Score (%)": [r.get(score_key, r["score_pct"]) for r in results],
            }).set_index("Роль")
            st.bar_chart(df, color="#4299e1")
        with c2:
            st.subheader("🕸️ Radar")
            all_jobs    = load_jobs(ROOT / "data" / "jobs.csv")
            all_results = predict_top_roles(profile, all_jobs, top_n=len(all_jobs))
            fig = radar_chart(all_results, score_key)
            if fig:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.caption("Нужно ≥ 3 роли для radar chart")

        if use_semantic:
            st.caption(f"🧠 TF-IDF {'✅' if SKLEARN_AVAILABLE else '⚠️ fallback: overlap'}")

        # Role cards
        st.subheader("🏆 Результаты")
        medals = ["🥇","🥈","🥉","4️⃣","5️⃣"]
        for i, r in enumerate(results):
            pct = r.get(score_key, r["score_pct"])
            with st.expander(f"{medals[i]} **{r['role']}** — {pct}%", expanded=(i==0)):
                st.progress(pct / 100)
                st.caption(r["description"])
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**✅ Есть:**")
                    st.markdown(tags(r["matched_skills"], "skill-match"), unsafe_allow_html=True)
                with c2:
                    st.markdown("**❌ Нужно:**")
                    st.markdown(tags(r["missing_skills"], "skill-missing"), unsafe_allow_html=True)

                if r["missing_skills"]:
                    st.markdown("---")
                    st.markdown("**📚 Ресурсы:**")
                    sel = st.selectbox("Навык:", r["missing_skills"], key=f"res_{i}")
                    res = find_resources(sel)
                    src = {"static":"📖","llm":"🤖","template":"🔗"}.get(res["source"],"🔗")
                    st.caption(f"Источник: {src}")
                    r1,r2,r3,r4 = st.columns(4)
                    with r1:
                        st.markdown("📄 **Docs**")
                        for u in res.get("docs",[]): st.markdown(f"[Открыть]({u})")
                    with r2:
                        st.markdown("🎓 **Курсы**")
                        for u in res.get("courses",[]): st.markdown(f"[Открыть]({u})")
                    with r3:
                        st.markdown("💻 **GitHub**")
                        for u in res.get("github",[]): st.markdown(f"[Открыть]({u})")
                    with r4:
                        st.markdown("🎬 **Видео**")
                        for u in res.get("video",[]): st.markdown(f"[Открыть]({u})")

        st.divider()
        st.download_button("📥 Скачать отчёт (Markdown)",
            data=export_md(profile, results),
            file_name="career_report.md", mime="text/markdown",
            use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 — Roadmap
# ---------------------------------------------------------------------------
def tab_roadmap(profile):
    st.subheader("🗺️ Персональный Career Roadmap")
    jobs       = load_jobs(ROOT / "data" / "jobs.csv")
    role_names = [j["role"] for j in jobs]
    target     = st.selectbox("Целевая роль:", role_names)
    results    = predict_top_roles(profile, jobs, top_n=len(jobs))
    role_data  = next((r for r in results if r["role"] == target), None)
    if not role_data: return

    c1,c2,c3 = st.columns(3)
    with c1: st.markdown(f'<div class="metric-box"><b>Match</b><br><h2>{role_data["score_pct"]}%</h2></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-box"><b>Есть</b><br><h2>{len(role_data["matched_skills"])}</h2></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-box"><b>Нужно</b><br><h2>{len(role_data["missing_skills"])}</h2></div>', unsafe_allow_html=True)

    st.divider()
    if st.button("🤖 Сгенерировать Roadmap", type="primary"):
        with st.spinner("Строю план..."):
            rm = generate_roadmap(
                profile=profile, target_role=target,
                missing_skills=role_data["missing_skills"],
                matched_skills=role_data["matched_skills"],
                score_pct=role_data["score_pct"],
            )
            st.session_state["roadmap"] = rm

    if "roadmap" not in st.session_state: return
    rm  = st.session_state["roadmap"]
    src = "🤖 LLM" if rm.get("source")=="llm" else "📋 Template"
    st.caption(f"Источник: {src}")

    lvl = {"high":"🟢","medium":"🟡","low":"🔴"}.get(rm.get("readiness_level","medium"),"🟡")
    st.info(f"{lvl} {rm.get('readiness','')}")

    if rm.get("total_months",0) > 0:
        st.markdown(f"**⏱ Время:** {rm['total_months']} мес. ({rm['total_weeks']} нед.)")
    if rm.get("quick_wins"):
        st.markdown("**⚡ Быстрые победы:**")
        st.markdown(tags(rm["quick_wins"]), unsafe_allow_html=True)
    if rm.get("phases"):
        st.markdown("### 📅 Фазы")
        for p in rm["phases"]:
            desc = f'<br><small>{p.get("description","")}</small>' if p.get("description") else ""
            st.markdown(
                f'<div class="phase-card"><b>{p["title"]}</b> · {p["weeks"]} нед.'
                f'<br>{tags(p["skills"])}{desc}</div>',
                unsafe_allow_html=True)
    if rm.get("tips"):
        st.markdown("### 💡 Советы")
        for t in rm["tips"]: st.markdown(f"- {t}")
    if rm.get("strengths"):
        st.markdown("### 💪 Сильные стороны")
        st.markdown(tags(rm["strengths"], "skill-match"), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 3 — Interview Coach
# ---------------------------------------------------------------------------
def tab_interview(profile):
    st.subheader("🎤 Interview Coach")
    st.caption("Подготовься к собеседованию — получи вопросы по твоим пробелам")

    jobs      = load_jobs(ROOT / "data" / "jobs.csv")
    role_names = [j["role"] for j in jobs]
    target    = st.selectbox("Целевая роль:", role_names, key="interview_role")
    results   = predict_top_roles(profile, jobs, top_n=len(jobs))
    role_data = next((r for r in results if r["role"] == target), None)
    if not role_data: return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**❌ Пробелы (вопросы по ним):**")
        st.markdown(tags(role_data["missing_skills"], "skill-missing"), unsafe_allow_html=True)
    with col2:
        st.markdown("**✅ Сильные стороны:**")
        st.markdown(tags(role_data["matched_skills"], "skill-match"), unsafe_allow_html=True)

    n_per = st.slider("Вопросов на навык:", 1, 4, 2)
    st.divider()

    if st.button("🎯 Сгенерировать вопросы", type="primary"):
        with st.spinner("Подбираю вопросы..."):
            qs = generate_interview_questions(
                target_role    = target,
                missing_skills = role_data["missing_skills"],
                matched_skills = role_data["matched_skills"],
                n_per_skill    = n_per,
            )
            st.session_state["interview_qs"] = qs

    if "interview_qs" not in st.session_state: return
    qs  = st.session_state["interview_qs"]
    src = "🤖 LLM" if qs.get("source")=="llm" else "📖 Static bank"
    st.caption(f"Источник: {src}")

    if qs.get("missing"):
        st.markdown("### ⚠️ Вопросы по пробелам")
        st.caption("Именно эти темы интервьюер скорее всего проверит у тебя")
        for q in qs["missing"]:
            st.markdown(
                f'<div class="q-card"><b>[{q["skill"]}]</b> {q["question"]}<br>'
                f'<small>💡 {q["hint"]}</small></div>',
                unsafe_allow_html=True)

    if qs.get("strengths"):
        st.markdown("### 💪 Вопросы по сильным сторонам")
        st.caption("Покажи глубину — не ограничивайся базовыми ответами")
        for q in qs["strengths"]:
            st.markdown(
                f'<div class="q-card" style="border-color:#68d391"><b>[{q["skill"]}]</b> {q["question"]}<br>'
                f'<small>💡 {q["hint"]}</small></div>',
                unsafe_allow_html=True)

    if qs.get("behavioral"):
        st.markdown("### 🧠 Поведенческие вопросы (STAR)")
        for q in qs["behavioral"]:
            hint = q.get("hint", "")
            st.markdown(
                f'<div class="q-card" style="border-color:#f6ad55">{q["question"]}<br>'
                + (f'<small>💡 {hint}</small>' if hint else "") + "</div>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 4 — Job Match
# ---------------------------------------------------------------------------
def tab_job_match(profile):
    st.subheader("💼 Job Description Matcher")
    st.caption("Вставь текст вакансии — получи анализ соответствия твоему профилю")

    has_llm = bool(os.environ.get("LLM_API_KEY",""))
    mode    = "🤖 LLM" if has_llm else "📖 Regex"
    st.caption(f"Режим парсинга: {mode}")

    sample_jd = """Senior Backend Engineer — Fintech Startup

We're looking for a Backend Engineer to build our payment platform.

Required:
- Python (3+ years), PostgreSQL, Redis
- Docker, Kubernetes
- REST API, Microservices
- Git, CI/CD

Nice to have:
- AWS
- System Design experience
- Kafka"""

    jd_text = st.text_area("Текст вакансии:", value=sample_jd, height=220)

    if st.button("🔍 Проанализировать", type="primary"):
        with st.spinner("Парсю вакансию..."):
            parsed = parse_job_description(jd_text, use_llm=has_llm)
            gap    = gap_analysis(parsed, profile)
            st.session_state["job_parsed"] = parsed
            st.session_state["job_gap"]    = gap

    if "job_parsed" not in st.session_state: return
    parsed = st.session_state["job_parsed"]
    gap    = st.session_state["job_gap"]

    # Заголовок
    seniority_emoji = {"senior":"🔴","middle":"🟡","junior":"🟢"}.get(parsed.get("seniority","middle"),"🟡")
    st.markdown(f"### {seniority_emoji} {parsed.get('role_name','Роль')} · {parsed.get('seniority','').capitalize()}")

    src = parsed.get("_meta",{}).get("source","")
    st.caption(f"Парсер: {'🤖 LLM' if 'llm' in src else '📖 Regex'}")

    # Fit level
    fit_labels = {"strong":"🟢 Strong fit","good":"🟡 Good fit",
                  "partial":"🟠 Partial fit","weak":"🔴 Weak fit"}
    fit_colors = {"strong":"fit-strong","good":"fit-good","partial":"fit-partial","weak":"fit-weak"}
    fit = gap["fit_level"]
    st.markdown(
        f'<h3 class="{fit_colors[fit]}">{fit_labels[fit]} — {gap["match_pct"]}%</h3>',
        unsafe_allow_html=True)
    st.progress(gap["match_pct"] / 100)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**✅ У тебя уже есть:**")
        st.markdown(tags(gap["you_have"], "skill-match"), unsafe_allow_html=True)
        if parsed.get("nice_to_have"):
            st.markdown("**⭐ Nice to have в вакансии:**")
            st.markdown(tags(parsed["nice_to_have"]), unsafe_allow_html=True)
    with col2:
        st.markdown("**❌ Нужно получить:**")
        st.markdown(tags(gap["you_need"], "skill-missing"), unsafe_allow_html=True)
        if gap.get("nice_missing"):
            st.markdown("**💡 Nice to have (нет у тебя):**")
            st.markdown(tags(gap["nice_missing"], "skill-missing"), unsafe_allow_html=True)

    if parsed.get("responsibilities"):
        st.markdown("**📋 Обязанности:**")
        for r in parsed["responsibilities"]:
            st.markdown(f"- {r}")

    if gap["you_need"]:
        st.divider()
        st.markdown("**📚 Ресурсы по нужным навыкам:**")
        sel = st.selectbox("Навык:", gap["you_need"], key="job_skill_res")
        res = find_resources(sel)
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.markdown("📄 **Docs**")
            for u in res.get("docs",[]): st.markdown(f"[Открыть]({u})")
        with c2:
            st.markdown("🎓 **Курсы**")
            for u in res.get("courses",[]): st.markdown(f"[Открыть]({u})")
        with c3:
            st.markdown("💻 **GitHub**")
            for u in res.get("github",[]): st.markdown(f"[Открыть]({u})")
        with c4:
            st.markdown("🎬 **Видео**")
            for u in res.get("video",[]): st.markdown(f"[Открыть]({u})")


# ---------------------------------------------------------------------------
# Tab 5 — Resume Parser
# ---------------------------------------------------------------------------
def tab_resume():
    st.subheader("📄 Resume Parser")
    st.caption("Вставь текст резюме — система извлечёт навыки автоматически")

    has_llm = bool(os.environ.get("LLM_API_KEY",""))
    if has_llm: st.success("🤖 LLM-режим активен")
    else:       st.info("📖 Regex fallback (добавь LLM_API_KEY для AI-парсинга)")

    sample = """John Doe — Backend Developer
Skills: Python, FastAPI, PostgreSQL, Docker, Git, Redis, AWS, Linux, Bash
Soft skills: problem solving, teamwork, communication
Interests: backend, cloud, automation, AI"""

    text = st.text_area("Текст резюме:", value=sample, height=180)

    if st.button("🔍 Разобрать резюме", type="primary"):
        with st.spinner("Анализирую..."):
            parsed = parse_resume(text, use_llm=has_llm)

        src = parsed.get("_meta",{}).get("source","")
        st.success(f"✅ Готово · {'🤖 LLM' if 'llm' in src else '📖 Regex'}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Имя:** {parsed.get('name','—')}")
            st.markdown("**Hard Skills:**")
            st.markdown(tags(parsed.get("hard_skills",[])), unsafe_allow_html=True)
        with c2:
            st.markdown("**Soft Skills:**")
            st.markdown(tags(parsed.get("soft_skills",[])), unsafe_allow_html=True)
            st.markdown("**Интересы:**")
            st.markdown(tags(parsed.get("interests",[])), unsafe_allow_html=True)

        st.divider()
        display = {k:v for k,v in parsed.items() if not k.startswith("_")}
        st.code(json.dumps(display, indent=2, ensure_ascii=False), language="json")

        if st.button("💾 Сохранить как profile.json"):
            with open(ROOT / "data" / "profile.json", "w", encoding="utf-8") as f:
                json.dump(display, f, indent=2, ensure_ascii=False)
            st.success("Сохранено! Перезагрузи страницу.")


# ---------------------------------------------------------------------------
# Tab 6 — About
# ---------------------------------------------------------------------------
def tab_about():
    st.subheader("🏗️ Архитектура")
    st.markdown("""
```
 Input: profile.json / resume text / job description
           │
           ▼
 ┌─────────────────────────────────────────────────┐
 │  Platform Layer                                 │
 │  profile.json · jobs.csv                        │
 │  resume_parser.py  ◄── LLM / Regex              │
 │  job_parser.py     ◄── LLM / Regex              │
 └──────────────────┬──────────────────────────────┘
                    ▼
 ┌─────────────────────────────────────────────────┐
 │  Model Layer                                    │
 │  predictor.py        ── overlap + алиасы        │
 │  semantic_matcher.py ── TF-IDF cosine           │
 └──────────────────┬──────────────────────────────┘
                    ▼
 ┌─────────────────────────────────────────────────┐
 │  Agent Layer (GenAI)                            │
 │  resource_finder.py ── курсы / docs / GitHub    │
 │  career_coach.py    ── roadmap генератор        │
 │  interview_coach.py ── вопросы для интервью     │
 └──────────────────┬──────────────────────────────┘
                    ▼
 ┌─────────────────────────────────────────────────┐
 │  Application Layer — Streamlit (6 вкладок)      │
 │  Predict · Roadmap · Interview · Job · Resume   │
 └─────────────────────────────────────────────────┘
```
""")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**🤖 GenAI компоненты:**
- `resume_parser` — LLM извлекает навыки из резюме
- `job_parser` — LLM парсит вакансию
- `career_coach` — LLM генерирует roadmap
- `interview_coach` — LLM создаёт вопросы интервью
- `resource_finder` — LLM подбирает ресурсы
- `semantic_matcher` — TF-IDF cosine similarity
""")
    with col2:
        st.markdown("""
**🔌 Fallback (без API-ключа):**
- Resume Parser → словарь 60+ навыков
- Job Parser → regex + паттерны секций
- Career Coach → шаблон с приоритетами
- Interview Coach → банк 100+ вопросов
- Resource Finder → статический словарь
- Semantic → exact overlap score
""")
    st.divider()
    st.markdown("""
**⚙️ Стек:** Python 3.10 · Streamlit · Pandas · Matplotlib · scikit-learn · Claude/GPT

**🚀 Запуск:**
```bash
pip install -r requirements.txt
streamlit run app/app.py
```
""")



# ---------------------------------------------------------------------------
# Tab — PDF Upload
# ---------------------------------------------------------------------------
def tab_pdf_upload():
    st.subheader("📎 Загрузить резюме (PDF)")
    st.caption("Загрузи PDF — система автоматически извлечёт навыки и обновит профиль")

    from utils.pdf_parser import get_available_backend
    backend = get_available_backend()
    if "не установлена" in backend:
        st.warning(
            "⚠️ PDF-библиотека не установлена.\n\n"
            "```bash\npip install pdfplumber\n```"
        )
    else:
        st.success(f"✅ PDF backend: `{backend}`")

    uploaded = st.file_uploader("Выбери PDF-файл резюме:", type=["pdf"])

    if uploaded is not None:
        st.info(f"📄 Файл: `{uploaded.name}` ({uploaded.size // 1024} KB)")

        has_llm = bool(os.environ.get("LLM_API_KEY",""))
        if st.button("🔍 Разобрать PDF", type="primary"):
            with st.spinner("Читаю PDF и извлекаю навыки..."):
                try:
                    from utils.pdf_parser import parse_pdf_resume
                    pdf_bytes = uploaded.read()
                    profile   = parse_pdf_resume(pdf_bytes, use_llm=has_llm)

                    src = profile.get("_meta",{}).get("source","")
                    chars = profile.get("_meta",{}).get("pdf_chars", 0)
                    st.success(f"✅ Извлечено {chars} символов · {'🤖 LLM' if 'llm' in src else '📖 Regex'}")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Имя:** {profile.get('name','—')}")
                        st.markdown("**Hard Skills:**")
                        st.markdown(tags(profile.get("hard_skills",[])), unsafe_allow_html=True)
                    with c2:
                        st.markdown("**Soft Skills:**")
                        st.markdown(tags(profile.get("soft_skills",[])), unsafe_allow_html=True)
                        st.markdown("**Интересы:**")
                        st.markdown(tags(profile.get("interests",[])), unsafe_allow_html=True)

                    st.divider()
                    display = {k:v for k,v in profile.items() if not k.startswith("_")}
                    st.code(json.dumps(display, indent=2, ensure_ascii=False), language="json")

                    if st.button("💾 Сохранить как profile.json"):
                        with open(ROOT / "data" / "profile.json", "w", encoding="utf-8") as ff:
                            json.dump(display, ff, indent=2, ensure_ascii=False)
                        st.success("Сохранено! Перезагрузи страницу для применения.")

                except RuntimeError as e:
                    st.error(str(e))
                    st.code("pip install pdfplumber", language="bash")
                except Exception as e:
                    st.error(f"Ошибка: {e}")


# ---------------------------------------------------------------------------
# Tab — Telegram Jobs
# ---------------------------------------------------------------------------
def tab_telegram_jobs(profile):
    st.subheader("📡 Telegram Job Search")
    st.caption("Поиск вакансий по IT-каналам Telegram с матчингом по твоему профилю")

    # Статус подключения — перечитываем .env каждый раз при рендере вкладки
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=True)
    except ImportError:
        pass

    _tg_id    = os.environ.get("TELEGRAM_API_ID", "").strip()
    _tg_hash  = os.environ.get("TELEGRAM_API_HASH", "").strip()
    _tg_phone = os.environ.get("TELEGRAM_PHONE", "").strip()
    _session  = ROOT / "data" / ".telegram_session.session"

    has_creds   = bool(_tg_id and _tg_hash and _tg_phone)
    has_session = _session.exists()
    has_tg      = has_creds and has_session   # готов к реальному поиску

    if has_tg:
        st.success("✅ Telegram API настроен и авторизован — используется реальный поиск")
    elif has_creds and not has_session:
        st.warning(
            "⚠️ Ключи найдены в .env, но сессия не создана. "
            "Запусти один раз из терминала:\n\n"
            "```\npython telegram_bot/auth.py\n```\n"
            "После ввода SMS-кода сессия сохранится и здесь появится реальный поиск."
        )
    else:
        st.info(
            "📋 Telegram API не настроен → показываются демо-данные (mock).\n\n"
            "Заполни в `.env`:\n"
            "```\nTELEGRAM_API_ID=...\nTELEGRAM_API_HASH=...\nTELEGRAM_PHONE=+7...\n```\n"
            "Ключи получить на https://my.telegram.org/apps"
        )

    # Настройки поиска
    col1, col2, col3 = st.columns(3)
    with col1:
        days_back   = st.slider("Глубина поиска (дней):", 1, 30, 7)
    with col2:
        min_match   = st.slider("Мин. совпадение (%):", 0, 80, 25)
    with col3:
        max_results = st.slider("Макс. результатов:", 3, 20, 8)

    # ── Добавить кастомный канал ──────────────────────────────────────────────
    with st.expander("➕ Добавить свой канал", expanded=False):
        col_a, col_b, col_c = st.columns([2, 2, 1])
        with col_a:
            custom_username = st.text_input(
                "Username канала:", placeholder="dev_kg  (без @)",
                key="custom_ch_input"
            )
        with col_b:
            custom_name = st.text_input(
                "Название (опц.):", placeholder="Dev KG",
                key="custom_ch_name"
            )
        with col_c:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Добавить", key="add_ch_btn"):
                if custom_username.strip():
                    from telegram_bot.job_scraper import add_custom_channel
                    added = add_custom_channel(
                        custom_username.strip().lstrip("@"),
                        name=custom_name.strip() or custom_username.strip()
                    )
                    if added:
                        st.success(f"✅ Канал @{custom_username.strip().lstrip('@')} добавлен!")
                        st.rerun()
                    else:
                        st.info("Канал уже есть в списке")
                else:
                    st.warning("Введи username канала")

    # ── Фильтр каналов ────────────────────────────────────────────────────────
    with open(ROOT / "data" / "telegram_channels.json", encoding="utf-8") as ff:
        ch_config = json.load(ff)
    all_channels  = ch_config["channels"]
    ch_options    = {c["name"]: c["username"] for c in all_channels}

    # dev_kg выбран по умолчанию если есть
    default_channels = [
        n for n, u in ch_options.items()
        if u in ("dev_kg", "getmatch", "python_jobs_ru", "devops_jobs_ru", "backend_jobs_ru")
    ][:5]

    selected_names = st.multiselect(
        "Каналы для поиска:",
        options=list(ch_options.keys()),
        default=default_channels or list(ch_options.keys())[:5],
    )
    selected_ch = [ch_options[n] for n in selected_names]

    st.divider()
    if st.button("🔍 Найти вакансии", type="primary"):
        with st.spinner("Ищу вакансии..."):
            try:
                from telegram_bot.job_scraper import scrape_jobs
                jobs = scrape_jobs(
                    profile       = profile,
                    channels      = selected_ch if selected_ch else None,
                    days_back     = days_back,
                    min_match_pct = min_match,
                    use_mock      = not has_tg,
                )
                st.session_state["tg_jobs"] = jobs
            except Exception as e:
                st.error(f"Ошибка: {e}")
                return

    if "tg_jobs" not in st.session_state:
        return

    jobs = st.session_state["tg_jobs"]

    if not jobs:
        st.warning("Вакансий не найдено. Снизь порог совпадения или расширь каналы.")
        return

    st.markdown(f"**Найдено: {len(jobs)} вакансий** (показаны топ-{min(max_results, len(jobs))})")

    seniority_emoji = {"senior":"🔴","middle":"🟡","junior":"🟢"}
    fit_color = {
        range(80, 101): "skill-match",
        range(50, 80):  "skill-tag",
        range(0, 50):   "skill-missing",
    }

    def score_cls(pct):
        if pct >= 80: return "skill-match"
        if pct >= 50: return "skill-tag"
        return "skill-missing"

    for j in jobs[:max_results]:
        emoji = seniority_emoji.get(j.seniority, "🟡")
        with st.expander(
            f"{emoji} **{j.role_name}** — {j.match_pct}% | {j.channel_name} · {j.date.strftime('%d.%m')}",
            expanded=False,
        ):
            st.progress(j.match_pct / 100)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**✅ У тебя есть:**")
                st.markdown(tags(j.you_have, "skill-match"), unsafe_allow_html=True)
            with c2:
                st.markdown("**❌ Нужно получить:**")
                st.markdown(tags(j.you_need, "skill-missing"), unsafe_allow_html=True)

            st.markdown(f"**Текст вакансии:**")
            preview = j.text[:600] + ("..." if len(j.text) > 600 else "")
            st.text(preview)
            st.markdown(f"[🔗 Открыть в Telegram]({j.url})")

            # Быстрые ресурсы по нужным навыкам
            if j.you_need:
                skill_res = st.selectbox(
                    "Ресурсы для изучения:", j.you_need, key=f"tg_res_{j.id}"
                )
                res = find_resources(skill_res)
                r1,r2,r3 = st.columns(3)
                with r1:
                    for u in res.get("docs",[]): st.markdown(f"📄 [Docs]({u})")
                with r2:
                    for u in res.get("courses",[]): st.markdown(f"🎓 [Course]({u})")
                with r3:
                    for u in res.get("github",[]): st.markdown(f"💻 [GitHub]({u})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.title("🎯 Digital Twin Career Engine")
    st.caption("Анализ навыков · GenAI Roadmap · Interview Prep · Telegram Jobs · PDF Parser")

    profile, top_n, use_sem = sidebar_profile()

    t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
        "🎯 Predict", "🗺️ Roadmap", "🎤 Interview",
        "💼 Job Match", "📡 Telegram Jobs", "📎 PDF Upload",
        "📄 Resume", "ℹ️ About"
    ])
    with t1: tab_predict(profile, top_n, use_sem)
    with t2: tab_roadmap(profile)
    with t3: tab_interview(profile)
    with t4: tab_job_match(profile)
    with t5: tab_telegram_jobs(profile)
    with t6: tab_pdf_upload()
    with t7: tab_resume()
    with t8: tab_about()

if __name__ == "__main__":
    main()
