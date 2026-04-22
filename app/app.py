"""
Application Layer — Streamlit UI для Digital Twin Career Engine.

Запуск: streamlit run app/app.py
"""

import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd

# Путь к корню проекта (чтобы импорты работали из любого cwd)
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.predictor import load_profile, load_jobs, predict_top_roles
from agent.resource_finder import find_resources

# ---------------------------------------------------------------------------
# Конфигурация страницы
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Digital Twin Career Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS-кастомизация
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  .skill-tag {
    display: inline-block;
    background: #e8f4f8;
    border: 1px solid #90cdf4;
    border-radius: 12px;
    padding: 2px 10px;
    margin: 2px;
    font-size: 0.82em;
    color: #2b6cb0;
  }
  .skill-missing {
    background: #fff5f5;
    border-color: #fc8181;
    color: #c53030;
  }
  .role-card {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
    background: #f7fafc;
  }
  .score-high   { color: #276749; font-weight: bold; }
  .score-medium { color: #744210; font-weight: bold; }
  .score-low    { color: #742a2a; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def score_color(score_pct: int) -> str:
    if score_pct >= 70:
        return "score-high"
    elif score_pct >= 40:
        return "score-medium"
    return "score-low"


def render_skill_tags(skills: list[str], missing: bool = False) -> str:
    cls = "skill-tag skill-missing" if missing else "skill-tag"
    return " ".join(f'<span class="{cls}">{s}</span>' for s in skills)


def export_report(profile: dict, top_roles: list[dict]) -> str:
    """Генерирует Markdown-отчёт."""
    lines = [
        f"# 🎯 Career Report — {profile.get('name', 'User')}",
        "",
        "## 👤 Профиль",
        f"**Hard skills:** {', '.join(profile.get('hard_skills', []))}",
        f"**Soft skills:** {', '.join(profile.get('soft_skills', []))}",
        f"**Интересы:** {', '.join(profile.get('interests', []))}",
        "",
        "## 🏆 Топ профессий",
    ]
    for i, r in enumerate(top_roles, 1):
        lines += [
            f"### {i}. {r['role']} — {r['score_pct']}%",
            f"> {r['description']}",
            "",
            f"**✅ Совпавшие навыки:** {', '.join(r['matched_skills']) or '—'}",
            f"**❌ Недостающие навыки:** {', '.join(r['missing_skills']) or '—'}",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sidebar — профиль
# ---------------------------------------------------------------------------

def sidebar_profile() -> dict:
    st.sidebar.header("👤 Профиль пользователя")

    profile_path = ROOT / "data" / "profile.json"
    with open(profile_path, encoding="utf-8") as f:
        default_profile = json.load(f)

    st.sidebar.markdown("**Имя**")
    name = st.sidebar.text_input("Имя", value=default_profile.get("name", ""), label_visibility="collapsed")

    st.sidebar.markdown("**Hard Skills** (через запятую)")
    hard_raw = st.sidebar.text_area(
        "Hard skills",
        value=", ".join(default_profile.get("hard_skills", [])),
        height=80,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("**Soft Skills** (через запятую)")
    soft_raw = st.sidebar.text_area(
        "Soft skills",
        value=", ".join(default_profile.get("soft_skills", [])),
        height=60,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("**Интересы** (через запятую)")
    interests_raw = st.sidebar.text_area(
        "Interests",
        value=", ".join(default_profile.get("interests", [])),
        height=60,
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.caption("🔮 Future: загрузка резюме (PDF/LinkedIn)")
    # uploaded = st.sidebar.file_uploader("Загрузить резюме", type=["pdf"])

    return {
        "name": name,
        "hard_skills": [s.strip() for s in hard_raw.split(",") if s.strip()],
        "soft_skills":  [s.strip() for s in soft_raw.split(",")  if s.strip()],
        "interests":    [s.strip() for s in interests_raw.split(",") if s.strip()],
    }


# ---------------------------------------------------------------------------
# Главная страница
# ---------------------------------------------------------------------------

def main():
    st.title("🎯 Digital Twin Career Engine")
    st.caption("Анализ навыков и предсказание карьерных направлений")

    # ── Профиль из sidebar ──────────────────────────────────────────────────
    raw_profile = sidebar_profile()

    # Нормализуем через predictor
    from model.predictor import normalize_skills
    profile = {
        **raw_profile,
        "_hard_skills_norm": normalize_skills(raw_profile["hard_skills"]),
        "_soft_skills_norm": normalize_skills(raw_profile["soft_skills"]),
    }
    profile["_all_skills_norm"] = (
        profile["_hard_skills_norm"] | profile["_soft_skills_norm"]
    )

    # ── Колонки ─────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 2], gap="large")

    # ── Левая колонка: профиль ──────────────────────────────────────────────
    with col_left:
        st.subheader(f"👤 {profile['name']}")

        st.markdown("**Hard Skills:**")
        st.markdown(
            render_skill_tags(profile["hard_skills"]) or "_не указаны_",
            unsafe_allow_html=True,
        )

        st.markdown("**Soft Skills:**")
        st.markdown(
            render_skill_tags(profile["soft_skills"]) or "_не указаны_",
            unsafe_allow_html=True,
        )

        st.markdown("**Интересы:**")
        st.markdown(
            render_skill_tags(profile["interests"]) or "_не указаны_",
            unsafe_allow_html=True,
        )

        st.divider()
        top_n = st.slider("Количество профессий", min_value=1, max_value=5, value=3)
        predict_btn = st.button("🚀 Predict", type="primary", use_container_width=True)

    # ── Правая колонка: результаты ──────────────────────────────────────────
    with col_right:
        if not predict_btn and "last_results" not in st.session_state:
            st.info("👈 Настрой профиль и нажми **Predict**")
            return

        if predict_btn:
            jobs = load_jobs(ROOT / "data" / "jobs.csv")
            results = predict_top_roles(profile, jobs, top_n=top_n)
            st.session_state["last_results"] = results
            st.session_state["last_profile"]  = profile

        results = st.session_state["last_results"]
        profile  = st.session_state["last_profile"]

        # ── Bar chart ──────────────────────────────────────────────────────
        st.subheader("📊 Score по профессиям")
        chart_data = pd.DataFrame({
            "Роль":  [r["role"] for r in results],
            "Score (%)": [r["score_pct"] for r in results],
        }).set_index("Роль")
        st.bar_chart(chart_data, color="#4299e1")

        # ── Топ ролей ──────────────────────────────────────────────────────
        st.subheader("🏆 Топ профессий")

        for i, r in enumerate(results):
            with st.expander(
                f"{'🥇' if i==0 else '🥈' if i==1 else '🥉'} "
                f"**{r['role']}** — {r['score_pct']}%",
                expanded=(i == 0),
            ):
                st.progress(r["score_pct"] / 100)
                st.caption(r["description"])

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**✅ Совпавшие навыки:**")
                    if r["matched_skills"]:
                        st.markdown(
                            render_skill_tags(r["matched_skills"]),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown("_нет совпадений_")

                with c2:
                    st.markdown("**❌ Недостающие навыки:**")
                    if r["missing_skills"]:
                        st.markdown(
                            render_skill_tags(r["missing_skills"], missing=True),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown("_все навыки есть! 🎉_")

                # ── Ресурсы по недостающим навыкам ──────────────────────
                if r["missing_skills"]:
                    st.markdown("---")
                    st.markdown("**📚 Ресурсы для изучения:**")

                    selected_skill = st.selectbox(
                        "Выбери навык для поиска ресурсов:",
                        r["missing_skills"],
                        key=f"skill_select_{i}",
                    )

                    res = find_resources(selected_skill)
                    src_label = "🤖 LLM" if res["source"] == "llm" else \
                                "📖 Static" if res["source"] == "static" else "🔗 Template"

                    st.caption(f"Источник: {src_label}")

                    r1, r2, r3, r4 = st.columns(4)
                    with r1:
                        st.markdown("**📄 Документация**")
                        for url in res.get("docs", []):
                            st.markdown(f"[🔗 Open]({url})")
                    with r2:
                        st.markdown("**🎓 Курсы**")
                        for url in res.get("courses", []):
                            st.markdown(f"[🔗 Open]({url})")
                    with r3:
                        st.markdown("**💻 GitHub**")
                        for url in res.get("github", []):
                            st.markdown(f"[🔗 Open]({url})")
                    with r4:
                        st.markdown("**🎬 Видео**")
                        for url in res.get("video", []):
                            st.markdown(f"[🔗 Open]({url})")

        # ── Экспорт отчёта ─────────────────────────────────────────────────
        st.divider()
        report_md = export_report(profile, results)
        st.download_button(
            label="📥 Скачать отчёт (Markdown)",
            data=report_md,
            file_name="career_report.md",
            mime="text/markdown",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
