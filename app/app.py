"""
Application Layer -- Streamlit UI
Digital Twin Career Engine v2

Tabs:
  1. CV Upload   -- PDF + text resume, auto-profile
  2. CV Roast    -- AI critique of your CV
  3. Telegram    -- job search + infographics (connected to CV)
  4. Roadmap     -- GenAI career plan
  5. Interview   -- interview preparation
  6. About       -- architecture

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
from utils.resume_parser    import parse_resume

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
        issues.append(("No name", "Anonymous resumes get rejected first. Add your full name."))

    n_hard = len(hard_orig)
    if n_hard == 0:
        issues.append(("Zero hard skills", "No skills = no resume. Add everything you have worked with."))
    elif n_hard < 5:
        issues.append(("Too few skills",
                        f"Only {n_hard} hard skill(s). Recruiters expect 6-8 minimum. "
                        "Add frameworks, tools, versions -- everything you have used."))
    elif n_hard >= 15:
        praise.append(("Wide tech stack", f"{n_hard} hard skills -- excellent coverage."))
    else:
        praise.append(("Decent stack", f"{n_hard} hard skills -- solid starting point."))

    langs = {"python","java","go","golang","javascript","typescript",
             "rust","c++","c#","kotlin","swift","scala","php","ruby"}
    found_langs = [s for s in hard if s in langs]
    if not found_langs:
        issues.append(("No programming language",
                        "Recruiter checks language first. State your main language explicitly."))
    elif len(found_langs) == 1:
        praise.append(("Primary language", f"Listed {found_langs[0].title()} -- good."))
    else:
        praise.append(("Multi-language", f"{', '.join(s.title() for s in found_langs[:3])}."))

    dbs = {"postgresql","mysql","mongodb","redis","sqlite","sql",
           "cassandra","elasticsearch","clickhouse","dynamodb"}
    found_dbs = [s for s in hard if s in dbs]
    if not found_dbs:
        issues.append(("No databases",
                        "Almost every job requires SQL or NoSQL. Add at least PostgreSQL or MongoDB."))
        tips.append("Learn PostgreSQL: https://www.postgresqltutorial.com/")
    else:
        praise.append(("DB in stack", f"{', '.join(s.upper() for s in found_dbs[:3])}."))

    devops = {"docker","kubernetes","k8s","terraform","ansible","helm"}
    found_devops = [s for s in hard if s in devops]
    if not found_devops:
        issues.append(("No containerization",
                        "Docker is a baseline skill for any backend/devops role in 2024. "
                        "You are missing 70%+ of job listings."))
        tips.append("Docker quickstart: https://docs.docker.com/get-started/")
    else:
        praise.append(("Containers", f"Has {', '.join(s.title() for s in found_devops[:3])}."))

    clouds = {"aws","gcp","azure","digitalocean","cloudflare"}
    found_cloud = [s for s in hard if s in clouds]
    if not found_cloud:
        issues.append(("No cloud platform",
                        "AWS/GCP/Azure appears in 60%+ of listings. Even basic AWS is a big plus."))
        tips.append("AWS Free Tier: https://aws.amazon.com/free/")
    else:
        praise.append(("Cloud", f"{', '.join(s.upper() for s in found_cloud)}."))

    cicd = {"ci/cd","github actions","gitlab ci","jenkins","circleci"}
    if harshness >= 5 and not any(s in hard for s in cicd):
        issues.append(("No CI/CD",
                        "Deployment automation is standard. GitHub Actions can be learned in a day."))

    market_missing = []
    for skill, demand in sorted(_MARKET_DEMAND.items(), key=lambda x: -x[1]):
        if skill not in hard and demand >= 70:
            market_missing.append(f"{skill.title()} ({demand}%)")
        if len(market_missing) >= 3:
            break
    if market_missing:
        issues.append(("High-demand skills missing",
                        "Missing skills with high market demand: " + ", ".join(market_missing) +
                        ". Prioritize learning these."))

    frontend_s = {"react","vue","angular","html","css","javascript","typescript"}
    backend_s  = {"python","java","go","fastapi","django","flask","spring"}
    data_s     = {"pandas","numpy","scikit-learn","pytorch","tensorflow","spark"}
    has_back  = any(s in hard for s in backend_s)
    has_front = any(s in hard for s in frontend_s)
    has_data  = any(s in hard for s in data_s)

    if has_back and not has_front and not has_data:
        praise.append(("Clear backend focus", "Specialization is visible -- that is a plus."))
    elif has_front and has_back:
        praise.append(("Fullstack stack", "Frontend + Backend present."))
    elif has_data:
        praise.append(("Data/ML stack", "Data science skills present."))

    if not soft:
        issues.append(("No soft skills",
                        "HR and managers check soft skills. Add teamwork, communication, ownership."))
    elif len(soft) >= 3:
        praise.append(("Soft skills present", f"{len(soft)} listed -- sufficient."))

    if not interests and harshness >= 4:
        issues.append(("No interests listed",
                        "Interests show motivation and help match with company culture."))

    n = len(issues)
    if n == 0:
        verdict, text, score = "Top profile", "Almost nothing to criticize -- strong application.", 9
    elif n <= 2:
        verdict, text, score = "Good profile with gaps", "Solid overall, but a few things need fixing before sending.", 7
    elif n <= 4:
        verdict, text, score = "Average profile -- work needed", "Recruiters will notice these gaps. Fix top 2 issues for a big boost.", 5
    elif n <= 6:
        verdict, text, score = "Weak profile -- needs rework", "Many red flags. Auto-screening will filter this out.", 3
    else:
        verdict, text, score = "Critical state", "Seriously overhaul before sending -- will be rejected at first stage.", 1

    if harshness >= 8 and score < 7:
        text += f" (Harshness {harshness}/10 -- no sugarcoating.)"

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
                f"You are a harsh career mentor (harshness {harshness}/10, 10=merciless).\n"
                "Give a deep critical analysis of this IT profile. Analyze:\n"
                "- Stack completeness and relevance (2024 market)\n"
                "- Skill balance (backend/frontend/devops/data)\n"
                "- Market demand alignment\n"
                "- What specifically to add/remove\n\n"
                + json.dumps({k:v for k,v in profile.items() if not k.startswith("_")},
                             ensure_ascii=False, indent=2)
                + "\n\nReturn JSON:\n"
                + '{\n  "verdict": "one-line verdict",\n  "roast_text": "2-3 sentence honest summary",\n  "roast_score": <1-10 where 10=excellent>,\n  "issues": [["issue title", "explanation + what to do"], ...],\n  "praise": [["strength title", "why this is strong"], ...],\n  "tips": ["concrete recommendation 1", "recommendation 2"]\n}\nJSON only. Be specific and helpful.'
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

    tab_pdf, tab_txt = st.tabs(["PDF file", "Text resume"])

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
    st.markdown("### Extracted profile")

    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Hard Skills", len(profile.get("hard_skills", [])))
    with m2: st.metric("Soft Skills", len(profile.get("soft_skills", [])))
    with m3: st.metric("Interests",   len(profile.get("interests",   [])))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Name:** {profile.get('name', '--')}")
        st.markdown("**Hard Skills:**")
        st.markdown(tags(profile.get("hard_skills", [])), unsafe_allow_html=True)
    with c2:
        st.markdown("**Soft Skills:**")
        st.markdown(tags(profile.get("soft_skills", [])), unsafe_allow_html=True)
        st.markdown("**Interests:**")
        st.markdown(tags(profile.get("interests",   [])), unsafe_allow_html=True)

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
**Fallback (no API key):**
- Resume -> 150+ skill dictionary
- Roadmap -> phase template
- Interview -> 100+ question bank
- CV Roast -> market demand rules
- Resources -> static dict
- Telegram -> 11 mock vacancies

**Stack:** Python 3.11 - Streamlit - Pandas
Matplotlib - scikit-learn - Telethon - Docker
""")
    st.divider()
    st.markdown("""
**Quick start:**
```bash
pip install -r requirements.txt
streamlit run app/app.py
```
**Docker:**
```bash
docker compose up --build
```
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("Digital Twin Career Engine")
    st.caption("PDF CV -> Skills -> Telegram Jobs -> CV Roast -> Roadmap -> Interview")

    sidebar_p = sidebar_profile()
    profile   = get_active_profile(sidebar_p)
    profile   = _enrich_profile(profile)

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "CV Upload",
        "CV Roast",
        "Telegram Jobs",
        "Roadmap",
        "Interview",
        "About",
    ])

    with t1: tab_cv_upload()
    with t2: tab_cv_roast(profile)
    with t3: tab_telegram_jobs(profile)
    with t4: tab_roadmap(profile)
    with t5: tab_interview(profile)
    with t6: tab_about()


if __name__ == "__main__":
    main()
