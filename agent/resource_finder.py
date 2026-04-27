"""
Agent Layer — поиск обучающих ресурсов по навыку.

Режимы работы:
  1. Fallback (по умолчанию) — статический словарь SKILL_RESOURCES.
  2. LLM-режим (опционально) — если в .env задан LLM_API_KEY,
     используется Anthropic Claude для генерации актуальных ресурсов.

Future hooks:
  - GitHub Search API (GITHUB_TOKEN в .env) для поиска репозиториев.
  - Парсер курсов с Coursera/Udemy API.
  - Замена статического словаря на векторный поиск по базе ресурсов.
"""

from __future__ import annotations
from utils.llm_client import call_llm_json, is_available

import os
from typing import Optional

# ---------------------------------------------------------------------------
# Статический словарь ресурсов (fallback)
# ---------------------------------------------------------------------------

SKILL_RESOURCES: dict[str, dict] = {
    # Backend
    "python": {
        "docs":    ["https://docs.python.org/3/"],
        "courses": ["https://realpython.com", "https://www.coursera.org/specializations/python"],
        "github":  ["https://github.com/TheAlgorithms/Python"],
        "video":   ["https://www.youtube.com/c/CoreySchafer"],
    },
    "sql": {
        "docs":    ["https://www.postgresql.org/docs/"],
        "courses": ["https://mode.com/sql-tutorial/", "https://www.sqlbolt.com/"],
        "github":  ["https://github.com/bregman-arie/devops-exercises"],
        "video":   ["https://www.youtube.com/watch?v=qw--VYLpxG4"],
    },
    "postgresql": {
        "docs":    ["https://www.postgresql.org/docs/"],
        "courses": ["https://www.postgresqltutorial.com/"],
        "github":  ["https://github.com/postgres/postgres"],
        "video":   ["https://www.youtube.com/watch?v=SpfIwlAYaKk"],
    },
    "rest api": {
        "docs":    ["https://restfulapi.net/"],
        "courses": ["https://www.udemy.com/course/rest-api-flask-and-python/"],
        "github":  ["https://github.com/public-apis/public-apis"],
        "video":   ["https://www.youtube.com/watch?v=SLwpqD8n3d0"],
    },
    "system design": {
        "docs":    ["https://github.com/donnemartin/system-design-primer"],
        "courses": ["https://www.educative.io/courses/grokking-the-system-design-interview"],
        "github":  ["https://github.com/donnemartin/system-design-primer"],
        "video":   ["https://www.youtube.com/c/GauravSensei"],
    },
    "microservices": {
        "docs":    ["https://microservices.io/"],
        "courses": ["https://www.udemy.com/course/microservices-with-spring-boot-and-spring-cloud/"],
        "github":  ["https://github.com/GoogleCloudPlatform/microservices-demo"],
        "video":   ["https://www.youtube.com/watch?v=lTAcCNbJ7KE"],
    },
    # DevOps
    "docker": {
        "docs":    ["https://docs.docker.com/"],
        "courses": ["https://kodekloud.com/courses/docker-for-the-absolute-beginner/"],
        "github":  ["https://github.com/docker/awesome-compose"],
        "video":   ["https://www.youtube.com/watch?v=3c-iBn73dDE"],
    },
    "kubernetes": {
        "docs":    ["https://kubernetes.io/docs/home/"],
        "courses": ["https://kodekloud.com/courses/certified-kubernetes-administrator-cka/"],
        "github":  ["https://github.com/kelseyhightower/kubernetes-the-hard-way"],
        "video":   ["https://www.youtube.com/watch?v=X48VuDVv0do"],
    },
    "aws": {
        "docs":    ["https://docs.aws.amazon.com/"],
        "courses": ["https://aws.amazon.com/training/", "https://acloudguru.com/"],
        "github":  ["https://github.com/donnemartin/awesome-aws"],
        "video":   ["https://www.youtube.com/watch?v=ulprqHHWlng"],
    },
    "linux": {
        "docs":    ["https://man7.org/linux/man-pages/"],
        "courses": ["https://linuxjourney.com/", "https://www.linuxfoundation.org/training/"],
        "github":  ["https://github.com/torvalds/linux"],
        "video":   ["https://www.youtube.com/watch?v=wBp0Rb-ZJak"],
    },
    "bash": {
        "docs":    ["https://www.gnu.org/software/bash/manual/"],
        "courses": ["https://www.learnshell.org/"],
        "github":  ["https://github.com/awesome-lists/awesome-bash"],
        "video":   ["https://www.youtube.com/watch?v=oxuRxtrO2Ag"],
    },
    "ci/cd": {
        "docs":    ["https://docs.github.com/en/actions", "https://docs.gitlab.com/ee/ci/"],
        "courses": ["https://www.udemy.com/course/github-actions-the-complete-guide/"],
        "github":  ["https://github.com/actions/starter-workflows"],
        "video":   ["https://www.youtube.com/watch?v=R8_veQiYBjI"],
    },
    "terraform": {
        "docs":    ["https://developer.hashicorp.com/terraform/docs"],
        "courses": ["https://kodekloud.com/courses/terraform-for-beginners/"],
        "github":  ["https://github.com/hashicorp/terraform"],
        "video":   ["https://www.youtube.com/watch?v=l5k1ai_GBDE"],
    },
    # Data Science / ML
    "pandas": {
        "docs":    ["https://pandas.pydata.org/docs/"],
        "courses": ["https://www.kaggle.com/learn/pandas"],
        "github":  ["https://github.com/pandas-dev/pandas"],
        "video":   ["https://www.youtube.com/watch?v=vmEHCJofslg"],
    },
    "machine learning": {
        "docs":    ["https://scikit-learn.org/stable/"],
        "courses": ["https://www.coursera.org/learn/machine-learning", "https://www.kaggle.com/learn"],
        "github":  ["https://github.com/ageron/handson-ml3"],
        "video":   ["https://www.youtube.com/watch?v=GwIo3gDZCVQ"],
    },
    "statistics": {
        "docs":    ["https://www.khanacademy.org/math/statistics-probability"],
        "courses": ["https://www.coursera.org/learn/bayesian-statistics"],
        "github":  ["https://github.com/rasbt/stat451-machine-learning-fs21"],
        "video":   ["https://www.youtube.com/c/StatQuestwithJoshStarmer"],
    },
    "numpy": {
        "docs":    ["https://numpy.org/doc/"],
        "courses": ["https://www.kaggle.com/learn/intro-to-machine-learning"],
        "github":  ["https://github.com/numpy/numpy"],
        "video":   ["https://www.youtube.com/watch?v=QUT1VHiLmmI"],
    },
    "jupyter": {
        "docs":    ["https://jupyter.org/documentation"],
        "courses": ["https://realpython.com/jupyter-notebook-introduction/"],
        "github":  ["https://github.com/jupyter/notebook"],
        "video":   ["https://www.youtube.com/watch?v=HW29067qVWk"],
    },
    "matplotlib": {
        "docs":    ["https://matplotlib.org/stable/tutorials/"],
        "courses": ["https://www.datacamp.com/courses/introduction-to-matplotlib"],
        "github":  ["https://github.com/matplotlib/matplotlib"],
        "video":   ["https://www.youtube.com/watch?v=UO98lJQ3QGI"],
    },
    # ML Engineering
    "pytorch": {
        "docs":    ["https://pytorch.org/docs/stable/"],
        "courses": ["https://pytorch.org/tutorials/", "https://fast.ai/"],
        "github":  ["https://github.com/pytorch/examples"],
        "video":   ["https://www.youtube.com/watch?v=c36lUUr864M"],
    },
    "mlops": {
        "docs":    ["https://mlflow.org/docs/latest/index.html"],
        "courses": ["https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops"],
        "github":  ["https://github.com/visenger/awesome-mlops"],
        "video":   ["https://www.youtube.com/watch?v=9-vAgXCVAi8"],
    },
    "scikit-learn": {
        "docs":    ["https://scikit-learn.org/stable/user_guide.html"],
        "courses": ["https://www.kaggle.com/learn/intro-to-machine-learning"],
        "github":  ["https://github.com/scikit-learn/scikit-learn"],
        "video":   ["https://www.youtube.com/watch?v=0B5eIE_1vpU"],
    },
    "math": {
        "docs":    ["https://www.khanacademy.org/math"],
        "courses": ["https://www.coursera.org/specializations/mathematics-for-machine-learning-and-data-science"],
        "github":  ["https://github.com/fastai/numerical-linear-algebra"],
        "video":   ["https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"],
    },
    # Frontend
    "javascript": {
        "docs":    ["https://developer.mozilla.org/en-US/docs/Web/JavaScript"],
        "courses": ["https://javascript.info/", "https://www.freecodecamp.org/"],
        "github":  ["https://github.com/getify/You-Dont-Know-JS"],
        "video":   ["https://www.youtube.com/watch?v=W6NZfCO5SIk"],
    },
    "react": {
        "docs":    ["https://react.dev/"],
        "courses": ["https://www.udemy.com/course/react-the-complete-guide-incl-redux/"],
        "github":  ["https://github.com/enaqx/awesome-react"],
        "video":   ["https://www.youtube.com/watch?v=bMknfKXIFA8"],
    },
    "html": {
        "docs":    ["https://developer.mozilla.org/en-US/docs/Web/HTML"],
        "courses": ["https://www.freecodecamp.org/learn/2022/responsive-web-design/"],
        "github":  ["https://github.com/diegocard/awesome-html5"],
        "video":   ["https://www.youtube.com/watch?v=pQN-pnXPaVg"],
    },
    "css": {
        "docs":    ["https://developer.mozilla.org/en-US/docs/Web/CSS"],
        "courses": ["https://www.freecodecamp.org/learn/2022/responsive-web-design/"],
        "github":  ["https://github.com/awesome-css-group/awesome-css"],
        "video":   ["https://www.youtube.com/watch?v=1Rs2ND1ryYc"],
    },
    "typescript": {
        "docs":    ["https://www.typescriptlang.org/docs/"],
        "courses": ["https://www.udemy.com/course/understanding-typescript/"],
        "github":  ["https://github.com/dzharii/awesome-typescript"],
        "video":   ["https://www.youtube.com/watch?v=BwuLxPH8IDs"],
    },
    "figma": {
        "docs":    ["https://help.figma.com/"],
        "courses": ["https://www.figma.com/resources/learn-design/"],
        "github":  ["https://github.com/thomas-lowry/figma-plugins-on-github"],
        "video":   ["https://www.youtube.com/watch?v=FTFaQWZBqQ8"],
    },
    "git": {
        "docs":    ["https://git-scm.com/doc"],
        "courses": ["https://learngitbranching.js.org/"],
        "github":  ["https://github.com/progit/progit2"],
        "video":   ["https://www.youtube.com/watch?v=RGOj5yH7evk"],
    },
}

# Ресурс по умолчанию (если навык не найден в словаре)
_DEFAULT_RESOURCE = {
    "docs":    ["https://google.com/search?q={skill}+documentation"],
    "courses": ["https://www.coursera.org/search?query={skill}",
                "https://www.udemy.com/courses/search/?q={skill}"],
    "github":  ["https://github.com/search?q={skill}+awesome"],
    "video":   ["https://www.youtube.com/results?search_query={skill}+tutorial"],
}


# ---------------------------------------------------------------------------
# Основная функция агента
# ---------------------------------------------------------------------------

def find_resources(skill: str, use_llm: bool = False) -> dict:
    """
    Возвращает ресурсы для изучения указанного навыка.

    Args:
        skill:   название навыка (например, "Kubernetes")
        use_llm: если True и задан LLM_API_KEY в .env — использует LLM

    Returns:
        {
            "skill":   str,
            "source":  "static" | "llm",
            "docs":    list[str],
            "courses": list[str],
            "github":  list[str],
            "video":   list[str],
        }
    """
    skill_key = skill.strip().lower()

    # LLM-режим (если доступен)
    if use_llm or os.getenv("LLM_API_KEY"):
        try:
            resources = _find_resources_llm(skill)
            resources["source"] = "llm"
            return resources
        except Exception as e:
            print(f"[Agent] LLM недоступен ({e}), переключаюсь на fallback.")

    # Статический fallback
    data = SKILL_RESOURCES.get(skill_key)
    if data:
        return {"skill": skill, "source": "static", **data}

    # Если навык не в словаре — шаблонные ссылки
    encoded = skill.replace(" ", "+")
    result = {}
    for key, urls in _DEFAULT_RESOURCE.items():
        result[key] = [u.replace("{skill}", encoded) for u in urls]
    return {"skill": skill, "source": "template", **result}


def find_resources_batch(skills: list[str], use_llm: bool = False) -> list[dict]:
    """Возвращает ресурсы для списка навыков."""
    return [find_resources(s, use_llm=use_llm) for s in skills]


# ---------------------------------------------------------------------------
# LLM hook (Future — подключить при наличии API-ключа)
# ---------------------------------------------------------------------------

def _find_resources_llm(skill: str) -> dict:
    """
    Future hook: запрашивает LLM для подбора актуальных ресурсов.

    Проверяет переменные окружения:
      LLM_PROVIDER  — "anthropic" (default) | "openai"
      LLM_API_KEY   — ключ API

    Raise: Exception если ключ не задан или API недоступен.
    """
    api_key  = os.getenv("LLM_API_KEY")
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if not api_key:
        raise ValueError("LLM_API_KEY не задан в .env")

    prompt = f"""
    Для навыка "{skill}" найди лучшие обучающие ресурсы.
    Верни JSON со структурой:
    {{
      "docs":    [list of 1-2 official documentation URLs],
      "courses": [list of 1-2 course URLs],
      "github":  [list of 1-2 GitHub repo URLs],
      "video":   [list of 1 YouTube tutorial URL]
    }}
    Только JSON, без пояснений.
    """

    data = call_llm_json(prompt, max_tokens=512)
    return {"skill": skill, **data}


# ---------------------------------------------------------------------------
# CLI-тест
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_skills = ["Kubernetes", "PyTorch", "System Design", "Figma"]
    for skill in test_skills:
        r = find_resources(skill)
        print(f"\n🔍 {r['skill']} [{r['source']}]")
        print(f"  📄 Docs:    {r['docs'][0]}")
        print(f"  🎓 Courses: {r['courses'][0]}")
        print(f"  💻 GitHub:  {r['github'][0]}")
        print(f"  🎬 Video:   {r['video'][0]}")
