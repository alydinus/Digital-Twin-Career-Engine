"""
Agent Layer — Web Searcher (Live Resource Search)

PDF requirement:
  > Grant the agent web-browsing capabilities. The agent must autonomously
  > execute a search query, parse the HTML of the top 3 live results,
  > and extract current, real-world resources.

Modes:
  1. GitHub Search API (if GITHUB_TOKEN set)
  2. DuckDuckGo HTML scrape (no API key needed)
  3. Google search fallback URLs

Public API:
    search_live_resources(skill, max_results=3) -> dict
"""

from __future__ import annotations
import os, re, json
from typing import Optional
from urllib.parse import quote_plus


# ---------------------------------------------------------------------------
# GitHub Search API
# ---------------------------------------------------------------------------

def _search_github(skill: str, max_results: int = 3) -> list[str]:
    """Search GitHub repos using the GitHub Search API."""
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        return []

    try:
        import urllib.request
        query = quote_plus(f"{skill} awesome tutorial")
        url = (f"https://api.github.com/search/repositories"
               f"?q={query}&sort=stars&per_page={max_results}")

        req = urllib.request.Request(url, headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "DigitalTwinCareerEngine/1.0",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [item["html_url"] for item in data.get("items", [])[:max_results]]
    except Exception as e:
        print(f"[WebSearcher] GitHub API error: {e}")
        return []


# ---------------------------------------------------------------------------
# DuckDuckGo HTML scrape
# ---------------------------------------------------------------------------

def _search_duckduckgo(query: str, max_results: int = 3) -> list[dict]:
    """Scrape DuckDuckGo HTML results (no API key needed)."""
    try:
        import urllib.request
        encoded = quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"

        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        results = []
        # Parse result links from DuckDuckGo HTML
        pattern = r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
        matches = re.findall(pattern, html, re.DOTALL)

        for href, title_html in matches[:max_results * 2]:
            # Clean URL (DuckDuckGo wraps URLs)
            if "uddg=" in href:
                url_match = re.search(r'uddg=([^&]+)', href)
                if url_match:
                    from urllib.parse import unquote
                    href = unquote(url_match.group(1))
            # Clean title
            title = re.sub(r'<[^>]+>', '', title_html).strip()
            if href.startswith("http") and title:
                results.append({"url": href, "title": title})
                if len(results) >= max_results:
                    break

        return results
    except Exception as e:
        print(f"[WebSearcher] DuckDuckGo error: {e}")
        return []


# ---------------------------------------------------------------------------
# Parse page content
# ---------------------------------------------------------------------------

def _extract_page_summary(url: str, max_chars: int = 500) -> str:
    """Fetch a URL and extract a text summary."""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        })
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # Extract title
        title_match = re.search(r'<title>(.*?)</title>', html, re.DOTALL)
        title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip() if title_match else ""

        # Extract meta description
        desc_match = re.search(
            r'<meta[^>]*name=["\']description["\'][^>]*content=["\'](.*?)["\']',
            html, re.DOTALL | re.IGNORECASE)
        desc = desc_match.group(1).strip() if desc_match else ""

        # Extract visible text (rough)
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        summary = f"{title}. {desc}" if desc else text[:max_chars]
        return summary[:max_chars]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_live_resources(skill: str, max_results: int = 3) -> dict:
    """
    Search the live web for learning resources about a skill.

    Returns:
        {
            "skill": str,
            "source": "web",
            "results": [{"url": str, "title": str, "summary": str}, ...],
            "github_repos": [str, ...],
            "search_queries": {
                "docs": str, "courses": str, "tutorials": str
            }
        }
    """
    results = {
        "skill": skill,
        "source": "web",
        "results": [],
        "github_repos": [],
        "docs": [],
        "courses": [],
        "github": [],
        "video": [],
    }

    # 1. GitHub repos
    github_urls = _search_github(skill, max_results=2)
    results["github_repos"] = github_urls
    results["github"] = github_urls

    # 2. DuckDuckGo — documentation
    doc_results = _search_duckduckgo(f"{skill} official documentation", max_results=2)
    for r in doc_results:
        results["docs"].append(r["url"])
        results["results"].append({
            "url": r["url"], "title": r["title"],
            "type": "docs", "summary": "",
        })

    # 3. DuckDuckGo — courses/tutorials
    course_results = _search_duckduckgo(
        f"{skill} tutorial course 2024", max_results=2)
    for r in course_results:
        results["courses"].append(r["url"])
        results["results"].append({
            "url": r["url"], "title": r["title"],
            "type": "course", "summary": "",
        })

    # 4. YouTube search URL
    yt_query = quote_plus(f"{skill} tutorial")
    results["video"] = [
        f"https://www.youtube.com/results?search_query={yt_query}"
    ]

    # 5. Optionally fetch page summaries for top 3 results
    for r in results["results"][:3]:
        summary = _extract_page_summary(r["url"])
        if summary:
            r["summary"] = summary[:200]

    return results


def search_and_format(skill: str) -> dict:
    """
    Search live web and return in the same format as resource_finder.py.
    Compatible drop-in for the static dict.
    """
    raw = search_live_resources(skill)
    return {
        "skill": skill,
        "source": "web-live",
        "docs":    raw.get("docs", [])[:2],
        "courses": raw.get("courses", [])[:2],
        "github":  raw.get("github", [])[:2],
        "video":   raw.get("video", [])[:1],
    }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    skill = sys.argv[1] if len(sys.argv) > 1 else "Kubernetes"
    print(f"=== Live Web Search: {skill} ===\n")

    r = search_live_resources(skill)
    print(f"Source: {r['source']}")
    print(f"\nGitHub repos ({len(r['github_repos'])}):")
    for url in r["github_repos"]:
        print(f"  {url}")
    print(f"\nDocs ({len(r['docs'])}):")
    for url in r["docs"]:
        print(f"  {url}")
    print(f"\nCourses ({len(r['courses'])}):")
    for url in r["courses"]:
        print(f"  {url}")
    print(f"\nResults with summaries:")
    for item in r["results"][:3]:
        print(f"  [{item['type']}] {item['title']}")
        if item.get("summary"):
            print(f"        {item['summary'][:100]}...")
