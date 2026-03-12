import os
import yaml
from ddgs import DDGS
import logging
import re
from bs4 import BeautifulSoup
import requests


def duckduckgo_search_(query, num_results=3):
    results = []
    query = (query or "").strip()

    if not query:
        return []

    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    params = {
        "q": query
    }

    try:
        res = requests.get(url, params=params, headers=headers, timeout=10)
        res.raise_for_status()

        soup = BeautifulSoup(res.text, "lxml")

        items = soup.select(".result")

        for item in items[:num_results]:
            title_el = item.select_one(".result__title a")
            body_el = item.select_one(".result__snippet")

            if not title_el:
                continue

            title = title_el.get_text(strip=True)
            href = title_el.get("href", "").strip()
            body = body_el.get_text(strip=True) if body_el else ""

            if href:
                results.append(f"[{title}] ({href}): {body}")
            else:
                results.append(f"[{title}]: {body}")

    except Exception as e:
        logging.error(f"DuckDuckGo search error: {e}")

    return results


def duckduckgo_search(query, num_results=3):
    results = []
    query = (query or "").strip()
    if not query:
        return []

    logging.disable(logging.INFO)

    with DDGS() as ddgs:
        for r in ddgs.text(
            query,
            safesearch="on",
            page=1,
            backend="auto",
            region="wt-wt",
            max_results=num_results
        ):
            title = r.get("title", "").strip()
            body = r.get("body", "").strip()
            href = r.get("href", "").strip()
            if href:
                results.append(f"[{title}] ({href}): {body}")
            else:
                results.append(f"[{title}]: {body}")

    logging.disable(logging.NOTSET)

    return results

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Конфіг {config_path} не знайдено!")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def replace_substring(pattern, repl, string) -> str:
    """Replace string"""
    occurences = re.findall(pattern, string, re.IGNORECASE)
    for occurence in occurences:
        string = string.replace(occurence, repl)
    return string
