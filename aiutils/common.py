import os
import yaml
from ddgs import DDGS
import logging

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
            timelimit="m",
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

