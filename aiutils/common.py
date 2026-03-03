import os
import yaml
from ddgs import DDGS
import logging

def duckduckgo_search(query, num_results=3):
    results = []
    query = f"{query} site:.ua"

    logging.disable(logging.INFO)

    with DDGS() as ddgs:
        for r in ddgs.text(
            query,
            safesearch='on', timelimit='y', page=1, backend="auto",
            region="ua-uk",
            max_results=num_results
        ):
            results.append(f"[{r['title']}]: {r['body']}")

    logging.disable(logging.NOTSET)

    return results

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Конфіг {config_path} не знайдено!")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

