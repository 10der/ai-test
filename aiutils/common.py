import os
import yaml
from ddgs import DDGS  # type: ignore

def duckduckgo_search(query, num_results=3):
    """Безкоштовний пошук через DuckDuckGo"""

    results = []
    query = f"{query} site:.ua"

    with DDGS() as ddgs:  # type: ignore
        for r in ddgs.text(
            query,
            region='ua-uk',
            max_results=num_results
        ):
            results.append(f"[{r['title']}]: {r['body']}")

    return results

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Конфіг {config_path} не знайдено!")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

