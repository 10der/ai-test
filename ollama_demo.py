import asyncio
import requests
import json
import time
from datetime import datetime
from bs4 import BeautifulSoup
from bs4 import Tag

import numpy as np

import logging
import sys
from sentence_transformers import SentenceTransformer

from typing import Type

from aiutils.ai_client import T, AirIntelligence, OpenAIAirIntelligence
from aiutils.common import load_config
from aiutils.hass_client import HassClient
from aiutils.tools import Tools

from wiki_ua_alerts import calculate_next_strike, wiki_to_csv


class PrintToLogger:
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            logging.info(line.strip())

    def flush(self):
        pass


def print_section(title: str) -> None:
    """
    Гарне ASCII-оформлення секції для демо в консолі.
    """
    bar = "=" * 80
    centered = f" {title} ".center(80, "=")
    print(f"\n{bar}\n{centered}\n{bar}\n")


def scrape_messages(url: str, delay: float = 1.0) -> list[dict]:
    """
    Fetch messages from public Telegram channel via web scraping.
    """
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/91.0.4472.124 Safari/537.36'
        )
    }

    max_retries = 3
    response = None
    for attempt in range(1, max_retries + 1):
        try:
            time.sleep(delay)  # rate limiting
            response = requests.get(url, headers=headers, timeout=60)

            if response.status_code != 200:
                print(
                    f"[scrape] HTTP {response.status_code} on attempt {attempt}")
                if attempt < max_retries:
                    time.sleep(delay * attempt)
                    continue
                return []

            break  # успішно отримали відповідь
        except requests.RequestException as e:
            print(f"[scrape] Request error on attempt {attempt}: {e}")
            if attempt < max_retries:
                time.sleep(delay * attempt)
            else:
                return []

    if not response:
        return []

    soup = BeautifulSoup(response.text, 'lxml')
    messages = soup.find_all('div', class_='tgme_widget_message')
    result = []

    for content in messages:
        try:
            if not isinstance(content, Tag):
                continue

            message_date_obj = content.find(
                "a", class_="tgme_widget_message_date")
            if not message_date_obj or not isinstance(message_date_obj, Tag):
                continue

            msg_url = message_date_obj.attrs.get("href")
            msg_id = int(str(msg_url).split("/")[-1]) if msg_url else None

            msg_time_obj = message_date_obj.find('time')
            if not msg_time_obj or not isinstance(msg_time_obj, Tag):
                continue

            raw_date = datetime.fromisoformat(
                str(msg_time_obj.attrs.get("datetime")))
            timestamp_str = raw_date.isoformat()

            msg_text_obj = content.find(
                'div', class_='tgme_widget_message_text')
            text = ""

            if msg_text_obj and isinstance(msg_text_obj, Tag):
                reply_obj = content.find(
                    'div', class_='tgme_widget_message_reply')
                if reply_obj:
                    text += "[Reply/Forward] "

                for br in msg_text_obj.find_all("br"):
                    br.replace_with("\n")  # type: ignore
                text += msg_text_obj.get_text(separator=' ')

            clean_text = ' '.join(text.split())
            if clean_text:
                result.append({
                    "id": msg_id,
                    "url": msg_url,
                    "date": timestamp_str,
                    "content": clean_text
                })
        except Exception as e:
            print(f"Error parsing message: {e}")
            continue

    return result


def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Обчислює cosine similarity між одним вектором та матрицею векторів.
    query_vec: (n,) — вектор питання
    matrix: (m, n) — всі вектори бази
    """
    dot_product = np.dot(matrix, query_vec)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query_vec)
    return dot_product / (matrix_norms * query_norm + 1e-10)


def chunk_text(text: str, size: int = 150, overlap: int = 25) -> list[str]:
    """
    Розбиває текст на чанки по словах із перекриттям.
    size: кількість слів у чанку.
    overlap: кількість слів, що повторюються.
    """
    words = text.split()

    if len(words) <= size:
        return [text]

    chunks = []
    stride = size - overlap

    for i in range(0, len(words), stride):
        chunk_slice = words[i: i + size]
        chunks.append(" ".join(chunk_slice))
        if i + size >= len(words):
            break

    return chunks


class MyBot:

    def __init__(self):
        self.rag_chunks: list[dict] = []
        self.model: SentenceTransformer = SentenceTransformer("../MiniLM")

        config = load_config()
        self.hass_client = HassClient(
            base_url=config.get("hass", {}).get("url"),
            token=config.get("hass", {}).get("token"),
        )
        self.tools = Tools(hass_client=self.hass_client)
        self._initialize_rag()

    def _initialize_rag(self) -> None:
        """Наповнює RAG-чанки інтентами та сутностями з Home Assistant."""
        intent_data = [
            ("погода дощ прогноз", "hass", "weather.my_weather_station"),
            ("курс валют долар євро гривня", "currency", None),
            ("новини події що сталось", "general_search", None),
            ("ціна вартість скільки коштує", "general_search", None),
        ]

        for text, tool_id, entity_id in intent_data:
            embedding = self.model.encode(text).tolist()
            self.rag_chunks.append({
                "tool": tool_id,
                "query": None,
                "params": entity_id,
                "embedding": json.dumps(embedding),
            })

        rooms = self.hass_client.render_template(
            "{{ areas() | map('area_name') | list | tojson }}"
        )

        if not rooms:
            rooms = []

        for room in rooms:
            for device in ['температура', 'вологість', 'освітлення', 'стан']:
                intent_text = (
                    f"Я хочу дізнатися показник {device} "
                    f"(датчик {device}) у приміщенні {room}"
                )
                embedding = self.model.encode(intent_text).tolist()
                self.rag_chunks.append({
                    "tool": "hass",
                    "query": f"?room={room}&device={device}",
                    "params": None,
                    "embedding": json.dumps(embedding),
                })

        print("[MyBot] Модель MiniLM завантажено, побудовано RAG-індекс інтентів.")

    def _get_best_tool(self, user_text: str, threshold: float = 0.45) -> dict | None:
        query_vec = self.model.encode(user_text, convert_to_numpy=True)
        embeddings = np.array([json.loads(r["embedding"])
                              for r in self.rag_chunks])
        similarities = cosine_similarity(query_vec, embeddings)

        best_idx = int(np.argmax(similarities))
        if similarities[best_idx] >= threshold:
            return self.rag_chunks[best_idx]

        return None

    async def ask(
        self,
        system_prompt: str,
        query: str,
        user_context: dict | None = None,
        ai_class: Type[T] = AirIntelligence,
    ) -> str:
        print(f"Ініціалізую AI клас: {ai_class.__name__}")
        bot = ai_class(tools=self.tools, system_prompt=system_prompt)

        local_kb: dict = {}

        if user_context:
            local_kb.update(user_context)
        else:
            tool = self._get_best_tool(query)
            if tool:
                call_tool = tool.get("tool")
                call_query = tool.get("query") or query
                call_params = tool.get("params")
                print(
                    f"Викликаю інструмент: {call_tool} "
                    f"з query='{call_query}' та params='{call_params}'"
                )
                method_to_call = getattr(bot.tools, f"tool_{call_tool}")
                result_data = method_to_call(
                    query=call_query, params=call_params)
                local_kb["search_results"] = result_data

        context = json.dumps(local_kb, ensure_ascii=False,
                             indent=2) if local_kb else None
        final_answer = await bot.process_request(query, context_data=context)

        print(f"User: {query}")
        print(f"[{ai_class.__name__}] Bot: {final_answer}")
        return final_answer


async def run_demo(bot: MyBot) -> None:
    """
    Основний сценарій демо:
    """

    print_section("ДЕМО — Спілкування з ботом)")
    def_system_prompt = (
        "Ти — корисний помічник. Відповідай чітко, стисло, без зайвих слів."
    )

    await bot.ask(def_system_prompt, "Хто зараз Президент у USA?",
            ai_class=AirIntelligence)

    await bot.ask(def_system_prompt, "яка температура у спальні?",
            ai_class=AirIntelligence)

    await bot.ask(def_system_prompt, "Яка зараз погода у Дніпрі?",
            ai_class=AirIntelligence)

    await bot.ask(def_system_prompt, "Яка зараз година?",
            ai_class=AirIntelligence)

    await bot.ask(def_system_prompt, "Який зараз курс USD та EUR до гривні?",
            ai_class=AirIntelligence)

    await bot.ask(def_system_prompt, "Дай приклад інструкції `for` в C#.",
            ai_class=AirIntelligence)

    print_section("КРОК 1 — Аналіз історичних даних (Wikipedia)")
    wiki_to_csv("https://uk.wikipedia.org/wiki/%D0%9F%D0%B5%D1%80%D0%B5%D0%BB%D1%96%D0%BA_%D1%80%D0%B0%D0%BA%D0%B5%D1%82%D0%BD%D0%B8%D1%85_%D1%83%D0%B4%D0%B0%D1%80%D1%96%D0%B2_%D0%BF%D1%96%D0%B4_%D1%87%D0%B0%D1%81_%D1%80%D0%BE%D1%81%D1%96%D0%B9%D1%81%D1%8C%D0%BA%D0%BE%D0%B3%D0%BE_%D0%B2%D1%82%D0%BE%D1%80%D0%B3%D0%BD%D0%B5%D0%BD%D0%BD%D1%8F_(%D0%B7%D0%B8%D0%BC%D0%B0_2025/2026)")
    math_report = calculate_next_strike()
    print("Математична оцінка наступного удару:")
    print(math_report)

    print_section("КРОК 2 — Збір свіжих повідомлень з моніторингових каналів")
    messages = scrape_messages("https://t.me/s/StrategicaviationT")
    print(f"Отримано {len(messages)} повідомлень з Telegram.")

    military_prompt = (
        "Ти — військовий аналітик. Твоє завдання: проаналізувати повідомлення."
    )

    print_section("КРОК 3 — Інференс LLM з урахуванням всього контексту")
    await bot.ask(
        military_prompt,
        f"""
        Ось математичний розрахунок: {math_report}
        Ось останні дані з моніторингових каналів - КОНТЕКСТ / messages.
        Проаналізуй ризики. Чи є ознаки підготовки, які математика не враховує?
        Надай коротку оцінку загрози (Low/Medium/High/Critical).
        """,
        user_context={"messages": messages},
        ai_class=OpenAIAirIntelligence,
    )


# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',  # Час + повідомлення
    datefmt='%Y-%m-%d %H:%M:%S'          # Формат часу
)

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

logging.info("Система запущена")
logging.info("Поточний стан: OK")

# Перенаправляємо стандартний вивід
sys.stdout = PrintToLogger()

async def main():
    bot = MyBot()
    await run_demo(bot)

if __name__ == "__main__":
    asyncio.run(main())
