import requests
import json
from datetime import datetime
from bs4 import BeautifulSoup
from bs4 import Tag

import numpy as np

import logging
from sentence_transformers import SentenceTransformer

from typing import Type, TypeVar

from aiutils.ai_client import AirIntelligence, OpenAIAirIntelligence, BaseAirIntelligence

from aiutils.hass_client import HassClient
from aiutils.ai_client import Tools
from aiutils.common import load_config


def scrape_messages(url):
    """
    Fetch messages from public Telegram channel via web scraping
    """
    try:
        # Додаємо User-Agent, щоб Telegram не заблокував нас як бота
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=60)

        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, 'lxml')
        # Telegram іноді змінює класи, 'tgme_widget_message_wrap' — більш стабільний контейнер
        messages = soup.find_all('div', class_='tgme_widget_message')
        result = []

        for content in messages:
            try:
                if not isinstance(content, Tag):
                    continue

                # Шукаємо метадані (час та посилання)
                message_date_obj = content.find(
                    "a", class_="tgme_widget_message_date")
                if not message_date_obj:
                    continue

                if not isinstance(message_date_obj, Tag):
                    continue

                msg_url = message_date_obj.attrs.get("href")
                msg_id = int(str(msg_url).split("/")[-1]) if msg_url else None

                msg_time_obj = message_date_obj.find('time')
                if not msg_time_obj:
                    continue

                if not isinstance(msg_time_obj, Tag):
                    continue

                # ФІКС DATETIME: конвертуємо в ISO string одразу, щоб JSON не падав
                raw_date = datetime.fromisoformat(
                    str(msg_time_obj.attrs.get("datetime")))
                timestamp_str = raw_date.isoformat()

                # Збір тексту повідомлення
                msg_text_obj = content.find(
                    'div', class_='tgme_widget_message_text')
                text = ""

                if msg_text_obj:
                    # Обробка репостів/відповідей (якщо є)
                    reply_obj = content.find(
                        'div', class_='tgme_widget_message_reply')
                    if reply_obj:
                        text += "[Reply/Forward] "

                    # Отримуємо текст, зберігаючи пробіли замість <br>
                    if not isinstance(msg_text_obj, Tag):
                        continue

                    for br in msg_text_obj.find_all("br"):
                        br.replace_with("\n")  # type: ignore
                    text += msg_text_obj.get_text(separator=' ')

                # Очищення тексту від зайвих пробілів
                clean_text = ' '.join(text.split())

                # Додаємо тільки якщо є текст (ігноруємо порожні фото без підпису)
                if clean_text:
                    result.append({
                        "id": msg_id,
                        "url": msg_url,
                        "date": timestamp_str,  # Тепер це String, JSON serializable!
                        "content": clean_text
                    })
            except Exception as e:
                print(f"Error parsing message: {e}")
                continue

        return result
    except Exception as e:
        print(f"Error scraping channel: {e}")
        return []


def cosine_similarity(query_vec, matrix):
    """
    Обчислює схожість між одним вектором та матрицею векторів.
    query_vec: (n,) - вектор питання
    matrix: (m, n) - всі вектори бази
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

    # Якщо текст коротший за розмір чанка, повертаємо його цілим
    if len(words) <= size:
        return [text]

    chunks = []
    # Крок (stride) — це скільки НОВИХ слів ми беремо в кожен наступний чанк
    stride = size - overlap

    for i in range(0, len(words), stride):
        # Беремо "вікно" слів
        chunk_slice = words[i: i + size]
        chunk = " ".join(chunk_slice)
        chunks.append(chunk)

        # Якщо ми вже охопили кінець тексту, виходимо
        if i + size >= len(words):
            break

    return chunks

######################################################


class MyBot:

    def __init__(self):
        self.rag_chunks = []

        self.model: SentenceTransformer = SentenceTransformer('../MiniLM')

        config = load_config()
        self.hass_client = HassClient(base_url=config.get("hass", {}).get(
            "url"), token=config.get("hass", {}).get("token"))
        self.tools = Tools(hass_client=self.hass_client)
        self.initialize_model()

    def initialize_model(self):
        # Категорії, які вимагають інтернет search
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
                "embedding": json.dumps(embedding)
            })

        rooms = self.hass_client.render_template(
            "{{ areas() | map('area_name') | list | tojson }}")
        if not rooms:
            rooms = []

        for room in rooms:
            for device in ['температура', 'вологість', 'освітлення', 'стан']:
                intent_text = f"Я хочу дізнатися показник {device} (датчик {device}) у приміщенні {room}"
                embedding = self.model.encode(intent_text).tolist()
                self.rag_chunks.append({
                    "tool": "hass",
                    "query": f"?room={room}&device={device}",
                    "params": None,
                    "embedding": json.dumps(embedding)
                })

        print("Model initialized and RAG chunks created.")

    def get_best_tool(self, user_text, threshold=0.45):
        query_vec = self.model.encode(user_text)
        embeddings = np.array([json.loads(r["embedding"])
                              for r in self.rag_chunks])
        similarities = cosine_similarity(query_vec, embeddings)

        best_idx = np.argmax(similarities)
        if similarities[best_idx] >= threshold:
            item = self.rag_chunks[best_idx]
            return item

        return None

    T = TypeVar('T', bound=BaseAirIntelligence)

    def ask(self, system_prompt: str, query: str, user_context: dict = {}, ai_class: Type[T] = AirIntelligence) -> str:
        print(f"Ініціалізую AI клас: {ai_class.__name__}")
        bot = ai_class(tools=self.tools, system_prompt=system_prompt)

        local_kb = {}
        if user_context:
            local_kb.update(user_context)
        else:
            # Якщо контексту немає, спробуємо визначити, чи потрібен інструмент
            tool = self.get_best_tool(query)
            if tool:
                call_tool = tool.get("tool")
                call_query = tool.get("query") or query
                call_params = tool.get("params")
                print(
                    f"Викликаю інструмент: {call_tool} з query='{call_query}' та params='{call_params}'")
                method_to_call = getattr(bot.tools, f"tool_{call_tool}")
                result_data = method_to_call(
                    query=call_query, params=call_params)
                local_kb["search_results"] = result_data

        context = json.dumps(local_kb, ensure_ascii=False,
                             indent=2) if local_kb else None
        final_answer = bot.process_request(query, context_data=context)

        print(f"User: {query}")
        print(f"[{ai_class.__name__}] Bot: {final_answer}")
        return final_answer


logging.getLogger("transformers").setLevel(logging.ERROR)

# 'Спальня', 'Зал', 'Дитяча', 'Ванна', 'Кухня', 'Передпокій', 'Задній двір', 'Двір', 'Гараж', 'Вхід'

if __name__ == "__main__":
    bot = MyBot()

    def_system_prompt = "Ти — корисний помічник. Відповідай чітко, стисло, без зайвих слів."

    bot.ask(def_system_prompt, "Хто зараз Президент у USA?",
            ai_class=OpenAIAirIntelligence)

    bot.ask(def_system_prompt, "яка температура у спальні?",
            ai_class=OpenAIAirIntelligence)

    bot.ask(def_system_prompt, "Яка зараз година?",
            ai_class=OpenAIAirIntelligence)
    bot.ask(def_system_prompt, "Дай приклад інструкції `for` в С#.",
            ai_class=OpenAIAirIntelligence)

    bot.ask(def_system_prompt, "Який зара курс USD та EUR до гривні?",
            ai_class=OpenAIAirIntelligence)

    bot.ask(def_system_prompt, "Яка зараз погода у Дніпрі?",
            ai_class=OpenAIAirIntelligence)

    def_system_prompt = "Ти — військовий аналітик. Твоє завдання: проаналізувати повідомлення."
    messages = scrape_messages("https://t.me/s/StrategicaviationT")
    print(f"Scraped {len(messages)} messages.")
    bot.ask(def_system_prompt,
            "Проаналізуй дай імовірність ударів по Україні найближчим часом, базуючись на цих повідомленнях. Відповідай чітко, стисло, без зайвих слів.",
            user_context={"messages": messages},
            ai_class=OpenAIAirIntelligence)
