import requests
from datetime import datetime
import json
from .common import load_config, duckduckgo_search

from urllib.parse import parse_qs
from .hass_client import HassClient

class Tools:
    def __init__(self, hass_client: HassClient):
        self.hass_client = hass_client

    @property
    def tool_names(self):
        """Безпечно повертає список імен методів, що починаються на 'tool_'"""
        names = []
        for name in dir(self):
            if name == "tool_names":
                continue
                
            if name.startswith("tool_"):
                attr = getattr(self, name)
                if callable(attr):
                    names.append(name)
        return names

    def tool_general_search(self, query, params=None):
        """General web search using DuckDuckGo"""
        results = duckduckgo_search(query, num_results=5)
        return "\n".join(results)

    def tool_currency(self, query, params=None):
        """Get currency rate"""

        currencies = requests.get(
            "https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange?json",
            timeout=1000,
        ).json()
        r = [c for c in currencies if c["cc"] in ['USD', 'EUR']]
        return json.dumps(r, ensure_ascii=False, indent=2)

    def tool_hass(self, query, params):
        if query and not params:
            # 1. Прибираємо знак питання, якщо він є
            clean_params = query.lstrip('?')
            
            # 2. Розбираємо рядок у словник
            # parse_qs повертає списки для кожного ключа: {'room': ['спальня'], 'device': ['температура']}
            parsed = parse_qs(clean_params)
            
            # 3. Витягуємо значення (беремо перший елемент списку або None)
            room = parsed.get('room', [None])[0]
            device = parsed.get('device', [None])[0]

            if not room or not device:
                return "Помилка: Не вказано room або device."

            params = self.hass_client.get_entity_by_room_and_friendly_name(room, device)
            if not params:
                return f"Помилка: Не знайдено сутність для кімнати '{room}' та пристрою '{device}'."
        
        # params тут — це sensor.home_temperature
        state_data = self.hass_client.get_entity(params)

        if not state_data:
            return f"Сутність {params} не знайдена або недоступна в Home Assistant."

        data = {
            "entity_id": state_data.get("entity_id"),
            "state": state_data.get("state"),
            "attributes": state_data.get("attributes", {})
        }

        return json.dumps(data, ensure_ascii=False, indent=2)

class BaseAirIntelligence:
    def __init__(self, tools:Tools, system_prompt=None):
        self.system_prompt = system_prompt if system_prompt else "Ти — корисний помічник. Відповідай чітко, стисло, без зайвих слів."
        self.tools = tools  # Ініціалізуємо інструменти з переданого класу

    def ask_ai(self, messages, temperature=0.1) -> str:
        return "This method should be implemented in subclasses."

    def process_request(self, user_text, context_data=None, system_prompt_override=None):
        if not context_data:
            # Промпт для РОУТЕРА (перший крок)
            sys_message = self.get_router_prompt(
                system_prompt_override=system_prompt_override)
        else:
            # Промпт для АНАЛІТИКА (другий крок)
            # Ми ПРИБИРАЄМО правила про SEARCH і CUTOFF,
            # бо дані вже перед очима моделі!
            current_date = datetime.now().strftime("%d.%m.%Y")
            current_time = datetime.now().strftime("%H:%M")
            sys_message = (
                f"{self.system_prompt}\n"
                f"CURRENT_DATE: {current_date}\n"
                f"CURRENT_TIME: {current_time}\n"
                f"ВИКОРИСТОВУЙ ТІЛЬКИ ЦІ ДАНІ - КОНТЕКСТ: \n"
                f"{context_data}\n\n"
            )

        # 1. Формуємо початковий промпт (Роутер)
        messages = [
            {"role": "system", "content": sys_message},
            {"role": "user", "content": user_text}
        ]

        # Додаємо контекст, якщо це вже "друге коло"
        if context_data:
            messages.insert(
                1, {"role": "system", "content": f"ADDITIONAL CONTEXT: {context_data}"})

        # Запит до AI
        print(
            f"--- Thinking... (Context: {'Yes' if context_data else 'No'}) ---")
        ai_response = self.ask_ai(messages)    

        if any(ai_response.startswith(name) for name in self.tools.tool_names) and not context_data:
            print(f"--- AI requested tool usage: {ai_response} ---")
            method_to_call = next(name for name in self.tools.tool_names if ai_response.startswith(name))
            query = ai_response.replace(method_to_call, "").strip()
            formatted_results = getattr(self.tools, method_to_call)(query, None)
            return self.process_request(user_text, context_data=formatted_results)

        # 3. Фінальний результат
        return ai_response

    def get_router_prompt(self, system_prompt_override=None):
        current_date = datetime.now().strftime("%d.%m.%Y")
        current_time = datetime.now().strftime("%H:%M")

        system_prompt = f"""
CURRENT_DATE: {current_date}
CURRENT_TIME: {current_time}
KNOWLEDGE_CUTOFF: 2023-10

RULES:
0. ЗАВЖДИ ВИКОРИСТОВУЙ ДАТУ CURRENT_DATE ТА ЧАС CURRENT_TIME У СВОЇХ ВІДПОВІДЯХ.
1. If the user's query refers to events, people, or facts occurring AFTER 2023-10, you MUST NOT provide a text answer.
2. For all post-2023 queries, your output must be EXACTLY in this format: tool_general_search: [query]
3. ABSOLUTELY FORBIDDEN: Do not apologize. Do not explain your knowledge cutoff.
4. If you are unsure about the current status of an entity as of {current_date}, use the tool_general_search command.
5. ЗАГАЛЬНІ ЗНАННЯ (математика, програмування, граматика, історія до 2023) — не потребують пошуку. Відповідай одразу.
6. Якщо користувач запитує про стан кімнати, ВИКОРИСТОВУЙ ІНСТРУМЕНТ! Визнач локацію (room). Визнач сутність (device). Виклич tool_hass?room=room&device=device. ПРИКЛАД: "яка температура в спальні?" -> room=спальня, device=температура -> tool_hass?room=спальня&device=температура

"""

        return (system_prompt_override if system_prompt_override else self.system_prompt) + "\n" + system_prompt


class AirIntelligence(BaseAirIntelligence):
    def __init__(self, tools:Tools, system_prompt=None):
        super().__init__(tools=tools, system_prompt=system_prompt)
        config = load_config()
        self.model = config.get("ollama", {}).get("model")
        self.url = config.get("ollama", {}).get("url")

    def ask_ai(self, messages, temperature=0.1):
        """Базовий метод для спілкування з Ollama"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_thread": 8
            }
        }
        response = requests.post(self.url, json=payload)
        return response.json()['message']['content']


class OpenAIAirIntelligence(BaseAirIntelligence):
    def __init__(self, tools:Tools, system_prompt=None):
        super().__init__(tools=tools, system_prompt=system_prompt)
        config = load_config()
        self.model = config.get("openai", {}).get("model")
        self.url = "https://api.openai.com/v1/chat/completions"
        self.api_key = config.get("openai", {}).get("api_key")

    def ask_ai(self, messages, temperature=0.1):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,  # "gpt-4o-mini"
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1000
        }

        try:
            response = requests.post(
                self.url, json=payload, headers=headers, timeout=30)

            # Якщо вилетіла помилка, глянемо, що саме каже OpenAI (дуже корисно для дебагу)
            if response.status_code != 200:
                print(f"Debug OpenAI Raw Response: {response.text}")

            response.raise_for_status()

            # 3. Структура відповіді для Chat API саме така:
            return response.json()['choices'][0]['message']['content']

        except Exception as e:
            return f"OpenAI Error: {str(e)}"
