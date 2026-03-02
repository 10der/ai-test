import requests
import json
from .common import load_config, duckduckgo_search

from urllib.parse import parse_qs
from .hass_client import HassClient

from typing import TypeVar
from .base import BaseAirIntelligence

T = TypeVar('T', bound=BaseAirIntelligence)


class Tools:
    def __init__(self, hass_client: HassClient):
        if not isinstance(hass_client, HassClient):
            raise TypeError(
                f"hass_client must be HassClient, got {type(hass_client)}")
        self.hass_client = hass_client

    @property
    def tool_names(self) -> list[str]:
        """Повертає список імен методів, що починаються на 'tool_'"""
        return [
            name for name in dir(self)
            if name != "tool_names" and name.startswith("tool_") and callable(getattr(self, name))
        ]

    def tool_general_search(self, query: str, params=None) -> str:
        """General web search using DuckDuckGo"""
        results = duckduckgo_search(query, num_results=5)
        return "\n".join(results)

    def tool_currency(self, query: str, params=None) -> str:
        """Get currency rate"""
        currencies = requests.get(
            "https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange?json",
            timeout=10,
        ).json()
        r = [c for c in currencies if c["cc"] in ['USD', 'EUR']]
        return json.dumps(r, ensure_ascii=False, indent=2)

    def tool_hass(self, query: str, params=None) -> str:
        if query and not params:
            clean_params = query.lstrip('?')
            parsed = parse_qs(clean_params)

            room = parsed.get('room', [None])[0]
            device = parsed.get('device', [None])[0]

            if not room:
                return "Помилка: Не вказано параметр 'room' у запиті."
            if not device:
                return "Помилка: Не вказано параметр 'device' у запиті."

            params = self.hass_client.get_entity_by_room_and_friendly_name(
                room, device)
            if not params:
                return (
                    f"Помилка: Не знайдено сутність для кімнати '{room}' "
                    f"та пристрою '{device}'. Перевірте назви в Home Assistant."
                )
        if not params:
            return "Помилка: Невірний формат запиту для інструменту 'hass'."

        state_data = self.hass_client.get_entity(params)

        if not state_data:
            return f"Сутність '{params}' не знайдена або недоступна в Home Assistant."

        data = {
            "entity_id": state_data.get("entity_id"),
            "state": state_data.get("state"),
            "attributes": state_data.get("attributes", {})
        }

        return json.dumps(data, ensure_ascii=False, indent=2)


class AirIntelligence(BaseAirIntelligence):
    def __init__(self, tools: Tools, system_prompt: str | None = None):
        super().__init__(tools=tools, system_prompt=system_prompt)
        config = load_config()
        self.model = config.get("ollama", {}).get("model")
        self.url = config.get("ollama", {}).get("url")

    def ask_ai(self, messages: list, temperature: float = 0.1) -> str:
        """Спілкування з Ollama"""
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
        response.raise_for_status()
        return response.json()['message']['content']


class OpenAIAirIntelligence(BaseAirIntelligence):
    def __init__(self, tools: Tools, system_prompt: str | None = None):
        super().__init__(tools=tools, system_prompt=system_prompt)
        config = load_config()
        self.model = config.get("openai", {}).get("model")
        self.url = "https://api.openai.com/v1/chat/completions"
        self.api_key = config.get("openai", {}).get("api_key")

    def ask_ai(self, messages: list, temperature: float = 0.1) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1000
        }

        try:
            response = requests.post(
                self.url, json=payload, headers=headers, timeout=30
            )

            if response.status_code != 200:
                print(f"Debug OpenAI Raw Response: {response.text}")

            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']

        except Exception as e:
            return f"OpenAI Error: {str(e)}"
