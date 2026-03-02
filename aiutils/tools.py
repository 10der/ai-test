import json
import requests
from urllib.parse import parse_qs
from .hass_client import HassClient
from .common import duckduckgo_search

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
