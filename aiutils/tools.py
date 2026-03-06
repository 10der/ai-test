import httpx
import json
from .hass_client import HassClient
from .common import duckduckgo_search
import inspect

def tool(name: str, description: str):
    def decorator(func):
        func._tool_meta = {
            "name": name,
            "description": description,
        }
        return func
    return decorator

class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, func):
        meta = func._tool_meta
        name = meta["name"]

        signature = inspect.signature(func)

        properties = {}
        required = []

        for param_name, param in signature.parameters.items():
            if param_name == "self":
                continue

            properties[param_name] = {
                "type": "string",  # можна розширити під int/bool
                # "description": "Query for search"
            }
            required.append(param_name)

        self._tools[name] = {
            "func": func,
            "description": meta["description"],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }

    def get_tools_schema(self):
        result = []

        for name, data in self._tools.items():
            result.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": data["description"],
                    "parameters": data["parameters"],
                }
            })

        return result

    async def execute(self, name: str, **kwargs) -> str:
        if name not in self._tools:
            return f"Error: Unknown tool: {name}"

        func = self._tools[name]["func"]

        # print(f"[TOOL]: {name}")
        return await func(**kwargs)


class Tools:
    def __init__(self, hass_client: HassClient):
        if not isinstance(hass_client, HassClient):
            raise TypeError(
                f"hass_client must be HassClient, got {type(hass_client)}")
        self.hass_client = hass_client

        self.registry = ToolRegistry()

        # авто-реєстрація всіх методів з декоратором
        for attr in dir(self):
            method = getattr(self, attr)
            if hasattr(method, "_tool_meta"):
                self.registry.register(method)

    @tool(
        name="tool_search",
        description="Web search"
    )
    async def tool_general_search(self, query: str) -> str:
        """
        Пошук в інтернеті (DuckDuckGo) для актуальних фактів/новин.

        Формат:
        - **query**: довільний текстовий запит, наприклад: "Хто президент США 2025".
        """
        results = duckduckgo_search(query, num_results=3)
        return "\n".join(results)

    @tool(
        name="tool_currency",
        description="Отримати курси валют НБУ"
    )
    async def tool_currency(self) -> str:
        """
        Отримати курс валют (USD/EUR) від НБУ.

        Формат:
        - **query**: ігнорується (можна передати пустий рядок).
        - Повертає ГОТОВИЙ ТЕКСТ (щоб модель не "перемножувала" і не перекручувала числа).
        """
        try:
            async with httpx.AsyncClient() as client:
                currencies = await client.get(
                    "https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange?json",
                    timeout=10,
                )

                allowed = ['USD', 'EUR']
                by_cc = {c["cc"]: c for c in currencies.json()
                         if c.get("cc") in allowed}
                usd = by_cc.get("USD", {}).get("rate")
                eur = by_cc.get("EUR", {}).get("rate")

                lines = ["Курс НБУ (UAH за 1 одиницю валюти):"]
                if usd is not None:
                    lines.append(f"- USD: {usd:.4f} UAH")
                if eur is not None:
                    lines.append(f"- EUR: {eur:.4f} UAH")

                if usd is None and eur is None:
                    return "Помилка: не вдалося отримати USD/EUR з відповіді НБУ."

                return "\n".join(lines)

        except Exception as e:
            return f"Помилка отримання курсу: {e}"

    @tool(
        name="tool_hass",
        description="Отримати стан/атрибути сутності в Home Assistant за кімнатою та типом пристрою."
    )
    async def tool_hass(self, room: str, device: str) -> str:
        """
        Отримати стан/атрибути сутності в Home Assistant за кімнатою та типом пристрою.

        Формат:
        - **query**: рядок параметрів виду `?room=<кімната>&device=<сутність>`
          Приклад: `?room=спальня&device=температура`
        - **params**: можна передати напряму `entity_id` (тоді query можна не використовувати).

        Повертає JSON: `entity_id`, `state`, `attributes`.
        """
        if not room:
            return "Помилка: Не вказано параметр 'room' у запиті."
        if not device:
            return "Помилка: Не вказано параметр 'device' у запиті."

        if device == "погода":
            entity_id = "weather.my_weather_station"
        else:
            entity_id = await self.hass_client.get_entity_by_room_and_friendly_name(
                room, device)

        if not entity_id:
            return (
                f"Помилка: Не знайдено сутність для кімнати '{room}' "
                f"та пристрою '{device}'. Перевірте назви в Home Assistant."
            )

        state_data = await self.hass_client.get_entity(entity_id)

        if not state_data:
            return f"Сутність '{entity_id}' не знайдена або недоступна в Home Assistant."

        data = {
            "entity_id": state_data.get("entity_id"),
            "state": state_data.get("state"),
            "attributes": state_data.get("attributes", {})
        }

        return json.dumps(data, ensure_ascii=False, indent=2)
