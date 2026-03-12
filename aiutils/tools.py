import httpx
from .hass_client import HassClient
from .common import duckduckgo_search
import inspect
import logging
import sqlite3
from telegram_tools import scrape_messages

from collections import defaultdict
import statistics
from datetime import datetime
import re


def to_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def tool(description: str, name: str | None = None):
    def decorator(func):
        tool_name = name if name else func.__name__
        func._tool_meta = {
            "name": tool_name,
            "description": description,
        }
        return func
    return decorator


class Intents:
    def __init__(self, hass_client: HassClient, db_path: str) -> None:
        self.hass_client = hass_client
        self.db_path = db_path

    async def intent_tele_chat(self, user_name: str,  **kwargs) -> list | None:
        chat_id = kwargs.get("chat_id")

        if not chat_id:
            return None

        if not user_name:
            return None

        user_messages = await self.get_user_messages(chat_id, user_name)
        if user_messages is not None:
            return user_messages

        return None

    async def intent_tele_channel(self, channel: str,  **kwargs) -> list | None:
        message = scrape_messages(f"https://t.me/s/{channel}")
        if message is not None:
            return message

        return None

    async def intent_weather_dnipro(self, **kwargs) -> dict | str:
        current = await self.hass_client.get_entity("weather.my_weather_station")
        forecast_all = await self.hass_client.get_entity("sensor.home_forecast_hourly")
        forecast = forecast_all["attributes"]["forecast"] if forecast_all else {
        }
        forecast = self.collapse_weather(forecast)
        return {
            "current": current,
            "forecast": forecast
        }
    
    async def intent_search(self, query: str, **kwargs) -> list | None:
        query = query if query else kwargs.get("user_query") or ""
        results = duckduckgo_search(query, num_results=3)
        return results

    async def intent_hass(self, room: str, device: str, **kwargs) -> dict | str:
        if not device:
            return "Помилка: Не вказано параметр 'device' у запиті."

        entity_id = kwargs.get("entity_id")
        if not entity_id:
            entity_id = await self.get_entity_by_room_and_friendly_name(
                room, device)

        if not entity_id:
            return (
                f"Помилка: Не знайдено сутність для кімнати '{room}' "
                f"та пристрою '{device}'. Перевірте назви в Home Assistant."
            )

        if isinstance(entity_id, list):
            states = [await self.hass_client.get_entity(eid) for eid in entity_id]

            values = [to_float(s['state'] if s else 0) for s in states]
            values = [v for v in values if v is not None]

            state_data = {
                "entity_id": "entity_id_avg",
                "state": sum(values) / len(values) if values else 0,
                "attributes": {}
            }
        else:
            state_data = await self.hass_client.get_entity(entity_id)

        if not state_data:
            return f"Сутність '{entity_id}' не знайдена або недоступна в Home Assistant."

        data = {
            "entity_id": state_data.get("entity_id"),
            "state": state_data.get("state"),
            "attributes": state_data.get("attributes", {})
        }

        return data

    async def intent_currency(self, **kwargs) -> list | None:
        default_currencies = ["USD", "EUR"]

        available = [
            "USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "PLN", "CNY", "SEK", "NOK", "DKK", "HUF",
            "CZK", "ILS", "MXN", "NZD", "SGD", "THB", "TRY", "RON", "RSD", "AED", "ZAR", "MYR", "HKD",
            "INR", "KRW", "KZT", "GEL", "XAU", "XAG", "XPT", "XPD", "XDR"
        ]

        user_query = kwargs.get("user_query", "")
        matches = re.findall(r"\b[a-zA-Z]{3,4}\b", user_query)
        codes = [m.upper() for m in matches]
        filtered = [c for c in codes if c in available]

        currencies = filtered or default_currencies

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange?json",
                    timeout=10,
                )
                data = resp.json()

            by_cc = {c["cc"]: c for c in data if c.get("cc") in currencies}

            lines = ["Курс НБУ (UAH за 1 одиницю валюти):"]
            for cc in currencies:
                rate = by_cc.get(cc, {}).get("rate")
                if rate is not None:
                    lines.append(f"{cc}: {rate:.4f} UAH")

            if len(lines) == 1:  # тобто жодної валюти не знайдено
                # f"Помилка: не вдалося отримати {', '.join(currencies)} з відповіді НБУ."
                return None

            return lines

        except Exception:
            return None

    async def get_user_messages(self, chat_id: int, username: str) -> list:
        """
        Отримати повідомлення конкретного користувача за сьогодні

        Приклад використання:
        messages = await self.get_user_messages(chat_id, "Dmitro")
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = '''
                SELECT username, first_name, message_text, timestamp
                FROM messages 
                WHERE chat_id = ? AND username = ?
                AND timestamp >= date('now', 'start of day')
                AND message_text IS NOT NULL
                ORDER BY timestamp
            '''

        cursor.execute(query, (chat_id, username))
        messages = []
        for row in cursor.fetchall():
            username_db, first_name, text, timestamp = row
            display_name = username_db or first_name or "Unknown"
            messages.append(f"[{timestamp}] {display_name}: {text}")

        conn.close()
        return messages

    def collapse_weather(self, data):
        periods = {
            "night": range(0, 6),
            "morning": range(6, 12),
            "day": range(12, 18),
            "evening": range(18, 24),
        }

        grouped = defaultdict(list)

        for entry in data:
            hour = datetime.fromisoformat(entry["datetime"]).hour
            for name, hours in periods.items():
                if hour in hours:
                    grouped[name].append(entry)
                    break

        result = {}
        for name, entries in grouped.items():
            result[name] = {
                "condition": statistics.mode([e["condition"] for e in entries]),
                "temperature_avg": round(statistics.mean([e["temperature"] for e in entries]), 1),
                "wind_speed_avg": round(statistics.mean([e["wind_speed"] for e in entries]), 1),
                "humidity_avg": round(statistics.mean([e["humidity"] for e in entries]), 1),
                "precipitation_sum": round(sum([e["precipitation"] for e in entries]), 1),
            }
        return result

    async def get_entity_by_room_and_friendly_name(self, room_name: str, friendly_name) -> str | None:

        room_name = room_name.lower()
        friendly_name = friendly_name.lower()

        tpl = f"""
{{% set sensors = namespace(items=[]) %}}
{{% for s in states
      | expand
      | selectattr('attributes.friendly_name', 'defined')
      | selectattr(
          'name',
          'search',
          '{friendly_name}',
          ignorecase=True
      )
%}}
  {{% set sensors.items = sensors.items + [{{'name': s.name, 'id': s.entity_id, 'area': area_name(s.entity_id)}}] %}}
{{% endfor %}}

{{{{
sensors.items 
       | selectattr(
          'area',
          'search',
          '{room_name}',
          ignorecase=True
      ) | first | tojson 
}}}}
"""

        area_device = await self.hass_client.render_template(tpl)

        if not area_device:
            return None

        if not isinstance(area_device, dict):
            return None

        return str(area_device.get('id'))


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
                "type": "string",
                # "description": "",
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

    def get_tools_schema(self, exclude: list = []):
        result = []

        for name, data in self._tools.items():
            if name not in exclude:
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

        logging.info(f"Bot: [TOOL]: {name}")

        func = self._tools[name]["func"]
        return await func(**kwargs)


class Tools(Intents):
    def __init__(self, hass_client: HassClient, db_path: str = "./bot_history.db"):
        super().__init__(hass_client, db_path)
        self.db_path = db_path
        if not isinstance(hass_client, HassClient):
            raise TypeError(
                f"hass_client must be HassClient, got {type(hass_client)}")
        self.hass_client = hass_client

        self.registry = ToolRegistry()

        for attr in dir(self):
            method = getattr(self, attr)
            if hasattr(method, "_tool_meta"):
                self.registry.register(method)

    @tool(description="Web search")
    async def tool_search(self, query: str, **kwargs) -> list | None:
        results = await self.intent_search(query=query) or []
        return results
