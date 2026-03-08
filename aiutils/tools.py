import httpx
import json
from .hass_client import HassClient
from .common import duckduckgo_search
import inspect
import logging

def to_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

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

        logging.debug(f"Bot: [TOOL]: {name}")

        func = self._tools[name]["func"]
        return await func(**kwargs)


class Tools:
    def __init__(self, hass_client: HassClient):
        if not isinstance(hass_client, HassClient):
            raise TypeError(
                f"hass_client must be HassClient, got {type(hass_client)}")
        self.hass_client = hass_client

        self.registry = ToolRegistry()

        for attr in dir(self):
            method = getattr(self, attr)
            if hasattr(method, "_tool_meta"):
                self.registry.register(method)

    @tool(
        name="tool_search",
        description="Web search"
    )
    async def tool_general_search(self, query: str) -> str:
        results = duckduckgo_search(query, num_results=3)
        return "\n".join(results)

    @tool(
        name="tool_currency",
        description="Отримати курси валют НБУ"
    )
    async def tool_currency(self) -> str:
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
    async def tool_hass(self, room: str, device: str, **kwargs) -> str:
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

        return json.dumps(data, ensure_ascii=False, indent=2)

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
    
    async def get_area_aliases(self) -> dict:
        """Повертає мапу {'Назва кімнати': ['аліас1', 'аліас2']}"""
        areas = await self.hass_client.call_ws_command("config/area_registry/list")
        if not areas:
            return {}
        
        # Створюємо мапу: Name -> List of Aliases
        return {a["name"]: a.get("aliases", []) for a in areas}