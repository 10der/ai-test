import httpx
import json
from urllib.parse import parse_qs
from dataclasses import dataclass
from typing import Callable, Any
from .hass_client import HassClient
from .common import duckduckgo_search


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_format: str
    when_to_use: str
    examples: tuple[str, ...] = ()
    is_fallback: bool = False


def tool_spec(
    *,
    description: str,
    input_format: str,
    when_to_use: str,
    examples: list[str] | None = None,
    is_fallback: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Декоратор для опису інструмента у вигляді структури.

    Це джерело правди для роутер-промпта: опис/формат/коли використовувати/приклади
    беруться саме звідси, а не з хардкоду в `_get_router_prompt`.
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        spec = ToolSpec(
            name=fn.__name__,
            description=description.strip(),
            input_format=input_format.strip(),
            when_to_use=when_to_use.strip(),
            examples=tuple((examples or ())),
            is_fallback=is_fallback,
        )
        setattr(fn, "_tool_spec", spec)
        return fn

    return decorator


class Tools:
    def __init__(self, hass_client: HassClient):
        if not isinstance(hass_client, HassClient):
            raise TypeError(
                f"hass_client must be HassClient, got {type(hass_client)}")
        self.hass_client = hass_client

    @property
    def tool_names(self) -> list[str]:
        """
        Повертає список імен *методів* інструментів, що починаються на `tool_`.

        Важливо: тут НЕ можна робити `getattr(self, ...)` на всьому `dir(self)`,
        бо це тригерить properties (наприклад `tool_specs`) і може викликати рекурсію.
        """
        names: list[str] = []
        cls = type(self)
        for name in dir(cls):
            if not name.startswith("tool_"):
                continue
            if name in {"tool_names", "tool_specs"}:
                continue
            attr = getattr(cls, name, None)
            if callable(attr):
                names.append(name)
        return sorted(names)

    @property
    def tool_specs(self) -> list[ToolSpec]:
        """
        Повертає специфікації інструментів (ToolSpec) для роутера.
        """
        specs: list[ToolSpec] = []
        for name in self.tool_names:
            fn = getattr(self, name, None)
            spec = getattr(fn, "_tool_spec", None)
            if isinstance(spec, ToolSpec):
                specs.append(spec)
        return sorted(specs, key=lambda s: s.name)

    @tool_spec(
        description="Пошук в інтернеті (DuckDuckGo) для актуальних фактів/новин.",
        input_format="query = довільний текстовий запит (1 рядок).",
        when_to_use=(
            "Використовуй, якщо запит стосується фактів/подій/людей після 2023-10 "
            "або ти не впевнений в актуальності інформації."
        ),
        examples=[
            "tool_general_search: Хто виграв оскара в цьому році?",
            "tool_general_search: Що цікавого у світі?",
        ],
        is_fallback=True,
    )
    async def tool_general_search(self, query: str, params=None) -> str:
        """
        Пошук в інтернеті (DuckDuckGo) для актуальних фактів/новин.

        Формат:
        - **query**: довільний текстовий запит, наприклад: "Хто президент США 2025".
        """
        results = duckduckgo_search(query, num_results=3)
        return "\n".join(results)

    @tool_spec(
        description="Отримати курс валют (USD/EUR) від НБУ.",
        input_format="query = може бути порожній рядок.",
        when_to_use="Коли користувач просить курс валют до гривні.",
        examples=[
            "tool_currency:",
        ],
    )
    async def tool_currency(self, query: str, params=None) -> str:
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

    @tool_spec(
        description="Отримати стан/атрибути сутності в Home Assistant.",
        input_format=(
            "query = `?room=<кімната>&device=<сутність>`\n"
            "приклад: `?room=спальня&device=температура`"
        ),
        when_to_use=(
            "Коли користувач питає про погоду/стан/клімат/температу, вологіть або сенсори/пристрої у будинку або кімнаті. "
            "Ти маєш сам визначити room і device з тексту користувача."
        ),
        examples=[
            "tool_hass: ?room=спальня&device=температура",
            "tool_hass: ?room=кухня&device=вологість",
        ],
    )
    async def tool_hass(self, query: str, params=None) -> str:
        """
        Отримати стан/атрибути сутності в Home Assistant за кімнатою та типом пристрою.

        Формат:
        - **query**: рядок параметрів виду `?room=<кімната>&device=<сутність>`
          Приклад: `?room=спальня&device=температура`
        - **params**: можна передати напряму `entity_id` (тоді query можна не використовувати).

        Повертає JSON: `entity_id`, `state`, `attributes`.
        """
        if query and not params:
            clean_params = query.lstrip('?')
            parsed = parse_qs(clean_params)

            room = parsed.get('room', [None])[0]
            device = parsed.get('device', [None])[0]

            if not room:
                return "Помилка: Не вказано параметр 'room' у запиті."
            if not device:
                return "Помилка: Не вказано параметр 'device' у запиті."

            if device == "погода":
                params = "weather.my_weather_station"
            else:
                params = await self.hass_client.get_entity_by_room_and_friendly_name(
                    room, device)
                if not params:
                    return (
                        f"Помилка: Не знайдено сутність для кімнати '{room}' "
                        f"та пристрою '{device}'. Перевірте назви в Home Assistant."
                    )
        if not params:
            return "Помилка: Невірний формат запиту для інструменту 'hass'."

        state_data = await self.hass_client.get_entity(params)

        if not state_data:
            return f"Сутність '{params}' не знайдена або недоступна в Home Assistant."

        data = {
            "entity_id": state_data.get("entity_id"),
            "state": state_data.get("state"),
            "attributes": state_data.get("attributes", {})
        }

        return json.dumps(data, ensure_ascii=False, indent=2)
