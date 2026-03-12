import asyncio
import logging
from typing import Type, Any

import json

from aiutils.ai_client import AirIntelligence, OpenAIAirIntelligence, T
from aiutils.common import load_config
from aiutils.hass_client import HassClient
from aiutils.tools import Tools
from wiki_ua_alerts import calculate_next_strike, wiki_to_csv

from aiutils.intent_classifier import IntentClassifier

from telegram_tools import scrape_messages


class MyBot:

    def __init__(self):
        self.rag_chunks: list[dict] = []
        self.classifier = IntentClassifier(threshold=0.7)

        config = load_config()
        self.hass_client = HassClient(
            base_url=config.get("hass", {}).get("url"),
            token=config.get("hass", {}).get("token"),
        )
        self.tools = Tools(hass_client=self.hass_client)

    async def init_intent_data(self):

        areas = await self.hass_client.call_ws_command("config/area_registry/list")
        if not areas:
            areas = []

        floors = await self.hass_client.call_ws_command("config/floor_registry/list")
        if not floors:
            floors = []

        # Створюємо мапу: Name -> List of Aliases
        rooms = {a["name"]: {"aliases":  a.get(
            "aliases", []), "data": a} for a in areas}

        main_floor_aliases = next(
            (a for a in floors if a["floor_id"] == "main"), None)
        main_floor_data = [a for a in areas if a["floor_id"] == "main"]
        temperature_entity_ids = [s["temperature_entity_id"]
                                  for s in main_floor_data]
        humidity_entity_ids = [s["humidity_entity_id"]
                               for s in main_floor_data]
        rooms['Будинок'] = {"aliases": main_floor_aliases["aliases"] if main_floor_aliases else [], "data": {
            "temperature_entity_id": temperature_entity_ids, "humidity_entity_id": humidity_entity_ids}}

        for room in rooms:
            names = [room] + rooms[room]["aliases"]
            for name in names:
                # Генеруємо СЕМАНТИЧНІ ПРИКЛАДИ
                self.classifier.add_intent(f"яка температура скільки градусів {name}", "intent_hass", {
                                           "room": room, "device": "температура", "entity_id": rooms[room]["data"].get('temperature_entity_id')})
                self.classifier.add_intent(f"яка вологість {name}", "intent_hass", {
                                           "room": room, "device": "вологість", "entity_id": rooms[room]["data"].get('humidity_entity_id')})

        self.classifier.add_intent(
            "курс валют долар євро гривня", "intent_currency", {})

        self.classifier.add_intent("прогноз погоди", "intent_weather_dnipro", {
                                   "room": "", "device": "погода"})

        self.classifier.build_index()

    async def ask(
        self,
        system_prompt: str,
        query: str,
        user_context: dict | None = None,
        ai_class: Type[T] = AirIntelligence,
    ) -> str:
        logging.info(f"Ініціалізую AI клас: {ai_class.__name__}")
        bot = ai_class(tools=self.tools, system_prompt=system_prompt)
        logging.info(f"User: {query}")

        local_kb: dict = {}

        if user_context:
            local_kb.update(user_context)
        else:
            intent = self.classifier.predict(query)
            if intent:
                call_tool = intent.get("tool")
                call_params: Any = intent.get("params") or {}
                call_params["user_query"] = query

                logging.info(
                    f"Викликаю інструмент: {call_tool} {call_params}"
                )
                method_to_call = getattr(bot.tools, f"{call_tool}")
                tool_result = await method_to_call(**call_params)
                local_kb["search_results"] = tool_result

                if intent:
                    print(
                        f"Знайдено: {intent['tool']} | Params: {intent['params']}")

        context = json.dumps(local_kb, ensure_ascii=False,
                             separators=(',', ':')) if local_kb else None
        final_answer = await bot.process_request(query, context_data=context)

        logging.info(f"[{ai_class.__name__}] Bot: {final_answer}")
        logging.info("")
        return final_answer


async def run_demo(bot: MyBot) -> None:
    """
    Основний сценарій демо:
    """

    def_system_prompt = (
        "Ти — корисний помічник. Відповідай чітко, стисло, без зайвих слів."
    )

    await bot.ask(def_system_prompt, "Хто зараз Президент у USA?",
                  ai_class=OpenAIAirIntelligence)

    await bot.ask(def_system_prompt, "яка температура у будинку?",
                  ai_class=OpenAIAirIntelligence)

    await bot.ask(def_system_prompt, "яка температура у спальні?",
                  ai_class=OpenAIAirIntelligence)

    await bot.ask(def_system_prompt, "Який прогноз погоди?",
                  ai_class=OpenAIAirIntelligence)

    await bot.ask(def_system_prompt, "Яка зараз година?",
                  ai_class=OpenAIAirIntelligence)

    await bot.ask(def_system_prompt, "Який зараз курс USD та EUR до гривні?",
                  ai_class=OpenAIAirIntelligence)

    await bot.ask(def_system_prompt, "Дай приклад інструкції `for` в C#.",
                  ai_class=OpenAIAirIntelligence)

    # Приклад матиматичного аналізу
    wiki_to_csv("https://uk.wikipedia.org/wiki/%D0%9F%D0%B5%D1%80%D0%B5%D0%BB%D1%96%D0%BA_%D1%80%D0%B0%D0%BA%D0%B5%D1%82%D0%BD%D0%B8%D1%85_%D1%83%D0%B4%D0%B0%D1%80%D1%96%D0%B2_%D0%BF%D1%96%D0%B4_%D1%87%D0%B0%D1%81_%D1%80%D0%BE%D1%81%D1%96%D0%B9%D1%81%D1%8C%D0%BA%D0%BE%D0%B3%D0%BE_%D0%B2%D1%82%D0%BE%D1%80%D0%B3%D0%BD%D0%B5%D0%BD%D0%BD%D1%8F_(%D0%B7%D0%B8%D0%BC%D0%B0_2025/2026)")
    math_report = calculate_next_strike()
    logging.info("Математична оцінка наступного удару:")
    logging.info(math_report)

    messages = scrape_messages("https://t.me/s/StrategicaviationT")
    logging.info(f"Отримано {len(messages)} повідомлень з Telegram.")

    military_prompt = (
        "Ти — військовий аналітик. Твоє завдання: проаналізувати повідомлення."
    )

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
    level=logging.DEBUG,
    format='%(asctime)s - %(message)s',  # Час + повідомлення
    datefmt='%Y-%m-%d %H:%M:%S'          # Формат часу
)

logging.getLogger("ddgs").setLevel(logging.ERROR)

logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger('websockets').setLevel(logging.ERROR)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

logging.info("Система запущена")
logging.info("Поточний стан: OK")


async def main():
    bot = MyBot()
    await bot.init_intent_data()
    await run_demo(bot)

if __name__ == "__main__":
    asyncio.run(main())
