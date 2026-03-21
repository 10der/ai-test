import asyncio
import logging
from typing import Type, Any
from datetime import datetime

import re
import json

from aiutils.ai_client import AirIntelligence, OpenAIAirIntelligence, T
from aiutils.common import load_config
from aiutils.hass_client import HassClient
from aiutils.db_client import DbClient
from aiutils.tools import Tools
from aiutils.my_rag import MyRAG
from wiki_ua_alerts import calculate_next_strike, wiki_to_csv, cleanup_temporary_files

from aiutils.intent_classifier import IntentClassifier

REMEMBER_PATTERNS = re.compile(
    r"запам|збереж|запиш|не забудь|зафіксуй|save",
    re.IGNORECASE
)

def is_remember_intent(text: str) -> bool:
    return bool(REMEMBER_PATTERNS.search(text))
                
class MyTools(Tools):
    def __init__(self, hass_client: HassClient, db_client: DbClient) -> None:
        super().__init__(hass_client, db_client)

    async def intent_war(self, **kwargs) -> dict | None:
        link = "https://uk.wikipedia.org/wiki/%D0%9F%D0%B5%D1%80%D0%B5%D0%BB%D1%96%D0%BA_%D1%80%D0%B0%D0%BA%D0%B5%D1%82%D0%BD%D0%B8%D1%85_%D1%83%D0%B4%D0%B0%D1%80%D1%96%D0%B2_%D0%BF%D1%96%D0%B4_%D1%87%D0%B0%D1%81_%D1%80%D0%BE%D1%81%D1%96%D0%B9%D1%81%D1%8C%D0%BA%D0%BE%D0%B3%D0%BE_%D0%B2%D1%82%D0%BE%D1%80%D0%B3%D0%BD%D0%B5%D0%BD%D0%BD%D1%8F_(%D0%B2%D0%B5%D1%81%D0%BD%D0%B0_2026)"
        wiki_to_csv(link)
        math_report = calculate_next_strike()
        cleanup_temporary_files()
        logging.info("Математична оцінка наступного удару:")
        logging.info(math_report)

        messages = await self.intent_tele_channel("StrategicaviationT")
        if messages:
            logging.info(f"Отримано {len(messages)} повідомлень з Telegram.")

        return {"military_math_stat_report": math_report, "military_channel_messages": messages}


class MyBot:

    def __init__(self):
        self.rag_chunks: list[dict] = []
        self.classifier = IntentClassifier(threshold=0.7)
        self.rag = MyRAG()

        config = load_config()
        self.hass_client = HassClient(
            base_url=config.get("hass", {}).get("url"),
            token=config.get("hass", {}).get("token"),
        )
        self.db_client = DbClient()
        self.tools = MyTools(hass_client=self.hass_client,
                             db_client=self.db_client)

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
                self.classifier.add_intent(f"яка температура скільки градусів {name}", self.tools.intent_hass, {
                                           "room": room, "device": "температура", "entity_id": rooms[room]["data"].get('temperature_entity_id')})
                self.classifier.add_intent(f"яка вологість {name}", self.tools.intent_hass, {
                                           "room": room, "device": "вологість", "entity_id": rooms[room]["data"].get('humidity_entity_id')})

        self.classifier.add_intent(
            "курс валют долар євро гривня", self.tools.intent_currency, {})

        self.classifier.add_intent("прогноз погоди", self.tools.intent_weather_dnipro, {
                                   "room": "", "device": "погода"})

        self.classifier.add_intent([
            "війна масований ракетний удар",
            "ракетна загроза сьогодні",
            "військова ситуація аналіз загрози",
            "ознаки підготовки до удару",
            "оцінка ракетної небезпеки",
        ], self.tools.intent_war, {"system_prompt": "Ти — військовий аналітик."})

        self.classifier.build_index()

    async def find_relevant_history(self, source: str, query: str, top_k: int = 3):
        results = await self.rag.search(query, top_k=top_k, source=source)
        return [{"role": "user", "content": r["static_data"]} for r in results]

    async def ask(
        self,
        system_prompt: str,
        query: str,
        user_context: dict | None = None,
        ai_class: Type[T] = AirIntelligence,
    ) -> None:
        logging.info(f"Ініціалізую AI клас: {ai_class.__name__}")
        logging.info(f"User: {query}")

        chat_id = 255464392
        user_id = 255464392
        username = "alf_brd"
        first_name = "IT Alf"

        local_kb: dict = {}
        system_prompt_override = None
        if user_context:
            local_kb.update(user_context)
        else:
            intent = self.classifier.predict(query)
            if intent:
                call_tool = intent["tool"]
                call_params: Any = intent.get("params") or {}
                call_params["user_query"] = query
                call_params["chat_id"] = chat_id
                call_params["user_id"] = user_id
                call_params["first_name"] = first_name

                if "system_prompt" in call_params:
                    system_prompt_override = call_params["system_prompt"]

                logging.info(
                    f"Викликаю інструмент: {call_tool.__name__} {call_params}"
                )
                tool_result = await intent["tool"](**call_params)
                if tool_result:
                    local_kb["search_results"] = tool_result

        context = json.dumps(local_kb, ensure_ascii=False,
                             separators=(',', ':')) if local_kb else None

        bot = ai_class(tools=self.tools, system_prompt=system_prompt)
        history = await self.db_client.get_user_history(chat_id, user_id, limit=10)
        relevant = await self.find_relevant_history(f"chat_{chat_id}-user_{user_id}", query)
        history.extend(relevant)
        final_answer = await bot.process_request(query, context_data=context,
                                                 system_prompt_override=system_prompt_override,
                                                 history=history)

        logging.info(f"[{ai_class.__name__}] Bot: {final_answer}")
        logging.info("")

        if final_answer:
            await self.db_client.save_message(chat_id, user_id, username, first_name, query)
            await self.db_client.save_message(chat_id, user_id, ai_class.__name__, "", final_answer, role="assistant")

            if is_remember_intent(query):
                # save as RAG
                self.rag.add_document(
                    query,
                    source=f"chat_{chat_id}-user_{user_id}",
                    date_str=datetime.now()
                )

        else:
            print("AI: Не зміг знайти відповідь.")


async def test(bot: MyBot) -> None:
    chat_id = -1001928843751
    user_id = 394234205

    all = await bot.db_client.get_user_messages(chat_id, "mrak_orm")
    print(all)

    history = await bot.db_client.get_user_history(chat_id, user_id, limit=10)
    print(history)
    relevant = await bot.find_relevant_history(f"chat_{chat_id}-user_{user_id}", "Як мене звати?")
    print(relevant)
    # bot.rag.add_document(
    #     "Мене зовуть Олег.",
    #     source=f"chat_{chat_id}-user_{user_id}",
    #     date_str=datetime.now()
    # )


async def run_demo(bot: MyBot) -> None:
    """
    Основний сценарій демо:
    """

    def_system_prompt = (
        "Ти — корисний помічник. Відповідай чітко, стисло, без зайвих слів."
    )

    await bot.ask(def_system_prompt, "Запам'ятай - я живу у місті Дніпро.",
                  ai_class=OpenAIAirIntelligence)

    # await bot.ask(def_system_prompt, "Запамʼятай - мене зовуть Олег.",
    #               ai_class=OpenAIAirIntelligence)

    # await bot.ask(def_system_prompt, "Як мене звати?",
    #               ai_class=OpenAIAirIntelligence)

    # await bot.ask(def_system_prompt, "Хто зараз Президент у USA?",
    #               ai_class=OpenAIAirIntelligence)

    # await bot.ask(def_system_prompt, "яка температура у будинку?",
    #               ai_class=OpenAIAirIntelligence)

    # await bot.ask(def_system_prompt, "яка температура у спальні?",
    #               ai_class=OpenAIAirIntelligence)

    # await bot.ask(def_system_prompt, "Який прогноз погоди?",
    #               ai_class=OpenAIAirIntelligence)

    # await bot.ask(def_system_prompt, "Яка зараз година?",
    #               ai_class=OpenAIAirIntelligence)

    # await bot.ask(def_system_prompt, "Який зараз курс USD та EUR до гривні?",
    #               ai_class=OpenAIAirIntelligence)

    # await bot.ask(def_system_prompt, "Дай приклад інструкції `for` в C#.",
    #               ai_class=OpenAIAirIntelligence)

    # Приклад матиматичного аналізу
    # await bot.ask(def_system_prompt, """Проаналізуй війскову ситуацію. Чи є ознаки підготовки до масованого ракетного удару.
    #               Надай коротку оцінку загрози (Low/Medium/High/Critical).""",
    #               ai_class=OpenAIAirIntelligence)

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

logging.getLogger("aiosqlite").setLevel(logging.WARNING)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

logging.info("Система запущена")
logging.info("Поточний стан: OK")


async def main():
    bot = MyBot()
    await bot.init_intent_data()
    await bot.db_client.init()
    # await test(bot)
    await run_demo(bot)

if __name__ == "__main__":
    asyncio.run(main())
