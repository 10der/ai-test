from abc import ABC
from datetime import datetime
from .tools import Tools
import json
import logging


class BaseAirIntelligence(ABC):
    def __init__(self, tools: Tools, system_prompt: str | None = None):
        self.system_prompt = (
            system_prompt
            if system_prompt
            else "Ти — корисний помічник. Відповідай чітко, стисло, без зайвих слів."
        )
        self.tools = tools

    async def ask_ai(self, messages: list, tools_schema: list | None = None, temperature: float = 0.1) -> dict:
        """Надіслати повідомлення до моделі та отримати відповідь."""

        logging.debug("Bot: Думаю...")
        return {"role": "assistant", "content": ""}

    async def process_request(
        self,
        user_text: str,
        context_data: str | None = None,
        system_prompt_override: str | None = None,
        history: list = [],
    ) -> str:

        if context_data:
            tools_schema = None
            current_date = datetime.now().strftime("%d.%m.%Y")
            current_time = datetime.now().strftime("%H:%M")
            sys_message = (
                f"{self.system_prompt}\n"
                f"CURRENT_DATE: {current_date}\n"
                f"CURRENT_TIME: {current_time}\n"
                "ВАЖЛИВО: якщо надано КОНТЕКСТ (результати інструментів) — використовуй його як джерело істини.\n"
                "Якщо КОНТЕКСТ суперечить твоїм знанням — довіряй КОНТЕКСТУ.\n"
            )

            messages = [{"role": "system", "content": sys_message}]
            messages.append({"role": "user", "content": f"КОНТЕКСТ:\n{context_data}\n\nЗАПИТ: {user_text}"})

        else:
            tools_schema = self.tools.get_tools_schema()
            sys_message = self._get_router_prompt(system_prompt_override)

            messages = [{"role": "system", "content": sys_message}]
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": user_text})

        message = await self.ask_ai(messages=messages, tools_schema=tools_schema)

        if "tool_calls" in message:
            messages.append(message)

            # Збираємо всі результати тулів за один прохід
            for tool_call in message["tool_calls"]:
                name = tool_call["function"]["name"]
                args = tool_call["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)

                if "parameters" in args:
                    args = args["parameters"]

                logging.info(f"[LLM TOOL]: {name} {args}")
                tool_result = await self.tools.execute(name, **args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(tool_result, ensure_ascii=False)
                })

            # Один фінальний запит після всіх тулів
            final_response = await self.ask_ai(messages)
            return final_response.get("content", "")
        else:
            return message.get("content", "")

    def _get_router_prompt(self, system_prompt_override: str | None = None) -> str:
        current_date = datetime.now().strftime("%d.%m.%Y")
        current_time = datetime.now().strftime("%H:%M")
        base = system_prompt_override or self.system_prompt

        return (
            f"{base}\n\n"
            f"CURRENT_DATE: {current_date}\n"
            f"CURRENT_TIME: {current_time}\n"
            "KNOWLEDGE_CUTOFF: 2023-10\n\n"
            "РОУТЕР:\n"
            "- Або відповідай коротко текстом.\n"
            "- Або виклич інструмент.\n\n"
            "БАЗОВІ ОБМЕЖЕННЯ:\n"
            "1) КАТЕГОРИЧНО ЗАБОРОНЕНО: Не вибачайтеся. Не пояснюйте, чому у вас обмежені знання.\n"
            f"2) Якщо ти не впевнений щодо поточного статусу запита станом на {current_date} - ЗАВЖДИ ВИКЛИКАЙ ІНСТРУМЕНТ!\n"
        )
