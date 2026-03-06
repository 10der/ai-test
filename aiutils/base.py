from abc import ABC, abstractmethod
from datetime import datetime
from .tools import Tools
import json


class BaseAirIntelligence(ABC):
    def __init__(self, tools: Tools, system_prompt: str | None = None):
        self.system_prompt = (
            system_prompt
            if system_prompt
            else "Ти — корисний помічник. Відповідай чітко, стисло, без зайвих слів."
        )
        self.tools = tools

    @abstractmethod
    async def ask_ai(self, messages: list, tools = None, temperature: float = 0.1) -> dict:
        """Надіслати повідомлення до моделі та отримати відповідь."""
        ...

    async def process_request(
        self,
        user_text: str,
        context_data: str | None = None,
        system_prompt_override: str | None = None,
    ) -> str:
    
        if context_data:
            tools = None
            # Аналітик — дані вже є, формуємо лаконічний системний промпт
            current_date = datetime.now().strftime("%d.%m.%Y")
            current_time = datetime.now().strftime("%H:%M")
            sys_message = (
                f"{self.system_prompt}\n"
                f"CURRENT_DATE: {current_date}\n"
                f"CURRENT_TIME: {current_time}\n"
                "ВАЖЛИВО: якщо надано КОНТЕКСТ (результати інструментів) — використовуй його як джерело істини.\n"
                "Якщо КОНТЕКСТ суперечить твоїм знанням — довіряй КОНТЕКСТУ.\n"
            )
            messages = [
                {"role": "system", "content": sys_message},
                {"role": "user", "content": f"КОНТЕКСТ:\n{context_data}\n\nЗАПИТ: {user_text}"},
            ]
        else:
            tools = self.tools.registry.get_tools_schema()
            # Роутер — визначаємо, чи треба інструмент
            sys_message = self._get_router_prompt(system_prompt_override)
            messages = [
                {"role": "system", "content": sys_message},
                {"role": "user", "content": user_text},
            ]
        
        print("Bot: Думаю...")
        message = await self.ask_ai(messages=messages, tools=tools)

        # якщо tool викликається
        if "tool_calls" in message:
            messages.append(message)

            final_text = ""
            for tool_call in message["tool_calls"]:
                name = tool_call["function"]["name"]
                print(f"Bot: [TOOL]: {name}")
                args = tool_call["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)

                if "parameters" in args:
                    args = args["parameters"]

                tool_result = await self.tools.registry.execute(name, **args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(tool_result, ensure_ascii=False)
                })

                # ---- другий запит ----
                print("Bot: Думаю уважно...")
                final_response = await self.ask_ai(messages)
                final_text = final_text + final_response["content"]

            return final_text
        else:
            # без tool
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
