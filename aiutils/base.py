from abc import ABC, abstractmethod
from datetime import datetime


class BaseAirIntelligence(ABC):
    def __init__(self, tools, system_prompt: str | None = None):
        self.system_prompt = (
            system_prompt
            if system_prompt
            else "Ти — корисний помічник. Відповідай чітко, стисло, без зайвих слів."
        )
        self.tools = tools

    @abstractmethod
    async def ask_ai(self, messages: list, temperature: float = 0.1) -> str:
        """Надіслати повідомлення до моделі та отримати відповідь."""
        ...

    async def process_request(
        self,
        user_text: str,
        context_data: str | None = None,
        system_prompt_override: str | None = None,
        _depth: int = 0,
    ) -> str:
        if _depth > 3:
            return "Помилка: перевищено максимальну глибину обробки запиту."

        if context_data:
            # Аналітик — дані вже є, формуємо лаконічний системний промпт
            current_date = datetime.now().strftime("%d.%m.%Y")
            current_time = datetime.now().strftime("%H:%M")
            sys_message = (
                f"{self.system_prompt}\n"
                f"CURRENT_DATE: {current_date}\n"
                f"CURRENT_TIME: {current_time}\n"
            )
            messages = [
                {"role": "system", "content": sys_message},
                {"role": "user", "content": f"КОНТЕКСТ:\n{context_data}\n\nЗАПИТ: {user_text}"},
            ]
        else:
            # Роутер — визначаємо, чи треба інструмент
            sys_message = self._get_router_prompt(system_prompt_override)
            messages = [
                {"role": "system", "content": sys_message},
                {"role": "user", "content": user_text},
            ]

        print(f"--- Thinking... (depth={_depth}, context={'Yes' if context_data else 'No'}) ---")
        ai_response = await self.ask_ai(messages)

        # Перевіряємо, чи модель попросила інструмент (тільки на першому рівні)
        if not context_data:
            method_name = next(
                (name for name in self.tools.tool_names if ai_response.startswith(name)),
                None,
            )
            if method_name:
                print(f"--- AI requested tool: {ai_response} ---")
                query = ai_response[len(method_name)+1:].strip()
                tool_result = await getattr(self.tools, method_name)(query, None)
                return await self.process_request(
                    user_text,
                    context_data=tool_result,
                    _depth=_depth + 1,
                )

        return ai_response

    def _get_router_prompt(self, system_prompt_override: str | None = None) -> str:
        current_date = datetime.now().strftime("%d.%m.%Y")
        current_time = datetime.now().strftime("%H:%M")
        base = system_prompt_override or self.system_prompt

        return (
            f"{base}\n\n"
            f"CURRENT_DATE: {current_date}\n"
            f"CURRENT_TIME: {current_time}\n"
            f"KNOWLEDGE_CUTOFF: 2023-10\n\n"
            "RULES:\n"
            "0. ЗАВЖДИ ВИКОРИСТОВУЙ ДАТУ CURRENT_DATE ТА ЧАС CURRENT_TIME У СВОЇХ ВІДПОВІДЯХ.\n"
            "1. If the user's query refers to events, people, or facts occurring AFTER 2023-10, "
            "you MUST NOT provide a text answer.\n"
            "2. For all post-2023 queries, output EXACTLY: tool_general_search: [query]\n"
            "3. ABSOLUTELY FORBIDDEN: Do not apologize. Do not explain your knowledge cutoff.\n"
            f"4. If you are unsure about the current status of an entity as of {current_date}, "
            "use tool_general_search.\n"
            "5. ЗАГАЛЬНІ ЗНАННЯ (математика, програмування, граматика, історія до 2023) — "
            "не потребують пошуку. Відповідай одразу.\n"
            "6. Якщо користувач запитує про стан кімнати, ВИКОРИСТОВУЙ ІНСТРУМЕНТ! "
            "Визнач локацію (room). Визнач сутність (device). "
            "Виклич tool_hass?room=room&device=device. "
            "ПРИКЛАД: 'яка температура в спальні?' -> room=спальня, device=температура "
            "-> tool_hass?room=спальня&device=температура\n"
        )
