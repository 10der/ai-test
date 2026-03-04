from abc import ABC, abstractmethod
from datetime import datetime
import re

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
                "ВАЖЛИВО: якщо надано КОНТЕКСТ (результати інструментів) — використовуй його як джерело істини.\n"
                "Якщо КОНТЕКСТ суперечить твоїм знанням — довіряй КОНТЕКСТУ.\n"
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
            tool_call = self._extract_tool_call(ai_response)
            if tool_call:
                method_name, query = tool_call
                print(f"--- AI requested tool: {method_name}: {query} ---")
                tool_result = await getattr(self.tools, method_name)(query, None)
                return await self.process_request(
                    user_text,
                    context_data=tool_result,
                    _depth=_depth + 1,
                )

        return ai_response

    def _get_fallback_tool_name(self) -> str | None:
        specs = getattr(self.tools, "tool_specs", None)
        if isinstance(specs, list) and specs:
            fb = next((s.name for s in specs if getattr(s, "is_fallback", False)), None)
            if fb:
                return fb
        return "tool_general_search" if "tool_general_search" in self.tools.tool_names else None

    def _extract_tool_call(self, ai_response: str) -> tuple[str, str] | None:
        """
        Витягує виклик інструмента з відповіді моделі.

        Підтримує кейси, коли модель загорнула виклик у бектики або код-блок.
        Очікуваний формат (після нормалізації): `tool_name: query`
        """
        text = (ai_response or "").strip()
        if not text:
            return None

        # ```tool_xxx: ...``` або ```\n tool_xxx: ... \n```
        if text.startswith("```"):
            lines = [ln.strip() for ln in text.splitlines()]
            # прибираємо першу/останню лінію з ```
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = " ".join(lines).strip()

        # `tool_xxx: ...`
        if text.startswith("`") and text.endswith("`") and len(text) >= 2:
            text = text[1:-1].strip()

        # прибираємо ще раз можливі випадкові бектики по краях
        text = text.strip().strip("`").strip()

        # 1) Перший варіант: модель написала щось на кшталт:
        # "я викликаю `tool_hass`:\n\n`?room=...&device=...`"
        m = re.search(r"(tool_[a-zA-Z0-9_]+)[` ]*:?\s*`?([^`\n]+)`?", text, re.MULTILINE)
        if m:
            name = m.group(1).strip()
            query = m.group(2).strip()
            if name in self.tools.tool_names:
                return name, query

        # Дозволяємо моделі писати додатковий текст.
        # Шукаємо РЯДОК, який починається з назви інструмента.
        candidate_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        for name in self.tools.tool_names:
            for line in candidate_lines:
                if not line.startswith(name):
                    continue

                rest = line[len(name):].lstrip()
                if rest.startswith(":"):
                    query = rest[1:].strip()
                else:
                    # на випадок "tool_xxx ?a=b" або "tool_xxx   query"
                    query = rest.strip()

                return name, query

        return None

    def _get_router_prompt(self, system_prompt_override: str | None = None) -> str:
        current_date = datetime.now().strftime("%d.%m.%Y")
        current_time = datetime.now().strftime("%H:%M")
        base = system_prompt_override or self.system_prompt

        tool_lines: list[str] = []

        # 1) Бажаний шлях: структуровані спеки з Tools (опис/формат/коли/приклади)
        specs = getattr(self.tools, "tool_specs", None)
        fallback_tool = None
        if isinstance(specs, list) and specs:
            fallback_tool = next((s.name for s in specs if getattr(s, "is_fallback", False)), None)
            for spec in specs:
                tool_lines.append(f"- {spec.name}: {spec.description}")
                tool_lines.append(f"  input: {spec.input_format}")
                tool_lines.append(f"  when: {spec.when_to_use}")
                if getattr(spec, "examples", None):
                    ex0 = spec.examples[0]
                    tool_lines.append(f"  example: {ex0}")
        else:
            # 2) Fallback: docstring з самого методу
            for name in sorted(self.tools.tool_names):
                fn = getattr(self.tools, name, None)
                doc = getattr(fn, "__doc__", "") or ""
                doc_one_line = " ".join(doc.split()).strip()
                if doc_one_line:
                    tool_lines.append(f"- {name}: {doc_one_line}")
                else:
                    tool_lines.append(f"- {name}")

        tools_block = "\n".join(tool_lines) if tool_lines else "- (інструменти недоступні)"
        fallback_line = (
            f"3) Якщо ти НЕ ВПЕВНЕНИЙ або потрібна актуальна інформація — виклич {fallback_tool}.\n"
            if fallback_tool
            else ""
        )

        return (
            f"{base}\n\n"
            f"CURRENT_DATE: {current_date}\n"
            f"CURRENT_TIME: {current_time}\n"
            "KNOWLEDGE_CUTOFF: 2023-10\n\n"
            "РОУТЕР:\n"
            "- Або відповідай коротко текстом.\n"
            "- Або виклич інструмент.\n\n"
            "БАЗОВІ ОБМЕЖЕННЯ:\n"            
            "0) КАТЕГОРИЧНО ЗАБОРОНЕНО: Не вибачайтеся. Не пояснюйте, чому у вас обмежені знання.\n"
            f"1) Якщо ти не впевнений щодо поточного статусу запита станом на {current_date} - ЗАВЖДИ ВИКЛИКАЙ ІНСТРУМЕНТ, АЛЕ НЕ ПИШИ ПРО ЦЕ!\n"
            "2) Якщо викликаєш інструмент — ПОВЕРНИ РІВНО ОДИН РЯДОК:\n"
            "   tool_name: query\n"
            "   Без будь-якого іншого тексту.\n"
            "3) Опис інструментів та правила формування query — нижче (бери їх звідти).\n"
            f"{fallback_line}\n"
            "ІНСТРУМЕНТИ (динамічно з підключених tool_*):\n"
            f"{tools_block}\n"
        )
