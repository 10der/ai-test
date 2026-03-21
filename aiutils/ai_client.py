import httpx
from .common import load_config
from .tools import Tools

from typing import TypeVar
from .base import BaseAirIntelligence

T = TypeVar('T', bound=BaseAirIntelligence)


class AirIntelligence(BaseAirIntelligence):
    def __init__(self, tools: Tools, system_prompt: str | None = None):
        super().__init__(tools=tools, system_prompt=system_prompt)
        config = load_config()
        self.model = config.get("ollama", {}).get("model")
        self.url = config.get("ollama", {}).get("url")

    async def ask_ai(self, messages: list,  tools_schema: list | None = None, temperature: float = 0.1) -> dict:
        """Спілкування з Ollama"""

        await super().ask_ai(messages, tools_schema, temperature)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_thread": 8,
                "num_ctx": 2048,
                "top_k": 20,
            }
        }

        if tools_schema:
            payload["tools"] = tools_schema

        headers = {
            'User-Agent':
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/91.0.4472.124 Safari/537.36',
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(self.url, json=payload, timeout=60, headers=headers)
            response.raise_for_status()
            return response.json()['message']


class OpenAIAirIntelligence(BaseAirIntelligence):
    def __init__(self, tools: Tools, system_prompt: str | None = None):
        super().__init__(tools=tools, system_prompt=system_prompt)
        config = load_config()
        self.model = config.get("openai", {}).get("model")
        self.url = "https://api.openai.com/v1/chat/completions"
        self.api_key = config.get("openai", {}).get("api_key")

    async def ask_ai(self, messages: list, tools_schema: list | None = None, temperature: float = 0.1) -> dict:
        """OpenAI call"""

        await super().ask_ai(messages, tools_schema, temperature)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1000
        }

        if tools_schema:
            payload["tools"] = tools_schema

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.url, json=payload, headers=headers, timeout=60
            )

            if response.status_code != 200:
                print(f"Debug OpenAI Raw Response: {response.text}")

            response.raise_for_status()
            return response.json()['choices'][0]['message']


class GeminiAirIntelligence(BaseAirIntelligence):
    def __init__(self, tools: Tools, system_prompt: str | None = None):
        super().__init__(tools=tools, system_prompt=system_prompt)
        config = load_config()
        self.model = config.get("gemini", {}).get("model", "gemini-1.5-flash")
        self.api_key = config.get("gemini", {}).get("api_key")
        self.url = config.get("gemini", {}).get("url")

    async def ask_ai(self, messages: list, tools_schema: list | None = None, temperature: float = 0.1) -> dict:
        """Gemini call"""

        await super().ask_ai(messages, tools_schema, temperature)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2048
        }

        if tools_schema:
            # Gemini підтримує формат інструментів OpenAI через цей endpoint
            payload["tools"] = tools_schema

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.url, json=payload, headers=headers, timeout=60
            )

            if response.status_code != 200:
                print(f"Debug Gemini Raw Response: {response.text}")

            response.raise_for_status()
            return response.json()['choices'][0]['message']
