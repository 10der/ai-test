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

    async def ask_ai(self, messages: list, temperature: float = 0.1) -> str:
        """Спілкування з Ollama"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_thread": 8
            }
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(self.url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()['message']['content']


class OpenAIAirIntelligence(BaseAirIntelligence):
    def __init__(self, tools: Tools, system_prompt: str | None = None):
        super().__init__(tools=tools, system_prompt=system_prompt)
        config = load_config()
        self.model = config.get("openai", {}).get("model")
        self.url = "https://api.openai.com/v1/chat/completions"
        self.api_key = config.get("openai", {}).get("api_key")

    async def ask_ai(self, messages: list, temperature: float = 0.1) -> str:
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

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.url, json=payload, headers=headers, timeout=60
                )

                if response.status_code != 200:
                    print(f"Debug OpenAI Raw Response: {response.text}")

                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']

        except Exception as e:
            return f"OpenAI Error: {str(e)}"
