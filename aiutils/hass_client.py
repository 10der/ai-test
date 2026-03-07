import httpx
import json
import websockets

class HassClient:
    def __init__(self, base_url: str, token: str, timeout: int = 5):
        self.base_url = base_url.rstrip("/")
        self.ws_url = self.base_url.replace("http", "ws") + "/api/websocket"
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        self.timeout = timeout

    async def get(self, path: str):
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    f"{self.base_url}{path}",
                    headers=self.headers,
                    timeout=self.timeout,
                )

            if r.status_code == 404:
                return None

            r.raise_for_status()
            return r.json()

        except httpx.HTTPError:
            return None

    async def post(self, path: str, payload: dict | None = None) -> dict | str | None:
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    f"{self.base_url}{path}",
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout,
                )

            if r.status_code == 404:
                return None

            r.raise_for_status()

            text = r.text.strip()

            if not text:
                return None

            try:
                return r.json()
            except ValueError:
                pass

            try:
                return json.loads(text)
            except ValueError:
                pass

            return text

        except httpx.HTTPError:
            return None

    async def get_entity(self, entity_id: str):
        data = await self.get(f"/api/states/{entity_id}")

        if not data:
            return None

        if data.get("state") in ("unavailable", "unknown"):
            return None

        return data

    async def get_entities(self, entity_ids: list[str]):
        data = await self.get("/api/states")
        result = {}

        if not data:
            return result

        wanted = set(entity_ids)

        for e in data:
            eid = e.get("entity_id")
            if eid not in wanted:
                continue

            if e.get("state") in ("unavailable", "unknown"):
                continue

            result[eid] = e

        return result

    async def get_state(self, entity_id: str):
        e = await self.get_entity(entity_id)
        return e["state"] if e else None

    async def get_attr(self, entity_id: str, name: str):
        e = await self.get_entity(entity_id)
        return e["attributes"].get(name) if e else None

    async def render_template(self, template):
        data = await self.post("/api/template", {
            "template": template
        })

        return data if data else None

    async def call_ws_command(self, msg_type: str, **kwargs):
            """Викликає разову команду через WebSocket і повертає результат"""
            try:
                async with websockets.connect(self.ws_url) as ws:
                    # 1. Авторизація
                    auth_msg = json.loads(await ws.recv())
                    if auth_msg.get("type") != "auth_required":
                        return None

                    await ws.send(json.dumps({
                        "type": "auth",
                        "access_token": self.token
                    }))

                    auth_result = json.loads(await ws.recv())
                    if auth_result.get("type") != "auth_ok":
                        return None

                    # 2. Відправка команди
                    msg_id = 1 # Для разового запиту вистачить
                    payload = {"id": msg_id, "type": msg_type, **kwargs}
                    await ws.send(json.dumps(payload))

                    # 3. Отримання результату
                    while True:
                        response = json.loads(await ws.recv())
                        if response.get("id") == msg_id:
                            if response.get("success"):
                                return response.get("result")
                            return None
            except Exception as e:
                print(f"WS Error: {e}")
                return None