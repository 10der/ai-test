import requests
import json


class HassClient:
    def __init__(self, base_url: str, token: str, timeout: int = 5):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self.timeout = timeout

    def get(self, path: str):
        try:
            r = requests.get(
                f"{self.base_url}{path}",
                headers=self.headers,
                timeout=self.timeout,
            )

            if r.status_code == 404:
                return None

            r.raise_for_status()
            return r.json()

        except requests.RequestException:
            return None

    def post(self, path: str, payload: dict | None = None) -> dict | str | None:
        try:
            r = requests.post(
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

            # пробуємо нормальний JSON
            try:
                return r.json()
            except ValueError:
                pass

            # пробуємо руками
            try:
                return json.loads(text)
            except ValueError:
                pass

            # fallback
            return text

        except requests.RequestException:
            return None

    def get_entity(self, entity_id: str):
        data = self.get(f"/api/states/{entity_id}")

        if not data:
            return None

        if data.get("state") in ("unavailable", "unknown"):
            return None

        return data

    def get_entities(self, entity_ids: list[str]):
        data = self.get("/api/states")
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

    def get_state(self, entity_id: str):
        e = self.get_entity(entity_id)
        return e["state"] if e else None

    def get_attr(self, entity_id: str, name: str):
        e = self.get_entity(entity_id)
        return e["attributes"].get(name) if e else None

    def render_template(self, template):
        """
        Renders a Home Assistant template.

        Args:
            template (str): The Jinja2 template string to render.

        Returns:
            str | None: The rendered template string if successful, otherwise None.
        """

        data = self.post("/api/template", {
            "template": template
        })

        return data if data else None

    def get_entity_by_room_and_friendly_name(self, room_name: str, friendly_name) -> str | None:

        room_name = room_name.lower()
        friendly_name = friendly_name.lower()

        tpl = f"""
{{% set ns = namespace(area_id=none) %}}
{{% for a in areas() %}}
{{% if area_name(a) | lower == '{room_name}' %}}
    {{% set ns.area_id = a %}}
{{% endif %}}
{{% endfor %}}

{{{{
    area_entities(ns.area_id)
        | expand
        | selectattr('attributes.friendly_name', 'defined')
        | selectattr(
            'attributes.friendly_name',
            'search',
            '{friendly_name}',
            ignorecase=True
        )
        | map(attribute='entity_id')
        | first
        | tojson

}}}}
"""

        area_device = self.render_template(tpl)

        if not area_device:
            return None

        return str(area_device)
