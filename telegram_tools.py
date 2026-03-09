import time
import random
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from datetime import timedelta

# Набір реальних UA для ротації
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

def build_headers() -> dict:
    """Генерує реалістичні headers з ротацією UA."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "uk-UA,uk;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0",
    }

def fetch_with_retry(url: str, max_retries: int = 3, base_delay: float = 2.0) -> requests.Response | None:
    """
    Робить запит з retry + exponential backoff при помилках.
    При 429 (rate limit) чекає довше.
    """
    for attempt in range(max_retries):
        try:
            # Невелика рандомна пауза перед кожним запитом (імітація людини)
            jitter = random.uniform(1.0, 3.0)
            time.sleep(jitter)

            response = requests.get(url, headers=build_headers(), timeout=60)

            if response.status_code == 200:
                return response

            if response.status_code == 429:
                wait = base_delay * (2 ** attempt) + random.uniform(0, 2)
                print(f"[Rate limit 429] Чекаємо {wait:.1f}с перед повторою...")
                time.sleep(wait)
                continue

            if response.status_code in (403, 404):
                print(f"[HTTP {response.status_code}] Зупиняємось.")
                return None

            # Інші помилки — retry
            print(f"[HTTP {response.status_code}] Спроба {attempt + 1}/{max_retries}")
            time.sleep(base_delay * (2 ** attempt))

        except requests.exceptions.Timeout:
            print(f"[Timeout] Спроба {attempt + 1}/{max_retries}")
            time.sleep(base_delay * (2 ** attempt))
        except requests.exceptions.ConnectionError as e:
            print(f"[ConnectionError] {e} | Спроба {attempt + 1}/{max_retries}")
            time.sleep(base_delay * (2 ** attempt))

    print(f"[FAIL] Не вдалося отримати: {url}")
    return None


def scrape_messages(channel_url: str, days_ago: int = 0) -> list[dict]:
    """Telegram"""
    target_date = datetime.now(timezone.utc).date() - timedelta(days=days_ago)
    result = []
    last_msg_id = None

    while True:
        url = f"{channel_url}?before={last_msg_id}" if last_msg_id else channel_url

        response = fetch_with_retry(url)
        if response is None:
            break

        soup = BeautifulSoup(response.text, 'html.parser')
        messages = soup.find_all('div', class_='tgme_widget_message', recursive=True)
        if not messages:
            break

        found_older_than_today = False
        batch_results = []

        for content in messages:
            try:
                timestamp_obj = content.find("span", class_="tgme_widget_message_meta")
                if not timestamp_obj:
                    continue

                message_date_obj = timestamp_obj.find("a", class_="tgme_widget_message_date")
                if not message_date_obj:
                    continue

                msg_url = message_date_obj.attrs.get("href")
                if not msg_url:
                    continue

                msg_id = int(str(msg_url).split("/")[-1])
                msg_time_obj = message_date_obj.find('time', class_='time')
                if not msg_time_obj:
                    continue

                timestamp = datetime.fromisoformat(str(msg_time_obj.attrs.get("datetime")))
                timestamp_str = timestamp.isoformat()
                msg_date = timestamp.date()

                if msg_date < target_date:
                    found_older_than_today = True
                    continue

                if msg_date > target_date:
                    continue

                text = ""
                msg_text_obj = content.find_all('div', class_='tgme_widget_message_text')
                for text_obj in msg_text_obj:
                    if text_obj is not None:
                        for br in text_obj.find_all("br"):
                            br.replace_with(" ")
                        classes = text_obj.attrs.get('class', [])
                        if 'js-message_reply_text' in classes:
                            text += " re: "
                        if 'js-message_text' in classes:
                            text += " " + text_obj.text

                batch_results.append({
                    "id": msg_id,
                    "url": msg_url,
                    "date": timestamp_str,
                    "content": ' '.join(text.split())
                })

            except Exception as e:
                print(f"[Parse error] {e}")
                continue

        result.extend(batch_results)

        try:
            first_msg_url = messages[0].find(
                "a", class_="tgme_widget_message_date").attrs.get("href")
            last_msg_id = str(first_msg_url).split("/")[-1]
        except Exception:
            break

        if found_older_than_today:
            break

    return sorted(result, key=lambda x: x['id'])
