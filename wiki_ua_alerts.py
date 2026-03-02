import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import glob
import re

def wiki_to_csv(wikiurl):
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    response = requests.get(wikiurl, headers=headers)
    response.encoding = 'utf-8' # Фікс для кирилиці
    
    soup = BeautifulSoup(response.text, 'html.parser')
    tables = soup.find_all('table', {'class': 'wikitable'})
    for i, table in enumerate(tables):
        if i == 0:
            continue

        print(f"Processing Table {i+1}...")
        df = pd.read_html(str(table))[0]
        
        df = df[['дата', 'Удари завдані Росією']]
        filename = f"strikes_{i+1}.csv"

        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"Saved: {filename}")        

def is_massive(row):
    stats_cell = str(row.iloc[4]).lower()    
    match = re.search(r'(\d+)/(\d+)', stats_cell)
    
    if match:
        total = int(match.group(2))
        if total >= 15:
            return True
            
    return False

def calculate_next_strike(target_date=None):    
    today = datetime.now()
    
    if target_date is not None:
        today = datetime.strptime(target_date, "%d.%m.%Y") if isinstance(target_date, str) else target_date
    
    today = today.replace(hour=23, minute=59, second=59)

    files = glob.glob("strikes_*.csv")
    if not files:
        print("Файли не знайдені!")
        return

    all_data = pd.concat([pd.read_csv(f) for f in files])
    
    print(f"Розрахунок ведеться відносно дати: {today.strftime('%d.%m.%Y')}")

    all_data['date_dt'] = pd.to_datetime(all_data.iloc[:, 0], dayfirst=True, errors='coerce')
    all_data = all_data[all_data['date_dt'] < today]

    mask = all_data.apply(is_massive, axis=1)

    mass_strikes = all_data[
        (mask) &
        (all_data.iloc[:,1] == "Україна")
    ].copy()

    #####
    # synthetic_today = pd.to_datetime('2026-02-22')
    # if synthetic_today <= today:
    #     new_event = {
    #         'date_dt': pd.to_datetime('2026-02-22'), # Сьогоднішній приліт
    #         all_data.columns[4]: '50/70' # Синтетична статистика для проходження фільтру
    #     }
    #     new_row = pd.DataFrame([new_event])
    #     mass_strikes = pd.concat([mass_strikes, new_row], ignore_index=True)
    #####

    mass_strikes = mass_strikes.drop_duplicates(subset=['date_dt'])   
    mass_strikes = mass_strikes.sort_values('date_dt')

    if len(mass_strikes) < 2:
        print("Недостатньо даних для розрахунку циклу.")
        return

    mass_strikes['delta'] = mass_strikes['date_dt'].diff().dt.days
    
    avg_interval = mass_strikes['delta'].median()
    std = mass_strikes['delta'].std()
    
    recent_trend = mass_strikes['delta'].rolling(window=3).mean().iloc[-1]
    trend_diff = avg_interval - recent_trend
    if trend_diff > 0.5:
        trend_status = f"ПРИСКОРЕННЯ (темп зріс на {abs(trend_diff):.1f} дн.)"
    elif trend_diff < -0.5:
        trend_status = f"УПОВІЛЬНЕННЯ (пауза зросла на {abs(trend_diff):.1f} дн.)"
    else:
        trend_status = "СТАБІЛЬНО (темп без змін)"
    
    last_strike = mass_strikes['date_dt'].max()
    days_since_last = (today - last_strike).days

    expected_date = last_strike + pd.Timedelta(days=avg_interval)
    
    # Вікно небезпеки (Confidence Interval)
    earliest = expected_date - timedelta(days=int(std))
    latest = expected_date + timedelta(days=int(std))

    # Створюємо список рядків для зручності
    report_lines = [
        "--- АНАЛІТИКА ШТАБУ ---",
        f"Зараз: {today.strftime('%d.%m.%Y')}",
        f"Останній масований обстріл: {last_strike.strftime('%d.%m.%Y')}",
        f"Тренд: {trend_status}",
        f"Середній цикл: {avg_interval:.1f} днів, Стандартне відхилення: {std:.1f} днів",
        f"Минуло днів з останньої атаки: {days_since_last} днів",
        f"Вікно очікуваної атаки: з {earliest.strftime('%d.%m')} по {latest.strftime('%d.%m')}",
        f"Мат. очікування наступного: {expected_date.strftime('%d.%m.%Y')}"
    ]

    # З'єднуємо все через символ переносу рядка
    full_report = "\n".join(report_lines)
    
    return full_report
    
def create_report():
    all_files = glob.glob("strikes_*.csv")
    combined_df = pd.concat([pd.read_csv(f) for f in all_files])
    combined_df = combined_df.dropna(how='all')

    combined_df.to_csv("all_strikes_2026_final.csv", index=False, encoding='utf-8-sig')
    print("Final dataset created for RAG!")    

if __name__ == "__main__":
    # grab_aviation_messages()
    wiki_to_csv("https://uk.wikipedia.org/wiki/%D0%9F%D0%B5%D1%80%D0%B5%D0%BB%D1%96%D0%BA_%D1%80%D0%B0%D0%BA%D0%B5%D1%82%D0%BD%D0%B8%D1%85_%D1%83%D0%B4%D0%B0%D1%80%D1%96%D0%B2_%D0%BF%D1%96%D0%B4_%D1%87%D0%B0%D1%81_%D1%80%D0%BE%D1%81%D1%96%D0%B9%D1%81%D1%8C%D0%BA%D0%BE%D0%B3%D0%BE_%D0%B2%D1%82%D0%BE%D1%80%D0%B3%D0%BD%D0%B5%D0%BD%D0%BD%D1%8F_(%D0%B7%D0%B8%D0%BC%D0%B0_2025/2026)")    
    calculate_next_strike()
