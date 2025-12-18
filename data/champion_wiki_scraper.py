import requests
from bs4 import BeautifulSoup
import re
from pprint import pprint
import json
from config import SCRAP_DATA_PATH, CHAMPION_FEATURES, CHAMPION_STATS_FILE
import pandas as pd
import os


def scrape_champion(champion_name: str):
    formatted_name = champion_name.replace(" ", "_").replace("'", "%27")
    url = f"https://wiki.leagueoflegends.com/en-us/{formatted_name}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(
            f"Failed to load page for {champion_name}: Status {response.status_code}"
        )
    soup = BeautifulSoup(response.text, "html.parser")
    champ_info = {}
    # Numeric Stats
    numeric_stats = {}
    for row in soup.select(".infobox.type-champion-stats .infobox-data-row"):
        label_tag = row.select_one(".infobox-data-label.statsbox")
        value_tag = row.select_one(".infobox-data-value.statsbox")
        if label_tag and value_tag:
            label = (
                label_tag.get_text(strip=True).replace("Attack range", "range").lower()
            )
            value_text = value_tag.get_text(strip=True)
            # Extract level 1 number (first number before space, +, or –)
            m = re.search(r"(\d+(\.\d+)?)", value_text)
            numeric_stats[label] = m.group(1) if m else value_text
    champ_info["numeric_stats"] = numeric_stats
    # Style
    style_div = soup.select_one(".champion_style")
    style_number = None
    if style_div:
        # Find the second <span title="..."> which contains the numeric style
        span_titles = style_div.find_all("span", title=True)
        for t in span_titles:
            if t["title"].isdigit():
                style_number = int(t["title"])
                break
    champ_info["style"] = style_number
    # Wheel Stats
    wheel_div = soup.select_one(".stat-wheel")
    wheel_stats = {}
    if wheel_div and wheel_div.has_attr("data-values"):
        for section, value in zip(
            ["damage", "toughness", "control", "mobility", "utility"],
            wheel_div["data-values"].split(";"),
        ):
            try:
                wheel_stats[section] = max(0, int(value) - 1)
            except:
                wheel_stats[section] = 0
    champ_info["wheel_stats"] = wheel_stats
    return champ_info


if __name__ == "__main__":
    while True:
        champ_name = input("Enter champion name: ").strip()
        try:
            data = scrape_champion(champ_name)
            # Save to file
            file_name = f"{SCRAP_DATA_PATH}/{champ_name.replace(' ', '_')}.json"
            os.makedirs(SCRAP_DATA_PATH, exist_ok=True)
            with open(file_name, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Data saved to {file_name}\n")
            champion_stats = {
                feature: data.get(feature)
                or data["numeric_stats"].get(feature)
                or data["wheel_stats"].get(feature)
                for feature in CHAMPION_FEATURES
            }
            pprint(champion_stats)
            # Cast values
            for k, v in list(champion_stats.items()):
                if v is None:
                    champion_stats[k] = 0
                    continue
                try:
                    champion_stats[k] = int(v)
                except Exception:
                    champion_stats[k] = float(v)
            # Save/update champion_stats.csv
            try:
                if os.path.exists(CHAMPION_STATS_FILE):
                    stats_df = pd.read_csv(CHAMPION_STATS_FILE)
                else:
                    stats_df = pd.DataFrame(
                        columns=["champion_name"] + CHAMPION_FEATURES
                    )
                # Prepare row
                row = {"champion_name": champ_name}
                row.update(champion_stats)
                # If champion exists, replace row; else append
                if champ_name in stats_df["champion_name"].values:
                    stats_df = stats_df[stats_df["champion_name"] != champ_name]
                stats_df = pd.concat([stats_df, pd.DataFrame([row])], ignore_index=True)
                # Save
                os.makedirs(os.path.dirname(CHAMPION_STATS_FILE), exist_ok=True)
                stats_df.to_csv(CHAMPION_STATS_FILE, index=False)
                print(f"Champion stats saved/updated in {CHAMPION_STATS_FILE}")
            except Exception as e:
                print(f"Warning: failed to save champion stats to CSV: {e}")
        except Exception as e:
            print(e)
