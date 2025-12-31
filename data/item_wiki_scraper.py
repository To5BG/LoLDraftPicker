import requests
from bs4 import BeautifulSoup
import re
from pprint import pprint
import json
import pandas as pd
import os
import time
from urllib.parse import urljoin
from config import *

# User-Agent header to avoid 403 errors
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def fetch_items():
    response = requests.get("https://u.gg/lol/items", headers=HEADERS)
    if response.status_code != 200:
        raise Exception(f"Failed to load page for items: Status {response.status_code}")
    soup = BeautifulSoup(response.content, "html.parser")
    # Get legendary items (last container)
    items_container = soup.find_all("div", class_="items-container")[-1]
    item_names = []
    for link in items_container.find_all("a", href=re.compile(r"/lol/items/")):
        # Extract item name
        item_name = link.find("img").get("alt", "").split("/lol/items/")[-1]
        item_names.append(item_name.replace(" ", "_"))
    print(f"Found {len(item_names)} items")
    return item_names


def scrape_item_stats(item_name):
    formatted_name = item_name.replace("'", "%27").replace("The", "the")
    url = f"https://wiki.leagueoflegends.com/en-us/{formatted_name}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(
            f"Failed to load page for {item_name}: Status {response.status_code}"
        )
    soup = BeautifulSoup(response.content, "html.parser")
    item_data = {}
    # Find the base stats tab
    base_tab = soup.find("div", class_="tabbertab", attrs={"data-title": "Base"})
    if not base_tab:
        base_tab = soup.find("div", class_="infobox-section-stacked")
    stat_rows = base_tab.find_all("div", class_="infobox-data-row")
    for row in stat_rows:
        value_div = row.find("div", class_="infobox-data-value")
        # Parse stat text
        stat_text = value_div.get_text(strip=True)
        # Try to extract numeric value and stat name
        match = re.match(r"([+\-]?(\d+)%?)\s*(.+)", stat_text)
        if match:
            value = match.group(2)
            stat_name = match.group(3).lower().strip().replace(" ", "_")
            # Excl. items with "per" or "or" in name -> ignored stats
            if stat_name.find("per") != -1 or stat_name.find("or") != -1:
                continue
            # Try to convert to float/int
            if "%" in value:
                value = float(value.replace("%", ""))
            elif "." in value:
                value = float(value)
            else:
                value = int(value)
            item_data[stat_name] = value
    pprint(item_data)
    return item_data


def save_item_data(item_name, data):
    # Save scraped data
    file_name = f"{SCRAP_ITEM_DATA_PATH}/{item_name}.json"
    os.makedirs(SCRAP_ITEM_DATA_PATH, exist_ok=True)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data saved to {file_name}")
    # Save/update item_stats.csv
    try:
        if os.path.exists(ITEM_STATS_FILE):
            stats_df = pd.read_csv(ITEM_STATS_FILE)
        else:
            stats_df = pd.DataFrame(columns=["item_name"] + ITEM_STATS)
        # Prepare row
        row = {
            "item_name": item_name,
        }
        row.update(item_data)
        for stat in ITEM_STATS:
            if stat not in row:
                row[stat] = 0
        # If item exists, replace row; else append
        if item_name in stats_df["item_name"].values:
            stats_df = stats_df[stats_df["item_name"] != item_name]
        stats_df = pd.concat([stats_df, pd.DataFrame([row])], ignore_index=True)
        # Save
        os.makedirs(os.path.dirname(ITEM_STATS_FILE), exist_ok=True)
        stats_df.to_csv(ITEM_STATS_FILE, index=False)
        print(f"Item stats saved/updated in {ITEM_STATS_FILE}")
    except Exception as e:
        print(f"Warning: failed to save item stats to CSV: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("League of Legends Item Scraper")
    print("=" * 60)
    # Fetch items
    item_names = fetch_items()
    if not item_names:
        print("No items found. Exiting.")
        exit()
    # Store item names
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ITEM_NAMES_FILE, "w", encoding="utf-8") as f:
        for name in sorted(item_names):
            f.write(f"{name}\n")
    print(f"Saved {len(item_names)} item names to {ITEM_NAMES_FILE}")
    # Scrape item stats
    print(f"\nScraping detailed stats for {len(item_names)} items...")
    for i, item_name in enumerate(item_names, 1):
        print(f"\n[{i}/{len(item_names)}] Processing {item_name}...")
        try:
            item_data = scrape_item_stats(item_name)
            item_name = item_name.replace("_", "").replace("'", "").lower()
            save_item_data(item_name, item_data)
        except Exception as e:
            print(e)
    print(f"Scraping complete!")
