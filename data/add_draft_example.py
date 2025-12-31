import json
import pandas as pd
import os
from config import CHAMPION_STATS_FILE, DRAFT_HISTORY_FILE


def load_champion_names():
    try:
        df = pd.read_csv(CHAMPION_STATS_FILE)
        return set(df["champion_name"].values)
    except Exception:
        return set()


def load_draft_history():
    if not os.path.exists(DRAFT_HISTORY_FILE):
        return []
    with open(DRAFT_HISTORY_FILE, "r") as f:
        return json.load(f)


def save_draft_history(data):
    os.makedirs(os.path.dirname(DRAFT_HISTORY_FILE), exist_ok=True)
    with open(DRAFT_HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_champion_input(prompt, champion_names, allow_blank=True):
    while True:
        value = input(prompt).strip()
        if not value and allow_blank:
            return ""
        if not value and not allow_blank:
            print("  Error: This field is required.")
            continue
        if value in champion_names:
            return value
        print(f"  Error: Champion '{value}' not found in database.")
        print(
            f"  Available champions: {', '.join(sorted(list(champion_names))[:10])}..."
        )


def main():
    print("=" * 60)
    print("Add Draft Example to History")
    print("=" * 60)
    # Load available champions
    champion_names = load_champion_names()
    if not champion_names:
        print("\nError: No champions found in database.")
        print("Please add champions first using add_champion.py")
        return
    print(f"\nLoaded {len(champion_names)} champions from database.")
    print("\nEnter draft scenario information:")
    print("(Leave blank if champion not picked)\n")
    # Define all input fields
    fields = [
        ("enemy_adc", "Enemy ADC: "),
        ("enemy_support", "Enemy Support: "),
        ("my_support", "My Support: "),
        ("teammate_1", "Teammate 1: "),
        ("teammate_2", "Teammate 2: "),
        ("teammate_3", "Teammate 3: "),
        ("enemy_1", "Other Enemy 1: "),
        ("enemy_2", "Other Enemy 2: "),
        ("enemy_3", "Other Enemy 3: "),
    ]
    example = {}
    # Collect draft state
    for key, prompt in fields:
        example[key] = get_champion_input(prompt, champion_names)
    # Optimal pick (required)
    print()
    example["optimal_pick"] = get_champion_input(
        "Optimal Pick (required): ", champion_names, allow_blank=False
    )
    # Load existing history, append, and save
    data = load_draft_history()
    data.append(example)
    save_draft_history(data)
    print(f"\nDraft example added successfully!")
    print(f"Total examples in database: {len(data)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
