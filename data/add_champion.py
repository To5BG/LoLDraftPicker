import pandas as pd
from config import CHAMPION_STATS_FILE, CHAMPION_FEATURES


def add_champion():
    # Load existing data
    try:
        df = pd.read_csv(CHAMPION_STATS_FILE)
    except FileNotFoundError:
        # Create new dataframe if file doesn't exist
        df = pd.DataFrame(columns=["champion_name"] + CHAMPION_FEATURES)

    print("=" * 60)
    print("Add Champion to Stats Database")
    print("=" * 60)
    print("\nEnter champion information:\n")
    print()

    # Get champion name
    champion_name = input("Champion name: ").strip()
    if champion_name in df["champion_name"].values:
        print(f"\nChampion '{champion_name}' already exists!")
        overwrite = input("Overwrite existing entry? (y/n): ").strip().lower()
        if overwrite != "y":
            print("Cancelled.")
            return
        # Remove existing entry
        df = df[df["champion_name"] != champion_name]
    # Collect feature values
    features = {"champion_name": champion_name}
    for feature in CHAMPION_FEATURES:
        while True:
            try:
                value = input(f"{feature}: ").strip()
                # Validation
                if feature == "style":
                    val = int(value)
                    if 0 <= val <= 100:
                        features[feature] = val
                        break
                    else:
                        print("  Error: Must be between 0 and 100")
                elif feature in [
                    "damage",
                    "toughness",
                    "control",
                    "mobility",
                    "utility",
                ]:
                    val = int(value)
                    if val in [0, 1, 2]:
                        features[feature] = val
                        break
                    else:
                        print("  Error: Must be 0, 1, or 2")
                else:  # pointclick_cc, skillshot_cc
                    val = int(value)
                    if val >= 0:
                        features[feature] = val
                        break
                    else:
                        print("  Error: Must be non-negative integer")
            except ValueError:
                print("  Error: Invalid input. Please enter a number.")
    # Add to dataframe
    new_row = pd.DataFrame([features])
    df = pd.concat([df, new_row], ignore_index=True)
    # Save to CSV
    df.to_csv(CHAMPION_STATS_FILE, index=False)
    print(f"\nChampion '{champion_name}' added successfully!")
    print(f"Total champions in database: {len(df)}")


if __name__ == "__main__":
    try:
        add_champion()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
