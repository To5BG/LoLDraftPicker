from inference import DraftPredictor


def get_champion_input(prompt):
    """Get champion name input (allows empty)"""
    value = input(prompt).strip()
    return value if value else None


def main():
    print("=" * 60)
    print("LoL Draft Picker - AI Recommendation System")
    print("=" * 60)
    print("\nLoading models...")
    try:
        predictor = DraftPredictor()
    except Exception as e:
        print(f"Error loading models: {e}")
        print("\nMake sure you have trained models.")
        return
    print("Models loaded successfully!\n")
    print("=" * 60)
    print("Enter draft information")
    print("(Leave empty if champion not yet picked)")
    print("=" * 60)
    print()
    # Collect draft state
    my_support = get_champion_input("My Support: ")
    enemy_support = get_champion_input("Enemy Support: ")
    enemy_adc = get_champion_input("Enemy ADC: ")
    print("\nOther teammates:")
    teammate_1 = get_champion_input("  Teammate 1: ")
    teammate_2 = get_champion_input("  Teammate 2: ")
    teammate_3 = get_champion_input("  Teammate 3: ")
    print("\nOther enemies:")
    enemy_1 = get_champion_input("  Enemy 1: ")
    enemy_2 = get_champion_input("  Enemy 2: ")
    enemy_3 = get_champion_input("  Enemy 3: ")
    # Get available champions (optional filter)
    print("\n" + "-" * 60)
    filter_available = input("Filter by available champions? (y/n): ").strip().lower()
    available_champions = None
    if filter_available == "y":
        print("\nEnter available champion names (comma-separated):")
        available_input = input("> ").strip()
        if available_input:
            available_champions = [c.strip() for c in available_input.split(",")]
    # Prepare input for predictor
    print("\n" + "=" * 60)
    print("Analyzing draft...")
    print("=" * 60)
    try:
        # Get recommendation
        best_pick, confidence = predictor.predict(
            my_support=my_support,
            enemy_support=enemy_support,
            enemy_adc=enemy_adc,
            teammate_1=teammate_1,
            teammate_2=teammate_2,
            teammate_3=teammate_3,
            enemy_1=enemy_1,
            enemy_2=enemy_2,
            enemy_3=enemy_3,
            available_champions=available_champions,
        )
        print(f"\nRecommended Pick: {best_pick}")
        print(f"   Confidence: {confidence:.1%}")
        # Get top 5 recommendations
        print("\nTop 5 Recommendations:")
        top_picks = predictor.get_top_k_recommendations(
            my_support=my_support,
            enemy_support=enemy_support,
            enemy_adc=enemy_adc,
            teammate_1=teammate_1,
            teammate_2=teammate_2,
            teammate_3=teammate_3,
            enemy_1=enemy_1,
            enemy_2=enemy_2,
            enemy_3=enemy_3,
            k=5,
            available_champions=available_champions,
        )
        for i, (champ, score) in enumerate(top_picks, 1):
            print(f"  {i}. {champ:20s} (score: {score:.3f})")
        print("\n" + "=" * 60)
    except Exception as e:
        print(f"\nError during prediction: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()
