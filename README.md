# LoL Draft Picker

AI-powered League of Legends draft pick recommendation system using a two-stage deep learning pipeline.

## Pipeline Overview

1. **Add Champions**

   - Use `add_champion.py` to interactively add or update champion stats in `data/champion_stats.csv`.

2. **Prepare Draft History**

   - Add draft scenarios to `data/draft_history_example.json` (see format below).
   - Each entry specifies the visible draft state and the optimal pick.

3. **Train Models**

   - **Stage 1:** Train champion embeddings:
     ```bash
     python -m train.train_embedding
     ```
   - **Stage 2:** Train the draft picker model:
     ```bash
     python -m train.train_picker
     ```

## Inference Usage

**Interactive:**

```bash
python main.py
```

**Programmatic:**

```python
from inference import DraftPredictor
predictor = DraftPredictor()
best_pick, confidence = predictor.predict(
        enemy_adc="Caitlyn",
        enemy_support="Lux",
        my_support="Thresh",
        teammate_1="Malphite",
        teammate_2="Zed",
        teammate_3="Jarvan IV",
        enemy_1="Yasuo",
        enemy_2="Lee Sin",
        enemy_3="Amumu"
)
print(f"Recommended: {best_pick} (confidence: {confidence:.2%})")
```

## Configuration

Edit `config.py` to adjust:

- Champion feature set
- Embedding dimensions
- Model architecture (hidden layers)
- Training hyperparameters
- File paths

## Next Steps

1. Collect real champion stats data
2. Gather historical draft data
3. Tune hyperparameters
4. Experiment with different embedding dimensions
5. Add more features
