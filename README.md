# LoL Draft Picker - AI Champion Recommendation System

A two-stage machine learning model for predicting optimal champion picks in League of Legends.
Currently only for ADC picks.

## Architecture

### Stage 1: Champion Embedding Model

- **Input**: Hard champion parameters (range, mobility, tankiness, etc.)
- **Output**: Learned embedding vector (2+ dimensions)

### Stage 2: Draft Picker Model

- **Input**: Embeddings of team champions, support, and visible enemies
- **Output**: Predicted optimal embedding for best pick

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare your data:
   - Add champion stats to `data/champion_stats.csv`
   - Add draft history to `data/draft_history.json`

## Training

### Stage 1: Train Champion Embeddings

```bash
python train_embedding.py
```

- Load champion stats
- Train the embedding model
- Save embeddings for all champions

### Stage 2: Train Draft Picker

```bash
python train_picker.py
```

- Load champion embeddings from Stage 1
- Train the draft prediction model
- Save the trained picker model

## Inference

```bash
python inference.py
```

Or use programmatically:

```python
from inference import DraftPredictor

predictor = DraftPredictor()

# Get single best pick
best_pick, confidence = predictor.predict(
    my_team=['Malphite', 'Zed', 'Jarvan IV'],
    my_support='Thresh',
    enemy_team=['Caitlyn', 'Lux']
)

# Get top 5 recommendations
top_picks = predictor.get_top_k_recommendations(
    my_team=['Malphite', 'Zed', 'Jarvan IV'],
    my_support='Thresh',
    enemy_team=['Caitlyn', 'Lux'],
    k=5
)
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
