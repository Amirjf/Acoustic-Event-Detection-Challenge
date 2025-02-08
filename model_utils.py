import joblib
from pathlib import Path

def save_model(model, model_name, directory="models"):
    models_dir = Path(directory)
    models_dir.mkdir(exist_ok=True)

    joblib.dump(model, models_dir / f"{model_name}.joblib")

