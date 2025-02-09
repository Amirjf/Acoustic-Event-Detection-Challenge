import shutil
import librosa
from sklearn.svm import SVC
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from feature_utils import compute_features_for_wave, load_and_split_features, preprocess_features
from model_utils import save_model
import numpy as np
import joblib
from pathlib import Path


app = FastAPI()


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FEATURES_FILE = "features/extracted_features_multiple_test.npz"

# Load extracted features
loaded_data = np.load(FEATURES_FILE)


class TrainingRequest(BaseModel):
    feature_selection: dict
    model_name: str
    normalize: bool = True
    apply_pca: bool = False
    n_pca_components: float = 0.9
    C: float = 1.0
    gamma: float = 0.01

class PredictRequest(BaseModel):
    model_name: str


loaded_data = np.load("features/extracted_features_multiple_test.npz")


@app.post("/train")
def train_model(request: TrainingRequest):
    try:
        # Load and split data based on feature selection
        selected_features, X_train, X_test, y_train, y_test, selected_features_names = load_and_split_features(
            loaded_data, request.feature_selection
        )

        # Preprocess features
        X_train, X_test = preprocess_features(
            X_train, X_test, normalize=request.normalize,
            apply_pca=request.apply_pca, n_pca_components=request.n_pca_components, verbose=True
        )

        # Train SVM model
        svm = SVC(kernel='rbf', C=request.C, gamma=request.gamma, probability=True, random_state=42)
        svm.fit(X_train, y_train)

        # Save trained model
        save_model(svm, request.model_name, directory=str(MODEL_DIR))

        return {"message": f"Model {request.model_name} trained and saved successfully!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/predict")
def predict_audio(model_name: str, file: UploadFile = File(...)):
    try:
        # Ensure model exists
        model_path = MODEL_DIR / f"{model_name}.joblib"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

        # Save uploaded file temporarily
        temp_audio_path = f"temp/{file.filename}"
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)

        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load audio
        sound_data, sample_rate = librosa.load(temp_audio_path)

        # Extract features
        mfcc_mean, delta_mfcc_mean, hist_features, spectral_centroid, spectral_contrast, pitch_features, zcr_features, envelope_features, hnr_features = compute_features_for_wave(sound_data, sample_rate)

        # Combine features
        combined_features = np.hstack([
            mfcc_mean, delta_mfcc_mean, hist_features, spectral_centroid,
            spectral_contrast, pitch_features, zcr_features, envelope_features, hnr_features
        ])

        # Reshape combined_features to 2D array
        combined_features = combined_features.reshape(1, -1)

        # Load and apply scaler
        scaler_path = MODEL_DIR / "scaler.joblib"
        if not scaler_path.exists():
            raise HTTPException(status_code=404, detail="Scaler not found")
        
        scaler = joblib.load(scaler_path)
        combined_features = scaler.transform(combined_features)

        # Load model and predict
        model = joblib.load(model_path)
        prediction = model.predict(combined_features)

        return {"class": prediction.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
