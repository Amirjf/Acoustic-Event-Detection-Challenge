import base64
import io
import shutil
import librosa
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from feature_utils import combine_features_with_flags, compute_features_for_wave, load_and_split_features, preprocess_features
from model_utils import save_model
import numpy as np
import joblib
from pathlib import Path

app = FastAPI()


origins = ['*']


feature_selection = {
     'mfcc': True,
    'delta_mfcc': True,
    'hist':True,
    'spectral_centroid':True,
    'spectral_contrast':True,
    'pitch_features':True,
    'zcr':True,
    'envelope':True,
    'hnr':True
}


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


CATEGORY_MAPPER = {
    0: "dog",
    36: "vacuum_cleaner",
    19: "thunderstorm",
    17: "pouring_water",
    37: "clock_alarm",
    40: "helicopter",
    28: "snoring",
    21: "sneezing",
    1: "rooster",
    42: "siren",
}


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

keys_list = loaded_data['keys'].tolist()  # Convert to list
mfcc_features = loaded_data['mfcc']
delta_mfcc_features = loaded_data['delta_mfcc']
hist_features = loaded_data['hist']
spectral_centroid_features = loaded_data['spectral_centroid']
spectral_contrast_features = loaded_data['spectral_contrast']
pitch_features = loaded_data['pitch_features']
zcr_features = loaded_data['zcr']
envelope_features = loaded_data['envelope']
hnr_features = loaded_data['hnr']

@app.get("/visualize")
def visualize_features():
    # Flatten and concatenate features
    feature_list = [
        mfcc_features, delta_mfcc_features, hist_features, spectral_centroid_features,
        spectral_contrast_features, pitch_features, zcr_features, envelope_features, hnr_features
    ]
    
    # Ensure all features are 2D (samples, feature_dim)
    feature_list = [f.reshape(f.shape[0], -1) for f in feature_list]

    # Concatenate features along the second axis
    feature_matrix = np.hstack(feature_list)  # Shape: (num_samples, total_feature_dim)

    # Apply PCA to reduce to 2D for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(feature_matrix)  # Shape: (num_samples, 2)

    # Get unique labels
    unique_labels = list(set(keys_list))

    # Convert to JSON format
    data_points = [
        {"x": float(reduced_features[i, 0]), "y": float(reduced_features[i, 1]), "label": keys_list[i]}
        for i in range(len(keys_list))
    ]

    return {"data": data_points, "labels": unique_labels}


@app.post("/train")
def train_model(request: TrainingRequest):
    try:
        # Load and split data based on feature selection
        selected_features, X_train, X_test, y_train, y_test, selected_features_names = load_and_split_features(
            loaded_data, feature_selection 
        )

        # Preprocess features
        X_train, X_test = preprocess_features(
            X_train, X_test, normalize=request.normalize,
            apply_pca=request.apply_pca, n_pca_components=request.n_pca_components, verbose=True
        )

        # Train SVM model

        svm = SVC(kernel='rbf', C=request.C, gamma=request.gamma, probability=True, random_state=42)

        svm.fit(X_train, y_train)


#         knn = KNeighborsClassifier(n_neighbors=7, weights='uniform', metric='euclidean')
#         knn.fit(X_train, y_train)

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
        predicted_target = prediction.tolist()[0]

        # Map the predicted target to its category name
        category_name = CATEGORY_MAPPER.get(predicted_target, "Unknown")

        return {"predicted_class": category_name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
def get_models():
    models = [f.stem for f in MODEL_DIR.glob("*.joblib")]
    return models

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
