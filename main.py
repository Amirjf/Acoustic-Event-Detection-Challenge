from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import json
import shutil
from enum import Enum
from pydantic import BaseModel
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import librosa
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
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
feee = {
    'mfcc': True,
    'delta_mfcc': True,
      'histogram': True, 'spectral_centroid': True, 'spectral_contrast': True, 'pitch': True, 'zcr': True, 'envelope': True, 'hnr': True}


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



class ModelType(str, Enum):
    SVM = "svm"
    RANDOM_FOREST = "random_forest"
    KNN = "knn"

class TrainingRequest(BaseModel):
    feature_selection: dict
    model_name: str
    model_type: ModelType
    normalize: bool = True
    apply_pca: bool = False
    n_pca_components: float = 0.9
    # Model specific parameters
    C: float = 1.0  # for SVM
    gamma: float = 0.01  # for SVM
    n_estimators: int = 100  # for Random Forest
    max_depth: int = 10  # for Random Forest
    n_neighbors: int = 3  # for KNN

@app.post("/train")
def train_model(request: TrainingRequest):

    print(request.feature_selection)
    metrics = {}
    
    try:
        # Load and split data based on feature selection
        selected_features, X_train, X_test, y_train, y_test, selected_features_names = load_and_split_features(
            loaded_data, feature_selection=request.feature_selection  
        )
        
        # Preprocess features
        X_train, X_test = preprocess_features(
            X_train, X_test, normalize=request.normalize,
            apply_pca=request.apply_pca, n_pca_components=request.n_pca_components, verbose=True
        )

        # Initialize the requested model
        if request.model_type == ModelType.SVM:
            model = SVC(kernel='rbf', C=request.C, gamma=request.gamma, 
                       probability=True, random_state=42)
        elif request.model_type == ModelType.RANDOM_FOREST:
            model = RandomForestClassifier(n_estimators=request.n_estimators, 
                                         max_depth=request.max_depth, 
                                         min_samples_split=10, random_state=42)
        else:  # KNN
            model = KNeighborsClassifier(n_neighbors=request.n_neighbors)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        y_train_pred = model.predict(X_train)

        # Calculate metrics
        metrics[request.model_type] = {
            'test_accuracy': float(accuracy_score(y_test, y_pred)),
            'test_auc': float(roc_auc_score(y_test, y_prob, multi_class='ovr')),
            'train_accuracy': float(accuracy_score(y_train, y_train_pred)),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        # Save metrics
        with open(MODEL_DIR / f"{request.model_name}_metrics.json", 'w') as f:
            json.dump(metrics, f)

        # Save trained model
        save_model(model, request.model_name, directory=str(MODEL_DIR))

        return {
            "message": f"Successfully trained and saved {request.model_type} model",
            "metrics": metrics
        }

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


@app.get("/model_metrics/{model_name}")
def get_model_metrics(model_name: str):
    try:
        metrics_path = MODEL_DIR / f"{model_name}_metrics.json"
        if not metrics_path.exists():
            raise HTTPException(status_code=404, detail=f"Metrics for model '{model_name}' not found")
            
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/models")
def get_models():
    models = [f.stem for f in MODEL_DIR.glob("*.joblib")]
    return models

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
