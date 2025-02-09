from sklearn.svm import SVC
from fastapi import FastAPI
from feature_utils import  compute_features_for_wave, compute_features_for_wave_data, compute_features_for_wave_list, load_and_split_features, preprocess_features
from model_utils import save_model
import numpy as np
import librosa
import joblib
from sklearn.preprocessing import StandardScaler

app = FastAPI()


loaded_data = np.load("features/extracted_features_multiple_test.npz")

feature_selection = {
'mfcc': True,
'delta_mfcc': True,
'hist': True,
'spectral_centroid': True,
'spectral_contrast': True,
'pitch_features': True,
'zcr': True,
'envelope': True,
'hnr': True
}

selected_features, X_train, X_test, y_train, y_test, selected_features_names = load_and_split_features(
    loaded_data, feature_selection
)



X_train, X_test = preprocess_features(
    X_train, X_test, normalize=True, apply_pca=False, n_pca_components=0.9, verbose=True
)

svm = SVC(kernel='rbf', C=1.0, gamma=0.01, probability=True, random_state=42)

# Train the SVM classifier on the training set
svm.fit(X_train, y_train)


save_model(svm, "svm_model_new_2")

model_new = joblib.load("models/svm_model_new_2.joblib")


@app.get("/predict")
def predict_audio():
    try:
        audio_path = "dog6.wav"
        sound_data, sample_rate = librosa.load(audio_path)

        # Extract features
        mfcc_mean, delta_mfcc_mean, hist_features, spectral_centroid, spectral_contrast, pitch_features, zcr_features, envelope_features, hnr_features = compute_features_for_wave(sound_data, sample_rate)

        # Combine features
        combined_features = np.hstack([mfcc_mean, delta_mfcc_mean, hist_features, spectral_centroid, spectral_contrast, pitch_features, zcr_features, envelope_features, hnr_features])

        # Reshape combined_features to 2D array
        combined_features = combined_features.reshape(1, -1)

        # Load and apply scaler
        scaler = joblib.load("models/scaler.joblib")
        combined_features = scaler.transform(combined_features)


        # Predict class
        prediction = model_new.predict(combined_features)
        print(prediction)

        return {"class": "ok"}
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
