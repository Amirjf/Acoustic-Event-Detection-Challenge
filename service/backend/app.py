from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import pickle
from typing import Dict
import json

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model and feature parameters
model = None  # Load your trained KNN model here

feature_selection = {
    'mfcc': True,
    'hist': False,
    'spectral': True,
    'zcr': False,
    'envelope': True,
    'hnr': True
}

def extract_features(audio_file) -> Dict:
    """Extract features from audio file using the same parameters as training"""
    # Load audio file
    y, sr = librosa.load(audio_file, sr=22050)
    
    features = {}
    
    if feature_selection['mfcc']:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc'] = np.mean(mfcc.T, axis=0)
        
    if feature_selection['spectral']:
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral'] = np.mean(spectral_centroids)
        
    if feature_selection['envelope']:
        envelope = np.abs(y)
        features['envelope'] = np.mean(envelope)
        
    if feature_selection['hnr']:
        hnr = librosa.effects.harmonic(y)
        features['hnr'] = np.mean(hnr)
    
    # Combine features in the same order as training
    combined_features = np.concatenate([features[f] for f, include in feature_selection.items() if include])
    return combined_features

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with open("temp_audio.wav", "wb") as buffer:
        buffer.write(await file.read())
    
    # Extract features
    features = extract_features("temp_audio.wav")
    
    # Make prediction
    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0]
    
    # Get top 3 predictions with probabilities
    top_3_idx = np.argsort(probabilities)[-3:][::-1]
    results = [
        {
            "class": str(idx),
            "probability": float(probabilities[idx])
        }
        for idx in top_3_idx
    ]
    
    return {
        "prediction": str(prediction),
        "confidence": float(probabilities.max()),
        "top_3": results
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}