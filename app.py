from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import librosa
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model("emotion_recognition_model.h5")

# Emotion labels
emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "surprised", "excited"]

# Home route for frontend
@app.route('/')
def index():
    return render_template("index.html")

# API route for emotion prediction
@app.route('/predict', methods=['POST'])
def predict_emotion():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    try:
        # Load the audio file with librosa
        signal, sr = librosa.load(file, sr=22050)
        
        # Ensure the audio file is of sufficient length
        if len(signal) < sr:  # Less than 1 second
            return jsonify({'error': 'Audio file too short. Please provide a longer file.'}), 400
        
        # Trim or pad the signal to a consistent duration (e.g., 3 seconds)
        max_duration = 3  # seconds
        max_length = sr * max_duration
        if len(signal) > max_length:
            signal = signal[:max_length]
        else:
            signal = np.pad(signal, (0, max_length - len(signal)))

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0).reshape(1, -1)
        
        # Get prediction probabilities
        prediction = model.predict(mfcc_scaled)
        print("Prediction probabilities:", prediction)  # Debugging line
        
        # Get the predicted emotion
        emotion = emotions[np.argmax(prediction)]
        
        # Return the emotion and the probability distribution
        return jsonify({'emotion': emotion, 'probabilities': {emotions[i]: float(prediction[0][i]) for i in range(len(emotions))}})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
