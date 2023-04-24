import os
import random
import difflib
import librosa
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3'}

# data = pd.read_csv("./music_features.csv")

# y = data["label"].unique()

# data = []
labels = []
categories = {}
parent_paths = []

current_folder = f"{os.getcwd()}/MusicDataset"

for parent in os.listdir(current_folder):
    parent_path = os.path.join(current_folder, parent)
    parent_paths.append(parent_path)

for parent_path in parent_paths:
    parent = parent_path.split("/")[-1]
    if os.path.isdir(parent_path):
        for child in os.listdir(parent_path):
            child_path = os.path.join(parent_path, child)

            if os.path.isdir(child_path):
                for file in os.listdir(child_path):
                    file_path = os.path.join(child_path, file)

                    if file_path.endswith('.mp3'):
                        # features = extract_features(file_path)
                        categories[parent] = categories.get(parent, 0) + 1
                        # data.append(features)
                        labels.append(parent)


scaler = StandardScaler()
label_encoder = LabelEncoder()
y = np.array(labels)
label_encoder.fit(y)

def predict_music_style(file_path):
    features = extract_features(file_path)
    features_scaled = scaler.fit_transform([features])
    prediction = model.predict(features_scaled)
    print(prediction)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return predicted_label

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


data = []
minimum_songs = 7
other_style = 'other_style'
categories = {}
current_folder = f"{os.getcwd()}/MusicDataset"

model_path = 'model.h5'
model = load_model(model_path)

def get_music_samples(style, num_samples=5):
    # Get a list of folder names in MusicDataset
    folder_names = [folder for folder in os.listdir(current_folder) if os.path.isdir(os.path.join(current_folder, folder))]
    
    # Find the best match for the requested style
    best_match = difflib.get_close_matches(style, folder_names, n=1, cutoff=0.8)
    
    if not best_match:
        return []

    style_path = os.path.join(current_folder, best_match[0])
    
    samples = []
    for subdir in os.listdir(style_path):
        subdir_path = os.path.join(style_path, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith('.mp3'):
                    samples.append({
                        'title': file[:-4],
                        'artist': '',  # Add artist information if available
                        'audio': os.path.join(best_match[0], subdir, file)
                    })
    return random.sample(samples, min(num_samples, len(samples)))


def extract_features(file_path, n_mfcc=20, n_mels=128):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mel_mean = np.mean(mel.T, axis=0)
    
    features = np.concatenate((mfccs_mean, mel_mean))
    return features


app = Flask(__name__)
CORS(app, supports_credentials=True, origins="*", methods=['GET', 'POST'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/api/examples', methods=['GET'])
def api_fetch_samples():
    try:
        style = request.args.get('style')
        if not style:
            return jsonify({'error': 'Missing style parameter'}), 400
        
        samples = get_music_samples(style)
        if not samples:
            return jsonify({'error': 'No matching music style found'}), 404
        
        return jsonify({'samples': samples})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

    
@app.route('/api/upload', methods=['POST'])
def api_upload_file():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    print("I am here")

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            predicted_style = predict_music_style(file_path)
            return jsonify({'style': predicted_style})
        except Exception as e:
            print(e)
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)