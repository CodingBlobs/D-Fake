import math
import os
import numpy as np
import time

# For Flask app
from flask import Flask, flash, jsonify, request, redirect, render_template, url_for
from flask_cors import CORS
import base64
from sklearn import feature_extraction

# For Machine Learning Model
from tensorflow import keras
from tensorflow_docs.vis import embed

import numpy as np
import cv2

from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

rootdir = os.getcwd()

# ML Constants
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 200
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
UPLOAD_FOLDER = 'static/uploads/'

# Configurations
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

IMAGES_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
VIDEO_EXTENSIONS = set(['mp4'])

PORT = os.environ.get('PORT', 3000)

#################################################
# Machine Learning Model                    #####
#################################################

reconstructed_model = keras.models.load_model("final.keras")

def _crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def _prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extraction.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask

def _load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = _crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

# predict
def _sequence_prediction(video_path: str):
    frames = _load_video(video_path)
    frame_features, frame_mask = _prepare_single_video(frames)
    return reconstructed_model.predict([frame_features, frame_mask], verbose=0)[0]

def predict(video_path: str) -> bool:
    """Return True if the video is fake, False if the video is real."""
    pred = _sequence_prediction(video_path)[0]
    if(pred>=0.5):
        print(f'The predicted class of the video is FAKE with probability {pred}')
    else:
        print(f'The predicted class of the video is REAL with probability {1-pred}')
    return pred>=0.5

#################################################
# End Machine Learning Model                #####
#################################################


@app.route('/')
def index():
    return render_template('hacknroll24.html')

def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions


@app.route('/verify-video', methods=['POST'])
def verify_video():
    if 'video' not in request.files:
        flash('No video file found')
        return redirect(request.url)
    
    video = request.files['video']
    
    if video.filename == '':
        flash('No selected video file')
        return redirect(request.url)
    
    if video and allowed_file(video.filename, VIDEO_EXTENSIONS):
        filename = secure_filename(video.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Filepath = {filepath}")
        video.save(filepath)
        
        result = predict(filepath)
        
        return jsonify({'result': result})
    
    flash('Invalid file format')
    return redirect(request.url)


if __name__ == '__main__':
    app.run(port=PORT, debug=True)
    