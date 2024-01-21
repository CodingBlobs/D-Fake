# D-Fake

## Folder Structure

```
├── data
│   ├── input
│       ├── deepfake_faces
│       │   └── faces_224
│       └── deepfake_videos
│           ├── test_videos
|           |   └── ...videos
│           └── train_sample_videos
|               ├── ...videos
|               └── metadata.json
├── static
│   ├── flatblob.png
│   ├── headerblob.png
│   └── script.js
├── templates
    └── hacknroll24.html
├── app.py
├── final.keras
├── model3.ipynb
├── README.md
└── requirements.txt
```

## Getting Started
```shell
# virtual env
python3.10 -m venv env
source env/bin/activate

# install requirements
pip install -r "requirements"

# run flask app
python app.py
```

## Reference

Data taken from: https://www.kaggle.com/competitions/deepfake-detection-challenge/overview
