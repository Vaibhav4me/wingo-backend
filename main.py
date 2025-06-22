import requests
import json
import numpy as np
from collections import deque
from sklearn.linear_model import LogisticRegression
from fastapi import FastAPI
from pydantic import BaseModel
import threading
import warnings
import os
import time
from tenacity import retry, stop_after_attempt, wait_fixed
warnings.filterwarnings("ignore")

API_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
DATA_FILE = "wingo_big_small_data.json"
WINDOW_SIZE = 3
MAX_DATA_SIZE = 1000
CONFIDENCE_THRESHOLD = 0.6

app = FastAPI()

# Global trained model and lock
trained_model = None
model_lock = threading.Lock()

class PredictionResponse(BaseModel):
    issue: str
    number: int
    size: str
    prediction: str
    confidence: float
    accuracy: float

def get_size(number):
    return "big" if number >= 5 else "small"

def size_to_label(size):
    return 1 if size == "big" else 0

def label_to_size(label):
    return "big" if label == 1 else "small"

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_latest():
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Referer": "https://www.jalwawin1.com/"
        }
        url = f"{API_URL}?ts={int(time.time() * 1000)}"
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        if not data.get('data', {}).get('list'):
            raise ValueError("Invalid API response structure")
        latest = data['data']['list'][0]
        if not all(key in latest for key in ['issueNumber', 'number']):
            raise ValueError("Missing required fields in API response")
        return latest['issueNumber'], int(latest['number']), get_size(int(latest['number']))
    except Exception as e:
        print(f"❌ Error fetching result: {e}")
        return None, None, None

def load_data():
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return []

def save_data(data):
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(data[-MAX_DATA_SIZE:], f)
    except Exception as e:
        print(f"❌ Error saving data: {e}")

def extract_features(window_nums):
    big_small_count = sum(1 for n in window_nums if n >= 5)
    diff = window_nums[-1] - window_nums[-2] if len(window_nums) >= 2 else 0
    avg = sum(window_nums) / len(window_nums)
    return window_nums + [big_small_count, diff, avg]

def prepare_dataset(data):
    X, y = [], []
    nums = [d['number'] for d in data]
    for i in range(len(nums) - WINDOW_SIZE):
        window = nums[i:i+WINDOW_SIZE]
        features = extract_features(window)
        label = size_to_label(get_size(nums[i+WINDOW_SIZE]))
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

def train_model(data):
    global trained_model
    if len(data) < WINDOW_SIZE + 5:
        return
    X, y = prepare_dataset(data)
    if len(X) == 0:
        return
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    with model_lock:
        trained_model = model
    print("🛠️ Model retrained in background.")

def train_model_background(data):
    thread = threading.Thread(target=train_model, args=(data,), daemon=True)
    thread.start()

def predict_with_model(last_numbers):
    global trained_model
    with model_lock:
        model = trained_model
    if model is None or len(last_numbers) < WINDOW_SIZE:
        return None, 0.0
    window_nums = last_numbers[-WINDOW_SIZE:]
    features = extract_features(window_nums)
    input_features = np.array(features).reshape(1, -1)
    pred_label = model.predict(input_features)[0]
    confidence = max(model.predict_proba(input_features)[0])
    if confidence > CONFIDENCE_THRESHOLD:
        return label_to_size(pred_label), confidence
    return None, 0.0

@app.on_event("startup")
async def startup_event():
    data = load_data()
    train_model_background(data)

@app.get("/predict", response_model=PredictionResponse)
async def predict():
    data = load_data()
    last_numbers = deque(maxlen=WINDOW_SIZE)
    for d in data[-WINDOW_SIZE:]:
        last_numbers.append(d['number'])
    
    issue, number, size = fetch_latest()
    if issue:
        data.append({"number": number})
        save_data(data)
        last_numbers.append(number)
        train_model_background(data)
    
    pred, conf = predict_with_model(list(last_numbers))
    accuracy = 0.0  # Simplified; add accuracy tracking if needed
    return {
        "issue": issue or "N/A",
        "number": number or 0,
        "size": size or "N/A",
        "prediction": pred or "Not enough data",
        "confidence": conf * 100,
        "accuracy": accuracy
    }