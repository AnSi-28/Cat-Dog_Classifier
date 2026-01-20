import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import joblib

# ---------------- CONFIG 

DATASET_DIR = "PetImages"
IMG_SIZE = 32
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
X = []
y = []
MAX_IMAGES = 3000
def load_images(folder, label):
    count=0
    for img_name in tqdm(os.listdir(folder)):
        if count >= MAX_IMAGES // 2:
            break
        try:
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.flatten()
            X.append(img)
            y.append(label)
            count += 1
        except:
            pass

load_images(os.path.join(DATASET_DIR, "Cat"), 0)
load_images(os.path.join(DATASET_DIR, "Dog"), 1)

X = np.array(X) / 255.0
y = np.array(y)

print("Dataset loaded:", X.shape, y.shape)

# ---------------- SPLIT DATA ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- SVM ----------------
svm = LinearSVC(max_iter=5000)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
joblib.dump(svm, f"{MODELS_DIR}/svm_model.pkl")

# ---------------- RANDOM FOREST ----------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
joblib.dump(rf, f"{MODELS_DIR}/rf_model.pkl")

# ---------------- LOGISTIC REGRESSION ----------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
joblib.dump(lr, f"{MODELS_DIR}/lr_model.pkl")

# ---------------- K-MEANS ----------------
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_train)
joblib.dump(kmeans, f"{MODELS_DIR}/kmeans_model.pkl")

print("All models trained and saved successfully.")