import glob
import os
import cv2
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
from myproject.path_utils import data_path
warnings.filterwarnings('ignore')

# ========================
# 1. DATA LOADING
# ========================
features = []
labels = []

# Paths
# train_dir = "data/Q2/train"
# test_dir  = "data/Q2/test"
train_dir = data_path("Q2","train")
test_dir  = data_path("Q2","test")

# Gather all image file paths
image_paths = glob.glob(os.path.join(train_dir, "*", "*")) + \
              glob.glob(os.path.join(test_dir, "*", "*"))

print(f"[INFO] Found {len(image_paths)} images in train/test folders.")

# Process each image
for i, path in enumerate(image_paths):
    img = cv2.imread(path)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = img.flatten()
    
    features.append(img)
    
    # Label is the parent folder name: 'cats' or 'dogs'
    label = os.path.basename(os.path.dirname(path))
    labels.append(label)
    
    if i % 200 == 0:
        print(f"[INFO] Processed {i} images")

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# ========================
# 2. TRAIN/TEST SPLIT
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========================
# 3. MODEL TRAINING
# ========================
# Using KNN classifier per fire_detection.py format
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "cat_dog_classifier.z")
print("[INFO] Model saved as cat_dog_classifier.z")

# ========================
# 4. EVALUATION
# ========================
preds = model.predict(X_test)
print(f"Accuracy Score: {accuracy_score(y_test, preds):.4f}")

# ========================
# 5. SAMPLE INFERENCE
# ========================
def predict_image(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = img.flatten().reshape(1, -1)
    return model.predict(img)[0]

# Example usage:
print(predict_image(data_path("Q2", "test", "Dog", "Dog (1).jpg"), model))
print(predict_image(data_path("Q2", "test", "Dog", "Dog (2).jpg"), model))
print(predict_image(data_path("Q2", "test", "Dog", "Dog (3).jpg"), model))
print(predict_image(data_path("Q2", "test", "Dog", "Dog (4).jpg"), model))
print(predict_image(data_path("Q2", "test", "Dog", "Dog (5).jpg"), model))
print(predict_image(data_path("Q2", "test", "Cat", "Cat (1).jpg"), model))
print(predict_image(data_path("Q2", "test", "Cat", "Cat (2).jpg"), model))
print(predict_image(data_path("Q2", "test", "Cat", "Cat (3).jpg"), model))
print(predict_image(data_path("Q2", "test", "Cat", "Cat (4).jpg"), model))
print(predict_image(data_path("Q2", "test", "Cat", "Cat (5).jpg"), model))
print("Knn bad accuracy score and not the best at diff b/w cat and dog")

# print(predict_image("data/Q2/test/Dog/Dog (2).jpg", model))
# print(predict_image("data/Q2/test/Dog/Dog (3).jpg", model))
# print(predict_image("data/Q2/test/Dog/Dog (4).jpg", model))
# print(predict_image("data/Q2/test/Dog/Dog (5).jpg", model))
# print(predict_image("data/Q2/test/Cat/Cat (1).jpg", model))
# print(predict_image("data/Q2/test/Cat/Cat (2).jpg", model))
# print(predict_image("data/Q2/test/Cat/Cat (3).jpg", model))
# print(predict_image("data/Q2/test/Cat/Cat (4).jpg", model))
# print(predict_image("data/Q2/test/Cat/Cat (5).jpg", model))

