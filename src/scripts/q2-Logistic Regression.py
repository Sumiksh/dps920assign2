import warnings
warnings.filterwarnings('ignore')

import os
import glob
import cv2
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from myproject.path_utils import data_path

# 1. DATA LOADING
features = []
labels = []

# Adjust this path to your Q2 folder root
data_dirs = [
    data_path("Q2","train"),
    data_path("Q2","test")
]

# Collect all image paths under cats/ and dogs/ subfolders
image_paths = []
for d in data_dirs:
    image_paths.extend(glob.glob(os.path.join(d, "*", "*")))

print(f"[INFO] Found {len(image_paths)} images in specified folders.")

# Process each image
for i, path in enumerate(image_paths):
    img = cv2.imread(path)
    if img is None:
        continue
    
    # Resize to 64x64, normalize, and flatten
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = img.flatten()
    features.append(img)
    
    # Label from parent folder name: 'cats' or 'dogs'
    label = os.path.basename(os.path.dirname(path))
    labels.append(label)
    
    if i % 200 == 0 and i > 0:
        print(f"[INFO] Processed {i}/{len(image_paths)} images")

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)


# 2. TRAIN/TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


# 3. MODEL TRAINING
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "cat_dog_classifier_lr.z")
print("[INFO] Model saved as cat_dog_classifier_lr.z")

# 4. EVALUATION
preds = model.predict(X_test)
print(f"[RESULT] Accuracy Score: {accuracy_score(y_test, preds):.4f}")

# 5. SAMPLE INFERENCE
def predict_image(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
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
print("Logistic Regression is better than KNN and harrcasscade for diff b/w cat and dogs")
