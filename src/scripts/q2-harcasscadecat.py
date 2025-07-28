import cv2
import os
from myproject.path_utils import data_path
# Load the cat‐face cascade
cascade_path = cv2.data.haarcascades + "haarcascade_frontalcatface.xml"
cat_cascade = cv2.CascadeClassifier(cascade_path)

# 2. Helper to test one image
def detect_cat(img_path):
    img   = cv2.imread(img_path)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cats  = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(cats) > 0:
        print(f"[+] Cat detected in {os.path.basename(img_path)} ({len(cats)} face(s))")
    else:
        print(f"[-] No cat detected in {os.path.basename(img_path)}")

# 3. Test on a folder of images
# test_folder = "data/Q2/test/Cat"
test_folder = data_path("Q2", "test", "Cat")
for fname in os.listdir(test_folder):
    if fname.lower().endswith((".jpg", ".png", ".jpeg")):
        detect_cat(os.path.join(test_folder, fname))
    
print("harrascades is not great at diff cat and dogs")