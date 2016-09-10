import random
import cv2, glob
import numpy as np

from skimage import io
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn.svm import SVC

# normalize image
def normalize(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)    
    image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_CUBIC)    
    return image

# Get histogram
def describe(image, mask = None):
    hist = cv2.calcHist([image], [0, 1, 2], mask, [8,8,8], [0, 256, 0, 256, 0, 256])    
    cv2.normalize(hist, hist)    
    return hist.flatten()

data = []
target = []

# Sort through normal images
image_paths = sorted(glob.glob("dataset/normal/*.jpg"))
for image_path in image_paths:    
    # Read image
    image = io.imread(image_path)
    features = describe(normalize(image))

    # Append to dataset
    data.append(features)
    target.append('normal')

# Sort through abnormal images
image_paths = sorted(glob.glob("dataset/abnormal/*.jpg"))
for image_paht in image_paths:
    # Read image
    image = io.imread(image_path)
    features = describe(normalize(image))

    # Append to dataset
    data.append(features)
    target.append('abnormal')

# Get target names
targetNames = np.unique(target)
le = LabelEncoder()
target = le.fit_transform(target)

model = RandomForestClassifier(n_estimators = 33, criterion='entropy')

trainData, testData, trainTarget, testTarget = cross_validation.train_test_split(data, target, test_size=0.4)
model.fit(trainData, trainTarget)
print(classification_report(testTarget, model.predict(testData),
                            target_names = targetNames))

def predict(image):
    global model

    # Get features
    features = describe(normalize(image))

    # Predict it
    result = model.predict([features])
    probability = model.predict_proba([features])[0][result][0]
    state = le.inverse_transform(result)[0]

    return {'type': state, 'confidence': probability} 
