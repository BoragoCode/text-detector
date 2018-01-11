from keras.models import Model
from keras.models import load_model
from keras.layers import *
import numpy as np
import cv2
import sys

# First load the model.
model = load_model("text_detector.hdf5")

# Load the image.
path = sys.argv[1]
img = cv2.imread(path)

# Use model to make prediction.
img = np.expand_dims(img, axis=0)
y_prob = model.predict(img)
result = np.argmax(y_prob, axis=-1)

# Display results.
if result[0] == 1:
    print("Prediction: Text")
else:
    print("Prediction: Not Text")
