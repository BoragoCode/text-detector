from keras.models import Model
from keras.models import load_model
from keras.layers import *
import numpy as np
from keras import optimizers
import sys
import cv2

# First load the model.
model = load_model('text_detector.hdf5')

# Load image.
path = sys.argv[1]
img = cv2.imread(path)
img_copy = img

# Get image shape.
height, width, channels = img.shape

# Define kernel dimensions.
kernel_w = 100
kernel_h = 60

# Define steps.
x_step = 50
y_step = 30

# Calculate limits.
x_len = int((width-kernel_w)/x_step)
y_len = int((height-kernel_h)/y_step)

# Now segment image into kernel pieces.
for y in range(0, y_len):
    for x in range(0, x_len):
        # Calculate origins and destination points of kernel.
        y_origin = y*y_step
        y_dest = y_origin+kernel_h
        x_origin = x*x_step
        x_dest = x_origin+kernel_w

        # Extract image segment currently being viewed by the kernel.
        img_segment = img_copy[y_origin:y_dest, x_origin:x_dest]

        # Make prediction.
        predict_segment = np.expand_dims(img_segment, axis=0)
        y_prob = model.predict(predict_segment)
        result = np.argmax(y_prob, axis=-1)

        # If prediction says there is text draw red rectangle around this section.
        if result[0] == 1:
            img = cv2.rectangle(img,(x_origin, y_origin), (x_dest,y_dest), (0,0,255))

# If you wish you wish to see the image without resizing it uncomment next line.
img = cv2.resize(img, (720,720))

# Display image.
while True:
  cv2.imshow("Press Enter to Exit", img)
  if cv2.waitKey(1) == 13:
    break
