import cv2 as cv
import numpy as np
from tensorflow.keras import models

class_names = [
    "Plane", "Car", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

model = models.load_model("image_classifier.keras")

img = cv.imread("car.jpg")

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (32, 32))

img = img / 255.0
img = np.array([img])

prediction = model.predict(img)


for i in range(10):
    print(class_names[i], prediction[0][i])

print("Prediction:", class_names[np.argmax(prediction)])