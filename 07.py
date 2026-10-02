import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from tensorflow.keras import datasets,layers,models
import pickle
import os

data_dir = "/Users/Zhuanz/Downloads/cifar-10-batches-py"

training_images = []
training_labels = []

for i in range(1, 6):
    with open(os.path.join(data_dir, f"data_batch_{i}"), "rb") as f:
        batch = pickle.load(f, encoding="bytes")
        training_images.append(batch[b"data"])
        training_labels.extend(batch[b"labels"])

training_images = np.concatenate(training_images)
training_labels = np.array(training_labels)

with open(os.path.join(data_dir, "test_batch"), "rb") as f:
    batch = pickle.load(f, encoding="bytes")
    testing_images = batch[b"data"]
    testing_labels = np.array(batch[b"labels"])

training_images = training_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
testing_images = testing_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

training_images = training_images / 255.0
testing_images = testing_images / 255.0

class_names = [
    "Plane", "Car", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i])
    plt.xlabel(class_names[training_labels[i]])

plt.show()

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images,testing_labels))

loss,accuracy = model.evaluate(testing_images, testing_labels)
print(f'loss={loss}')
print(f'accuracy={accuracy}')

model.save('image_classifier.keras')


model = models.load_model('image_classifier.keras')

img = cv.imread('deer.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img,(32,32))

plt.imshow(img,cmap=plt.cm.binary)

prediction = model.predict(np.array([img])/255)
index = np.argmax(prediction)
print(f'Predicted is {class_names[index]}')




