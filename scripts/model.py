import os
import numpy as np
from keras.preprocessing import image

def load_images_from_path(path, label):
    images = []
    labels = []

    for file in os.listdir(path):
        images.append(image.img_to_array(image.load_img(os.path.join(path, file), target_size=(224, 224, 3))))
        labels.append((label))
        
    return images, labels

x = []
y = []

images, labels = load_images_from_path('../spectrograms/Bach', 0)
    
x += images
y += labels

images, labels = load_images_from_path('../spectrograms/Chopin', 1)

x += images
y += labels

images, labels = load_images_from_path('../spectrograms/Liszt', 2)

x += images
y += labels

images, labels = load_images_from_path('../spectrograms/Scarlatti', 3)

x += images
y += labels

images, labels = load_images_from_path('../spectrograms/Schubert', 4)

x += images
y += labels

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=0)

x_train_norm = np.array(x_train) / 255
x_test_norm = np.array(x_test) / 255

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet import preprocess_input
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x_train_norm = preprocess_input(np.array(x_train))
x_test_norm = preprocess_input(np.array(x_test))

train_features = base_model.predict(x_train_norm)
test_features = base_model.predict(x_test_norm)

model = Sequential()
model.add(Input(shape=train_features.shape[1:]))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(train_features, y_train_encoded, validation_data=(test_features, y_test_encoded), batch_size=8, epochs=7)

model.save('../ClassicalNN.keras')

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

y_predicted = model.predict(test_features)
mat = confusion_matrix(y_test_encoded.argmax(axis=1), y_predicted.argmax(axis=1))
class_labels = ['Bach', 'Chopin', 'Liszt', 'Scarlatti', 'Schubert']

sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)

plt.xlabel('Predicted label')
plt.ylabel('Actual label')

plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
