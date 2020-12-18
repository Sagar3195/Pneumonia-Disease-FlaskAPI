# -*- coding: utf-8 -*-
"""Pneumonia Prediction based on CXR image using TransferLerning(VGG19).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JZgp9JjfQ0Xvvshu-W9INGDoa8mOkTOs
"""

from tensorflow.keras.layers import Input, Lambda, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import cv2
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

### Re-sizing the image
IMAGE_SIZE = [224, 224]
train_path = "/content/drive/My Drive/Lung_Disease/lung_disease/train"
test_path = "/content/drive/My Drive/Lung_Disease/lung_disease/test"



vgg = VGG19(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)

### Don't train the existing the weights
for layer in vgg.layers:
  layer.trainable = False

### Here getting number of output classes
folders = glob("/content/drive/My Drive/Lung_Disease/lung_disease/train/*")
folders

x = Flatten()(vgg.output)

prediction = Dense(len(folders), activation = 'softmax')(x)

### create the model object

model = Model(inputs = vgg.input, outputs = prediction)

### Now we view the structure of the model

model.summary()

#Now we compile the model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./ 255)

## here we provide same target size as initialised for images

train_set = train_datagen.flow_from_directory(train_path, target_size = (224, 224), batch_size = 32, class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(test_path, target_size = (224, 224), batch_size = 32, class_mode = 'categorical')

#now we train the model
result = model.fit_generator(train_set, validation_data = test_set, epochs = 10, steps_per_epoch = len(train_set), validation_steps = len(test_set))

plt.plot(result.history['loss'], label = 'Train Loss')
plt.plot(result.history['val_loss'], label = 'Val Loss')
plt.legend()
plt.show()

plt.plot(result.history['accuracy'], label = 'Train Acc')
plt.plot(result.history['val_accuracy'], label = 'Val Acc')
plt.legend()
plt.show()



from tensorflow.keras.models import load_model
model.save("pneumonia_model_vgg19.h5")

## Now we load model for prediction of dataset
 
model_img = load_model("pneumonia_model_vgg19.h5")

img = image.load_img("/content/drive/My Drive/Lung_Disease/lung_disease/test/PNEUMONIA/person100_bacteria_478.jpeg", target_size = (224, 224))

img

x = image.img_to_array(img)

x

x = np.expand_dims(x, axis = 0)

x

img_data = preprocess_input(x)

img_data

classes = model.predict(img_data)

#In folder, first folder is Normal ie 0 and second folder is pneumonia ie 1 , this is how specify image.
classes

a = np.expand_dims(classes, axis = 0)
a.ndim

a = np.argmax(classes, axis = 1)

if a == 0:
  print("Great!!!, You don't have pneumonia disease.")
else:
  print("Sorry, You have pneumonia disease,kindly contact your doctor.")








