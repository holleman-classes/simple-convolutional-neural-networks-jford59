import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import keras

from tensorflow.keras import Input, layers, Model

def build_model1():
  model1 = tf.keras.Sequential([
    Input(shape=(32, 32, 3)),
    layers.Conv2D(32, kernel_size=(3,3), activation="relu", padding='same', strides=(2,2)),    
    layers.BatchNormalization(), # optional, but may help convergence 
    layers.Conv2D(64, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
    layers.BatchNormalization(), # optional, but may help convergence 
    layers.Conv2D(128, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
    layers.BatchNormalization(), # optional, but may help convergence 
    layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), # optional, but may help convergence 
    layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), # optional, but may help convergence 
    layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), # optional, but may help convergence 
    layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), # optional, but may help convergence 
    layers.MaxPooling2D(pool_size=(4, 4), strides=(4,4)), 
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(), # optional, but may help convergence 
    layers.Dense(10, activation='relu'),
  ])
  return model1

def build_model2():
  model2 = tf.keras.Sequential([
    Input(shape=(32, 32, 3)),
    layers.Conv2D(32, kernel_size=(3,3), activation="relu", padding='same', strides=(2,2)),    
    layers.BatchNormalization(), # optional, but may help convergence 
    layers.SeparableConv2D(64, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
    layers.BatchNormalization(), # optional, but may help convergence 
    layers.SeparableConv2D(128, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
    layers.BatchNormalization(), # optional, but may help convergence 
    layers.SeparableConv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), # optional, but may help convergence 
    layers.SeparableConv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), # optional, but may help convergence 
    layers.SeparableConv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), # optional, but may help convergence 
    layers.SeparableConv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
    layers.BatchNormalization(), # optional, but may help convergence 
    layers.MaxPooling2D(pool_size=(4, 4), strides=(4,4)), 
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(), # optional, but may help convergence 
    layers.Dense(10, activation='relu'),
  ])
  return model2

def build_model3():
    input_layer = Input(shape=(32, 32, 3))

    conv1 = layers.Conv2D(32, kernel_size=(3,3), activation="relu", padding='same', strides=(2,2))(input_layer)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Dropout(0.25)(conv1)
    conv2 = layers.Conv2D(64, kernel_size=(3,3), activation="relu", padding='same', strides=(2,2))(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Dropout(0.25)(conv2)
    conv3 = layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same', strides=(2,2))(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    skip1 = layers.Conv2D(128, kernel_size=(1,1), strides=(4,4), activation='relu', padding='same')(conv1)
    skip1 = layers.add([conv3, skip1])
    skip1 = layers.Dropout(0.25)(skip1)
    conv4 = layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same', strides=(2,2))(skip1)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Dropout(0.25)(conv4)
    conv5 = layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same', strides=(2,2))(conv4)
    conv5 = layers.BatchNormalization()(conv5)
    skip2 = layers.add([conv5, skip1])
    skip2 = layers.Dropout(0.25)(skip2)
    conv6 = layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same', strides=(2,2))(skip2)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Dropout(0.25)(conv6)
    conv7 = layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same', strides=(2,2))(conv6)
    conv7 = layers.BatchNormalization()(conv7)
    skip3 = layers.add([conv7, skip2])
    skip3 = layers.Dropout(0.25)(skip3)
    pool = layers.MaxPooling2D(pool_size=(4, 4), strides=(4,4))(skip3)
    flat = layers.Flatten()(pool)
    dense = layers.Dense(128, activation='relu')(flat)
    dense = layers.BatchNormalization()(dense) # optional, but may help convergence 
    dense = layers.Dense(10, activation='relu')(dense)
    model3 = keras.Model(inputs = input_layer, outputs = dense)

    return model3

model3 = build_model3()
model3.summary()

def build_model50k():
  model4 = tf.keras.Sequential([
  Input(shape=(32, 32, 3)),
  layers.Conv2D(16, (3, 3), activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.Conv2D(32, (3, 3), activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.Flatten(),
  layers.Dense(32, activation='relu'),
  layers.Dense(10, activation='softmax'),
  ])
  return model4

# no training or dataset construction should happen above this line
if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
  class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    # Now separate out a validation set.
# choose num_val_samples indices up to the size of train_images, !replace => no repeats

  ########################################
  ## Build and train model 1
  train_labels = train_labels.squeeze()
  test_labels = test_labels.squeeze()

  input_shape  = train_images.shape[1:]
  train_images = train_images / 255.0
  test_images  = test_images  / 255.0

  model1 = build_model1()
  model1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])  

  model1.summary()
  #model1.fit(train_images, train_labels, epochs=50)

  image_path = "test_image_horse.jpg"
  image = Image.open(image_path)
  image = np.array(image) / 255.0
  predictions = model1.predict(np.expand_dims(image, axis=0))
  print(predictions)

  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()
  model2.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])  
  
  model2.summary()
  #model2.fit(train_images, train_labels, epochs=50)
  
  ### Repeat for model 3 and your best sub-50k params model
  model3 = build_model3()
  model3.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])  
  
  model3.summary()
  #model3.fit(train_images, train_labels, epochs=50)

  model4 = build_model50k()
  model4.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])  
  
  model4.summary()
  model4.fit(train_images, train_labels, epochs=50)
  model4.save("best_model.h5")
