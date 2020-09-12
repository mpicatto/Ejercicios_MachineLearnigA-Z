#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 19:25:17 2020

@author: root
"""

#Convolutional Neural Networks

#------<Parte 1:Construir Red Neurnonal CNN>----------
#importar las librerias y paquetes

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Iniciar la CNN
classifier=Sequential()

#Crear Capa de Convolucion
classifier.add(Conv2D(filters=32, kernel_size=(3,3),
                      input_shape=(64,64,3), activation="relu"))

#Crear Capa de Max Pulling
classifier.add(MaxPooling2D(pool_size=(2,2),))

#Data Flattening
classifier.add(Flatten())

#Agregar Full Connections
classifier.add(Dense(units=128, activation = "relu"))
classifier.add(Dense(units=1, activation = "sigmoid"))  

#Compilar la CNN
classifier.compile(optimizer="adam", loss = "binary_crossentropy", metrics = ["accuracy"])


#------<Parte 2:Ajustar Red Neurnonal CNN a los datos>----------

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_dataset = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

testing_dataset = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

classifier.fit_generator(training_dataset,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=testing_dataset,
                        validation_steps=2000)






