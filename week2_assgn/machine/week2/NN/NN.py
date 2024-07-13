#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os


# In[67]:


dataset = tf.keras.preprocessing.image_dataset_from_directory(r"C:\Users\HP\Downloads\homer_bart",image_size=(64, 64),label_mode = "binary")
train_data = dataset.take(8)
test_data = dataset.skip(8)
train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_data = test_data.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[68]:


preprocess = tf.keras.Sequential(
    [tf.keras.layers.Rescaling(1.0/255)]
)
NN = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(64, 64, 3)),
    preprocess,
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])


# In[69]:


NN.compile(optimizer='adam',
           loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
           metrics=['accuracy'])


# In[70]:


history = NN.fit(train_data,
                 epochs = 50,
                 batch_size = 32,
                 verbose=1,
                 validation_data=test_data)


# In[71]:


loss, accuracy = NN.evaluate(test_data)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# In[ ]:





# In[ ]:




