#!/usr/bin/env python
# coding: utf-8

# In[5]:


pip install --user protobuf==4.25.7


# In[6]:


#Build a Multiclass classifier using the CNN model. Use MNIST or any other suitable dataset. 
#a. Perform Data Pre-processing 
#b. Define Model and perform training 
#c. Evaluate Results using confusion matrix. 
 

get_ipython().system('pip install tensorflow scikit-learn matplotlib')
 

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
 

# a. Data Preprocessing
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
 
# Normalize images to the range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
 
# Reshape to include channel dimension (28x28x1)
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]
 

 

# b. Define Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes for digits 0–9
])
 

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
 
# Train model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
 

# c. Evaluate using confusion matrix
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
 

cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - MNIST CNN")
plt.show()


# In[ ]:




