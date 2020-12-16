#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Network

# ### Importing the libraries

# In[3]:




import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[4]:


tf.__version__


# ## Part 1 - Data Preprocessing

# ### Preprocessing the Training set

# In[ ]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
        'training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


# ### Preprocessing the Test set

# In[ ]:


test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'test_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')


# ## Part 2 - Building the CNN

# ### Initialising the CNN

# In[ ]:


cnn=tf.keras.models.Sequential()


# ### Step 1 - Convolution

# In[ ]:


cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))


# ### Step 2 - Pooling

# In[ ]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))


# ### Adding a second convolutional layer

# In[ ]:


cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))


# ### Step 3 - Flattening

# In[ ]:


cnn.add(tf.keras.layers.Flatten())


# ### Step 4 - Full Connection

# In[ ]:


cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))


# ### Step 5 - Output Layer

# In[ ]:


cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))


# ## Part 3 - Training the CNN

# ### Compiling the CNN

# In[ ]:


cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# ### Training the CNN on the Training set and evaluating it on the Test set

# In[ ]:


import sys
from PIL import Image
sys.modules['Image'] = Image 


# In[ ]:


from PIL import Image
print(Image.__file__)


# In[ ]:


import Image
print(Image.__file__)


# In[18]:



conda install Pillow


# In[19]:



cnn.fit(x=training_set,validation_data=test_set,epochs=25)


# ## Part 4 - Making a single prediction

# In[39]:


import numpy as np
from keras.preprocessing import image
test_image=image.load_img(r'C:\Users\DPM\Downloads\Section+40+-+Convolutional+Neural+Networks+(CNN)\Section 40 - Convolutional Neural Networks (CNN)\dataset\single_prediction\cat_or_dog_3.jpg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction='dog'
else:
  prediction='cat'  


# In[40]:


print(prediction)


# In[ ]:





# In[ ]:





# In[ ]:




