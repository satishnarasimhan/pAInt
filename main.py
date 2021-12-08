# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 15:49:49 2021

@author: Satish Narasimhan
"""

import matplotlib.pyplot as plt
import cv2
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import file_path as f

# Load the model from tensorflow hub
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
#model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-inceptionv3')

# Function to load images
def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels = 3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

content_image = load_image(f.content_path)
style_image = load_image(f.style_path)

# Style your image based on a style image.
# Tensor flow image stylization is used ie. based on the model loaded
stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

# Plot the image
plt.imshow(np.squeeze(stylized_image))
plt.show()

#Save the image
cv2.imwrite('image_generated.jpg', cv2.cvtColor(np.squeeze(stylized_image)*255, cv2.COLOR_BGR2GRAY))