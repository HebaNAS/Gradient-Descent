
# coding: utf-8

# # Gradient Descent
# 
# **F21BC Coursework 1**
# 
# <sub>Name: **Heba El-Shimy**</sub>
# <br>
# <sub>Based on code obtained from coursework specfication report, written by **Dr. Marta Vallejo**</sub>

# In[1]:


# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from math import sqrt


# In[2]:


np.set_printoptions(precision=2)


# In[3]:


# Loading the dataset
os.getcwd()
train_dataset = h5py.File('trainCats.h5', "r")
trainSetX = np.array(train_dataset["train_set_x"][:]) # your train set features
trainSetY = np.array(train_dataset["train_set_y"][:]) # your train set labels
trainSetY = trainSetY.reshape((1, trainSetY.shape[0]))

test_dataset = h5py.File('testCats.h5', "r")
testSetX = np.array(test_dataset["test_set_x"][:]) # your test set features
testSetY = np.array(test_dataset["test_set_y"][:]) # your test set labels
testSetY = testSetY.reshape((1, testSetY.shape[0]))

classes = np.array(test_dataset["list_classes"][:]) # the list of classes


# In[4]:


# Example of a picture
index = 20
plt.imshow(trainSetX[index])
plt.show()
print ("y = " + str(trainSetY[:, index]) + ", it's a '" + classes[np.squeeze(trainSetY[:, index])].decode("utf-8") +  "' picture.")


# In[5]:


# Images dimensions
print(trainSetX.shape)
print(testSetX.shape)

print('Image dimensions: {}px x {}px '.format(trainSetX.shape[1], trainSetX.shape[2]))
print('Image channels: {}'.format(trainSetX.shape[-1]))
print('Number of training examples: {} images'.format(trainSetX.shape[0]))
print('Number of test examples: {} images'.format(testSetX.shape[0]))


# In[6]:


# Flatten the pictures
# Applying (num_pixel x num_pixel x num_channels)
trainSetXF= trainSetX.reshape(trainSetX.shape[0], -1).T
testSetXF = testSetX.reshape(testSetX.shape[0], -1).T

print('Shape of training data after flattening: {}'.format(trainSetXF.shape))
print('Shape of test data after flattening: {}'.format(testSetXF.shape))


# In[7]:


# Normalize images
# Applying (pixel_value/255)
trainSetXFN = trainSetXF / 255
testSetXFN = testSetXF / 255

print('Shape of training data after normalizing: {}'.format(trainSetXFN.shape))
print('Shape of test data after normalizing: {}'.format(testSetXFN.shape))

print('First row of training data before normalizing: \n{}\n'.format(trainSetXF[0]))
print('First row of training data after normalizing: \n{}\n'.format(trainSetXFN[0]))

print('First row of test data before normalizing: \n{}\n'.format(testSetXF[0]))
print('First row of test data after normalizing: \n{}\n'.format(testSetXFN[0]))


# In[8]:


# Network Topology
print('Number of input units: {}'.format(trainSetXFN.shape[0]))
print('Number of outputs: {}'.format(classes.shape[0]))


# In[336]:


# Initialize weights
W = np.random.uniform(low=-0.5, high=0.5, size=(trainSetXFN.shape[0], 1)) / sqrt(trainSetXFN.shape[1])

print('Shape of weights matrix: {}'.format(W.shape))
print('Range of values in weights matrix = [{} - {}]'.format(W.min(), W.max()))
print('First (only) column in weights matrix: \n{}'.format(W))


# In[26]:


# Initialize biases
b = np.zeros([1, ])

#print('Shape of bias vector: {}'.format(b.shape))
print('First value in bias vector: \n{}'.format(b))


# In[21]:


# Activation function
# Sigmoid

def sigmoid(z):
    """
    Compute sigmoid function
    @param z: value to compute sigmoid for (WX + b)
    """
    
    a = np.zeros([1, 1])
    a = 1 / (1 + np.exp(-z))

    return a


# In[22]:


# Cost calculation
# Cross-Entropy as the loss function

def cost(a, y):
    """
    Compute loss function
    @param a: predicted label
    @param y: actual label
    """
    
    L = np.sum((y * np.log(a)) + ((1 - y) * np.log(1 - a)))
    
    J = (-1 / y.shape[1]) * L
    
    return J


# In[338]:


# Training the neuron

W_mod = np.copy(W) # modified weights matrix
b_mod = np.copy(b) # modidied bias vector
lr = 0.1 # learning rate
epochs = 1000 # number of iterations
costs = [] # store all calculated costs

# Training iterations
for i in range(epochs):  
    J = 0
    dW = np.zeros(W_mod.shape)   
    db = b_mod
    
    z = np.dot(W_mod.T, trainSetXFN) + b_mod
    predicted_labels = sigmoid(z)
    
    J = cost(predicted_labels, trainSetY)
    dW = (1 / trainSetY.shape[1]) * np.dot(trainSetXFN, (predicted_labels - trainSetY).T)
    db = (1 / trainSetY.shape[1]) * np.sum((predicted_labels - trainSetY), axis=1)
    
    costs.append(J)
    # learning rate decay
    if i > 0:
        if i % 100 == 0: #and abs(costs[i-50] - J) <= 0.5:
            lr /= 2
    
    W_mod = W_mod - (lr * dW)
    b_mod = b_mod - (lr * db)
    print('Learning rate: {}'.format(lr))
    print('Iteration {}\t =======> \t Cost: {:.2f}\n'.format(i, J))


# In[339]:


# Test the model on the test set
test_pred = sigmoid(np.dot(W_mod.T, testSetXFN) + b_mod)

# Measure model accuracy
count = 0
for pred, label in zip(test_pred.ravel(), testSetY.ravel()):
    if round(pred) == label:
        count += 1

accuracy = float(count) / testSetY.shape[1] * 100
print('Model\'s accuracy = {:.2f}%'.format(accuracy))


# In[343]:


# Example of a picture
index = int(np.random.randint(low=0, high=49))
plt.imshow(testSetX[index])
plt.show()

class_name = [0, 0]

if round(test_pred.ravel()[index]) == 0:
    class_name[0] = 'non-cat'
elif round(test_pred.ravel()[index]) == 1:
    class_name[1] = 'cat'

print ("y = " + str(testSetY[:, index]) + 
       ", predicted class= " + str(round(test_pred.ravel()[index])) +
       ", it's a '" + class_name[int(round(test_pred.ravel()[index]))] +  "' picture.")


# In[405]:


import glob
import cv2
import os
import matplotlib.image as mpimg

# Experimenting on collected images

img_dir = "./images" 
img_files = glob.glob(os.path.join(img_dir,'*.jpg'))
imgs = []

for img in img_files:
    imgs.append(img)


# In[406]:


image_1 = plt.imread(imgs[0])
plt.imshow(image_1)

# Flatten and normalize image
image_1FN = image_1.reshape(-1).T / 255
print(image_1FN.shape)

# Use optimized weights for predicting the ouput
test_1 = sigmoid(np.dot(W_mod.T, image_1FN) + b_mod)

class_test_1 = [0, 0]

if round(test_1[0]) == 0:
    class_test_1[0] = 'non-cat'
elif round(test_1[0]) == 1:
    class_test_1[1] = 'cat'

print ("y = [0]"  + 
       ", predicted class= " + str(round(test_1[0])) +
       ", it's a '" + class_test_1[int(round(test_1[0]))] +  "' picture.")


# In[407]:


image_2 = plt.imread(imgs[1])
plt.imshow(image_2)

# Flatten and normalize image
image_2FN = image_2.reshape(-1).T / 255
print(image_2FN.shape)

# Use optimized weights for predicting the ouput
test_2 = sigmoid(np.dot(W_mod.T, image_2FN) + b_mod)

class_test_2 = [0, 0]

if round(test_2[0]) == 0:
    class_test_2[0] = 'non-cat'
elif round(test_2[0]) == 1:
    class_test_2[1] = 'cat'

print ("y = [0]"  + 
       ", predicted class= " + str(round(test_2[0])) +
       ", it's a '" + class_test_2[int(round(test_2[0]))] +  "' picture.")


# In[408]:


image_3 = plt.imread(imgs[2])
plt.imshow(image_3)

# Flatten and normalize image
image_3FN = image_3.reshape(-1).T / 255
print(image_3FN.shape)

# Use optimized weights for predicting the ouput
test_3 = sigmoid(np.dot(W_mod.T, image_3FN) + b_mod)

class_test_3 = [0, 0]

if round(test_3[0]) == 0:
    class_test_3[0] = 'non-cat'
elif round(test_3[0]) == 1:
    class_test_3[1] = 'cat'

print ("y = [0]"  + 
       ", predicted class= " + str(round(test_3[0])) +
       ", it's a '" + class_test_3[int(round(test_3[0]))] +  "' picture.")

