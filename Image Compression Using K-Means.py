
# coding: utf-8

# In[233]:


# image compression using k-means clustering
# inspired by Andrew Ng's Machine Learning Course assignment on k-means clustering


# In[234]:


# import necessary libraries
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from joblib import dump


# In[235]:


# import image
im = Image.open("bridge.jpeg")
# display image
im


# In[237]:


# get pixels of the image
pixel_np = np.asarray(im)
# reshape array (remove rows and columns)
image_height = im.height
image_width = im.width
pixel_np = np.reshape(pixel_np, (height * width, 3))
# display as df
pd.DataFrame(pixel_np, columns=["r", "g", "b"]).head()


# In[238]:


# run k-means clustering on the pixel data
num_of_centroids = 16 # a 4-bit image is represented by 2^4 colours
num_of_runs = 10 # number of times to run the k-means algorithm before determining the best centroids
max_iterations = 300 # number of iterations before k-means comes to an end for a single run
verbosity = 0 # show what's going on when the algorithm is running

# initiate a kmeans object
compressor = KMeans(n_clusters=num_of_centroids, n_init=num_of_runs, max_iter=max_iterations, verbose=verbosity)
# run k-means clustering
compressor.fit(pixel_np)


# In[239]:


# save the fitted model
dump(compressor, "compressor.joblib")


# In[240]:


# create an array replacing each pixel label with its corresponding cluster centroid
pixel_centroid = np.array([list(compressor.cluster_centers_[label]) for label in compressor.labels_])


# In[245]:


# convert the array to an unsigned integer type
pixel_centroid = pixel_centroid.astype("uint8")
# reshape this array according to the height and width of our image
pixel_centroids_reshaped = np.reshape(pixel_centroid, (height, width, 3), "C")


# In[246]:


# create the compressed image
compressed_im = Image.fromarray(pixel_centroids_reshaped)
# display compressed image
compressed_im

