# Import necessary libraries

from minisom import MiniSom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data from file
data = pd.read_csv('seeds_dataset.txt', names=['area', 'perimeter', 'compactness', 'length_kernel', 'width_kernel', 'asymmetry_coefficient', 'length_kernel_groove', 'target'], usecols=[0,2], sep='\t+', engine='python')
data = data.values

# Normalize data
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# 3 classes
som_shape = (1,3)

# Initialize SOM (self-organizing map)
som = MiniSom(som_shape[0], som_shape[1], data.shape[1], sigma=.5, learning_rate=.5,neighborhood_function='gaussian', random_seed=10)

# Train SOM
som.train_batch(data, 10000, verbose=False)

# Get winner coordinates for each data point
winner_coordinates = np.array([som.winner(x) for x in data]).T

# Get cluster index for each data point
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

# Plot clusters
for c in np.unique(cluster_index):
    plt.scatter(data[cluster_index == c, 0],data[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)
for centroid in som.get_weights():
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x',s=3, linewidths=25, color='k', label='centroid')
plt.legend()
plt.show()
