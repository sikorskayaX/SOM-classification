# Import necessary libraries

from minisom import MiniSom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set the number of rows and columns for generating the SOM
num_rows = 210  
num_features = 7  

# Generate random numbers for features
np.random.seed(10)  
features = np.random.normal(loc=0.0, scale=1.0, size=(num_rows, num_features))

# Generate a target variable
targets = np.random.choice([1, 2, 3], size=num_rows)

# Combine features and target variable into one array
data = np.hstack((features, targets.reshape(num_rows, 1)))

# Create a DataFrame
df = pd.DataFrame(data, columns=['area', 'perimeter', 'compactness', 'length_kernel', 'width_kernel', 'asymmetry_coefficient', 'length_kernel_groove', 'target'])

# Save the DataFrame to a text file with tab separators
df.to_csv('seeds_dataset.txt', sep='\t', index=False, header=False)

# Load data from the file
data = pd.read_csv('seeds_dataset.txt', names=['area', 'perimeter', 'compactness', 'length_kernel', 'width_kernel', 'asymmetry_coefficient', 'length_kernel_groove', 'target'], usecols=[0,2], sep='\t+', engine='python')
data = data.values

# Normalize the data
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Generate 5 random test samples
np.random.seed(10)  # for reproducibility
test_samples = np.random.rand(5, 2) * 2 - 1  # 5 points in the range [-1, 1]


# Visualize the training and test samples
plt.figure(figsize=(8, 8))
plt.scatter(data[:, 0], data[:, 1], label='Training Samples')
plt.scatter(test_samples[:, 0], test_samples[:, 1], label='Test Samples', c='red')
plt.title('Training and Test Samples')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Classification for different number of classes
# For 6 classes
som_shape_6 = (1, 6)
som_6 = MiniSom(som_shape_6[0], som_shape_6[1], 2, sigma=.5, learning_rate=.5, neighborhood_function='gaussian', random_seed=10)
som_6.train_batch(data, 10000, verbose=True)

# For 7 classes
som_shape_7 = (1, 7)
som_7 = MiniSom(som_shape_7[0], som_shape_7[1], 2, sigma=.5, learning_rate=.5, neighborhood_function='gaussian', random_seed=10)
som_7.train_batch(data, 10000, verbose=True)


# Visualize the 6-class clusters
winner_coordinates = np.array([som_6.winner(x) for x in data]).T
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape_6)

plt.figure(figsize=(8, 8))
for c in np.unique(cluster_index):
    plt.scatter(data[cluster_index == c, 0], data[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)
for centroid in som_6.get_weights():
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', s=80, linewidths=2, color='k', label='centroid')

plt.title('SOM 6 Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Visualize the 7-class clusters
winner_coordinates = np.array([som_7.winner(x) for x in data]).T
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape_7)

plt.figure(figsize=(8, 8))
for c in np.unique(cluster_index):
    plt.scatter(data[cluster_index == c, 0], data[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)
for centroid in som_7.get_weights():
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', s=80, linewidths=2, color='k', label='centroid')

plt.title('SOM 7 Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
