# Unsupervised Learning

- What is unsupervised learning?
- Clustering data with K-Means algorithm
- Estimating the number of clusters with Mean Shift algorithm
- Estimating the quality of clustering with silhouette scores
- What are Gaussian Mixture Models?
- Building a classifier based on Gaussian Mixture Models
- Finding subgroups in stock market using Affinity Propagation model
- Segmenting the market based on shopping patterns

## Table of contents

- [Unsupervised Learning](#unsupervised-learning)
  - [Table of contents](#table-of-contents)
  - [What is unsupervised learning?](#what-is-unsupervised-learning)
  - [Clustering data with K-Means algorithm](#clustering-data-with-k-means-algorithm)
    - [K-Means Algorithm Overview:](#k-means-algorithm-overview)
    - [Complex Real-World Example: Customer Segmentation](#complex-real-world-example-customer-segmentation)
  - [K-Means in real world datasets](#k-means-in-real-world-datasets)
    - [Example: K-Means Clustering on the Iris Dataset](#example-k-means-clustering-on-the-iris-dataset)
  - [Euclidean distance and ...](#euclidean-distance-and-)
  - [Scikit-learn measurement methods](#scikit-learn-measurement-methods)
    - [1. Manhattan Distance (L1 Norm):](#1-manhattan-distance-l1-norm)
    - [2. Chebyshev Distance (L∞ Norm):](#2-chebyshev-distance-l-norm)
    - [3. Cosine Similarity:](#3-cosine-similarity)
    - [4. Hamming Distance:](#4-hamming-distance)
    - [5. Jaccard Distance:](#5-jaccard-distance)
    - [6. Minkowski Distance:](#6-minkowski-distance)
    - [7. Correlation Distance:](#7-correlation-distance)
    - [8. Mahalanobis Distance:](#8-mahalanobis-distance)
    - [9. Custom Distance Metrics:](#9-custom-distance-metrics)
  - [Mean Shift algorithm](#mean-shift-algorithm)
    - [Mean Shift Algorithm Overview:](#mean-shift-algorithm-overview)
    - [Complex Dataset Example: Image Segmentation](#complex-dataset-example-image-segmentation)
  - [Estimating the number of clusters with Mean Shift algorithm](#estimating-the-number-of-clusters-with-mean-shift-algorithm)
    - [Let's use a financial price dataset to estimate the number of clusters with the Mean Shift algorithm.](#lets-use-a-financial-price-dataset-to-estimate-the-number-of-clusters-with-the-mean-shift-algorithm)
  - [Silhouette scores](#silhouette-scores)
  - [Silhouette Score example](#silhouette-score-example)
  - [Estimating the quality of clustering with silhouette scores](#estimating-the-quality-of-clustering-with-silhouette-scores)
  - [Gaussian Mixture Models](#gaussian-mixture-models)
  - [Building a classifier based on Gaussian Mixture Models](#building-a-classifier-based-on-gaussian-mixture-models)
  - [Affinity Propagation model](#affinity-propagation-model)
  - [Finding subgroups in stock market using Affinity Propagation model](#finding-subgroups-in-stock-market-using-affinity-propagation-model)
  - [Segmenting the market based on shopping patterns](#segmenting-the-market-based-on-shopping-patterns)

## What is unsupervised learning?

- A category of machine learning where the algorithm learns patterns, structures, or relationships within a dataset without explicit supervision or labeled target outputs. 
- In unsupervised learning, the algorithm tries to find hidden structures or patterns within the data on its own.

There are two primary types of unsupervised learning techniques:

1. **Clustering:** Clustering algorithms group similar data points together into clusters or groups based on certain similarities or patterns. Common clustering algorithms include K-Means clustering, Hierarchical clustering, and DBSCAN. Clustering can be used for tasks such as customer segmentation, anomaly detection, and image segmentation.

2. **Dimensionality Reduction:** Dimensionality reduction techniques aim to reduce the number of features or variables in a dataset while preserving as much relevant information as possible. Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) are popular dimensionality reduction methods. Dimensionality reduction is useful for data visualization, feature selection, and simplifying complex datasets.

- Unsupervised learning is particularly valuable when dealing with large and complex datasets, as it can help discover underlying patterns or structures that may not be apparent through manual inspection. It is also used in various fields, including:

- **Anomaly Detection:** Identifying unusual or unexpected patterns in data, such as detecting fraudulent transactions or network intrusions.

- **Recommendation Systems:** Recommending products, services, or content to users based on their past behavior and preferences.

- **Natural Language Processing (NLP):** Grouping similar documents, topics, or sentiments in text data.

- **Image and Speech Processing:** Segmenting images, extracting features, or reducing noise in audio data.

- **Biology and Genomics:** Clustering genes based on expression patterns or finding subpopulations of cells.

Unsupervised learning is a valuable tool for exploratory data analysis and can serve as a precursor to more targeted supervised learning tasks. It allows data scientists and analysts to gain insights into the underlying structure of data and make data-driven decisions.

## Clustering data with K-Means algorithm

- A type of unsupervised learning where the goal is to group similar data points together into clusters or groups. 
- One of the most widely used clustering algorithms is the K-Means algorithm. 
- K-Means is an iterative algorithm that partitions a dataset into K clusters, with each cluster represented by its centroid (the mean of the data points in the cluster).

### K-Means Algorithm Overview:

1. **Initialization:** Choose K initial centroids randomly from the data points. These initial centroids represent the initial cluster centers.

2. **Assignment:** For each data point, calculate its distance to each centroid and assign it to the cluster whose centroid is the closest.

3. **Update:** Recalculate the centroids of each cluster by taking the mean of all data points assigned to that cluster.

4. **Repeat:** Repeat the assignment and update steps until convergence. Convergence typically occurs when the centroids no longer change significantly or when a maximum number of iterations is reached.

### Complex Real-World Example: Customer Segmentation

Let's consider a real-world example of customer segmentation for a retail business. We have a dataset of customer purchase history, including variables like purchase frequency, total spending, and product category preferences. We want to group customers into distinct segments to tailor marketing strategies.

**Step 1: Data Preparation**

First, gather and preprocess the data. Standardize the features if necessary to ensure they have similar scales.

**Step 2: Choosing K**

Decide on the number of clusters, K, based on domain knowledge, business requirements, or using methods like the Elbow Method or Silhouette Score. For our example, let's say we choose K=3.

**Step 3: Initialize Centroids**

Randomly select K data points as initial centroids.

**Step 4: Iterative Assignment and Update**

Repeat the following steps until convergence:

- **Assignment:** For each customer, calculate the distance to each centroid and assign them to the nearest cluster.

- **Update:** Recalculate the centroids of each cluster by taking the mean of all customer data points in that cluster.

**Step 5: Convergence**

Stop the algorithm when the centroids no longer change significantly or after a predefined number of iterations.

**Step 6: Interpretation**

After convergence, you will have clustered customers into three segments. These segments may represent "Frequent Shoppers," "High Spenders," and "Occasional Shoppers," for example.

**Step 7: Evaluation and Action**

Evaluate the quality of the clusters using internal metrics (e.g., inertia) or external metrics (e.g., silhouette score). Then, take specific actions based on the customer segments, such as designing targeted marketing campaigns or adjusting product offerings.

**Code Example:**

```python
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load and preprocess the customer data
data = pd.read_csv('customer_data.csv')
# Standardize the data if needed

# Choose the number of clusters (K)
K = 3

# Initialize and fit the K-Means model
kmeans = KMeans(n_clusters=K, random_state=42)
kmeans.fit(data)

# Assign customers to clusters
data['cluster'] = kmeans.predict(data)

# Visualize the clusters
plt.scatter(data['purchase_frequency'], data['total_spending'], c=data['cluster'], cmap='viridis')
plt.xlabel('Purchase Frequency')
plt.ylabel('Total Spending')
plt.title('Customer Segmentation')
plt.show()
```

In this example, we use the K-Means algorithm to cluster customers based on their purchase frequency and total spending. The resulting clusters can guide marketing and business decisions tailored to the distinct customer segments.

Please note that K-Means clustering is just one of many clustering algorithms available, and its effectiveness depends on the nature of your data and the specific problem you are trying to solve.

## K-Means in [real world datasets](http://archive.ics.uci.edu/)

### [Example](https://www.kaggle.com/code/ryanholbrook/clustering-with-k-means): K-Means Clustering on the Iris [Dataset](https://data.world/datasets/clustering)

- The Iris dataset contains measurements of four features (sepal length, sepal width, petal length, and petal width) for three different species of iris flowers (setosa, versicolor, and virginica). 
- We'll use K-Means clustering to group the iris flowers based on their feature measurements.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Choose the number of clusters (K)
K = 3  # Since there are three species in the dataset

# Initialize and fit the K-Means model
kmeans = KMeans(n_clusters=K, random_state=42)
kmeans.fit(data)

# Add cluster labels to the dataset
data['cluster'] = kmeans.labels_

# Visualize the clusters (2D scatter plot)
plt.scatter(data['sepal length (cm)'], data['sepal width (cm)'], c=data['cluster'], cmap='viridis')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('K-Means Clustering of Iris Flowers (Sepal Features)')
plt.show()
```

In this example, we:

1. Load the Iris dataset, which is included in Scikit-Learn.

2. Choose the number of clusters (K) as 3, corresponding to the three species of iris flowers.

3. Apply the K-Means clustering algorithm to the four feature columns (sepal length, sepal width, petal length, and petal width).

4. Add cluster labels to the dataset.

5. Visualize the clusters by creating a 2D scatter plot of sepal length versus sepal width, with each point color-coded according to its cluster assignment.

## Euclidean distance and ...

- In K-Means clustering, the choice of distance metric (or dissimilarity metric) is crucial because it determines how the algorithm calculates the distance between data points and cluster centroids. 
- While the Euclidean distance is the most commonly used metric, there are other distance metrics that you can consider depending on the nature of your data and the problem you are trying to solve.

1. **Manhattan Distance (L1 Norm):** Also known as the taxicab distance or city block distance, it calculates the distance by summing the absolute differences between coordinates. It is suitable for cases where movement can only occur along gridlines.

2. **Chebyshev Distance (L∞ Norm):** This metric calculates the maximum absolute difference between coordinates. It is useful when you want to measure similarity based on the greatest difference in any dimension.

3. **Minkowski Distance:** A generalized distance metric that includes both Manhattan distance and Euclidean distance as special cases. It introduces a parameter (p) that controls the level of norm (Lp) used. When p=1, it's Manhattan distance, and when p=2, it's Euclidean distance.

4. **Cosine Similarity:** Instead of measuring distance, cosine similarity measures the cosine of the angle between two vectors. It is often used for text data or high-dimensional data, where the magnitude of the vectors is less important than their direction.

5. **Correlation Distance:** It measures the dissimilarity between data points based on their correlation coefficient. It is useful when you want to capture relationships between variables rather than their absolute values.

6. **Hamming Distance:** Primarily used for binary data, Hamming distance calculates the number of positions at which two binary strings of equal length differ. It is often applied in text analysis and genetics.

7. **Jaccard Distance:** Suitable for sets or binary data, Jaccard distance calculates dissimilarity as the size of the symmetric difference divided by the size of the union of two sets. It is commonly used in text mining and recommendation systems.

8. **Mahalanobis Distance:** It accounts for correlations between variables and scales them by the inverse of the covariance matrix. Mahalanobis distance is useful when dealing with high-dimensional data with varying scales.

9. **Custom Distance Metrics:** Depending on your specific problem domain, you may define custom distance metrics that capture domain-specific notions of similarity or dissimilarity.

- The choice of distance metric should align with the characteristics of your data and the goals of your clustering task. 
- Experimenting with different metrics and evaluating the clustering results using appropriate validation measures can help you determine the most suitable distance metric for your specific use case.

## Scikit-learn measurement methods

- Supporting many of the distance metrics mentioned above through the use of the `sklearn.metrics.pairwise` module.

### 1. Manhattan Distance (L1 Norm):

```python
from sklearn.metrics.pairwise import manhattan_distances

# Example data
data = [[1, 2], [4, 6], [7, 8]]

# Calculate Manhattan distances between data points
manhattan_dist = manhattan_distances(data)
print(manhattan_dist)
```

### 2. Chebyshev Distance (L∞ Norm):

```python
from sklearn.metrics.pairwise import chebyshev_distances

# Example data
data = [[1, 2], [4, 6], [7, 8]]

# Calculate Chebyshev distances between data points
chebyshev_dist = chebyshev_distances(data)
print(chebyshev_dist)
```

### 3. Cosine Similarity:

```python
from sklearn.metrics.pairwise import cosine_similarity

# Example data
data = [[1, 2], [4, 6], [7, 8]]

# Calculate cosine similarities between data points
cosine_sim = cosine_similarity(data)
print(cosine_sim)
```

### 4. Hamming Distance:

- Scikit-Learn doesn't have a built-in function for Hamming distance, but you can implement it yourself if you're working with binary data:

```python
def hamming_distance(x, y):
    return sum(x != y)

# Example binary data
data1 = [0, 1, 1, 0, 1]
data2 = [1, 1, 0, 0, 1]

hamming_dist = hamming_distance(data1, data2)
print(hamming_dist)
```

### 5. Jaccard Distance:

- Scikit-Learn provides a `pairwise_distances` function that allows you to specify custom distance metrics.

```python
from sklearn.metrics import pairwise_distances

# Example binary data
data = [[0, 1, 1, 0, 1], [1, 1, 0, 0, 1], [0, 1, 0, 1, 1]]

# Define a custom function for Jaccard distance
def jaccard_distance(u, v):
    intersection = len(set(u).intersection(v))
    union = len(set(u).union(v))
    return 1.0 - (intersection / union)

# Calculate Jaccard distances between data points
jaccard_dist = pairwise_distances(data, metric=jaccard_distance)
print(jaccard_dist)
```

### 6. Minkowski Distance:

- Scikit-Learn's `pairwise_distances` function allows you to specify the `p` parameter for Minkowski distance. 
- For example, for Euclidean distance (p=2) and Manhattan distance (p=1):

```python
from sklearn.metrics import pairwise_distances

# Example data
data = [[1, 2], [4, 6], [7, 8]]

# Calculate Euclidean distances (p=2) between data points
euclidean_dist = pairwise_distances(data, metric='minkowski', p=2)
print(euclidean_dist)

# Calculate Manhattan distances (p=1) between data points
manhattan_dist = pairwise_distances(data, metric='minkowski', p=1)
print(manhattan_dist)
```

### 7. Correlation Distance:

- Scikit-Learn doesn't provide a built-in function for correlation distance, but you can calculate it using NumPy as follows:

```python
import numpy as np

# Example data
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Calculate correlation distances between data points
correlation_dist = 1.0 - np.corrcoef(data)
print(correlation_dist)
```

### 8. Mahalanobis Distance:

- Scikit-Learn provides `mahalanobis_distances` in the `sklearn.metrics.pairwise` module, but it requires specifying a precision matrix.

```python
from sklearn.metrics.pairwise import mahalanobis_distances
from sklearn.covariance import EmpiricalCovariance

# Example data
data = [[1, 2], [4, 6], [7, 8]]

# Calculate Mahalanobis distances with a custom precision matrix
cov_matrix = EmpiricalCovariance().fit(data).covariance_
precision_matrix = np.linalg.inv(cov_matrix)
mahalanobis_dist = mahalanobis_distances(data, [np.mean(data, axis=0)], precision_matrix)
print(mahalanobis_dist)
```

These examples illustrate how to calculate the remaining distance metrics (Minkowski, Correlation, and Mahalanobis) using Scikit-Learn or NumPy as appropriate. You can adapt these examples to your specific use cases and data.

### 9. Custom Distance Metrics:

- Creating custom distance metrics in Scikit-Learn is possible by defining a custom Python function that computes the distance between data points. 
- This custom function can then be used with the `pairwise_distances` function. 
- Suppose you want to create a custom distance metric based on the Euclidean distance but with a specific weighting for each feature. 

```python
import numpy as np
from sklearn.metrics import pairwise_distances

# Define a custom distance metric function
def custom_weighted_euclidean(u, v, weights):
    # Ensure that u and v have the same length
    assert len(u) == len(v) == len(weights)
    
    # Calculate the weighted Euclidean distance
    squared_diff = [(x - y) ** 2 * w for x, y, w in zip(u, v, weights)]
    return np.sqrt(sum(squared_diff))

# Example data
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
weights = [0.5, 1.0, 2.0]  # Weights for each feature

# Calculate distances using the custom distance metric
custom_dist = pairwise_distances(data, metric=custom_weighted_euclidean, **{'weights': weights})
print(custom_dist)
```

In this example:

1. We define the `custom_weighted_euclidean` function, which takes three arguments: `u` and `v` (the data points to compare) and `weights` (a list of weights for each feature).

2. Inside the custom function, we calculate the weighted squared differences between the corresponding features of `u` and `v` and then take the square root of the sum of these weighted squared differences.

3. We apply the custom distance metric to the example data, providing the `weights` parameter as a dictionary argument.

You can create more complex custom distance metrics based on the specific requirements of your problem. Custom distance metrics are valuable when you need to incorporate domain-specific knowledge or tailor the similarity measure to your data characteristics.

## Mean Shift algorithm

- A clustering algorithm used for finding the modes or peaks in a dataset. 
- It is a non-parametric algorithm, meaning it doesn't assume any specific shape for the clusters. 
- Instead, it identifies clusters by shifting data points towards the mode of their local density distribution. 

### Mean Shift Algorithm Overview:

1. **Kernel Selection:** Choose a kernel function (e.g., Gaussian kernel) that defines the shape of the neighborhood around each data point. The kernel function determines how much nearby data points influence the position of a point.

2. **Bandwidth Selection:** Choose a bandwidth parameter that determines the size of the kernel's neighborhood. A larger bandwidth results in smoother clusters, while a smaller bandwidth results in more granular clusters.

3. **Initialization:** Place a kernel (e.g., a small circle) at each data point in the dataset.

4. **Mean Shift Iteration:** For each data point, calculate the mean shift vector, which points towards the weighted mean of the data points within the kernel's neighborhood.

5. **Update:** Update the position of each kernel by shifting it along the mean shift vector.

6. **Convergence:** Repeat the mean shift iteration until the kernels no longer move significantly or for a specified number of iterations.

7. **Clustering:** Assign each data point to the nearest kernel, resulting in clusters.

### Complex Dataset Example: Image Segmentation

- Let's use the Mean Shift algorithm to perform image segmentation on a complex dataset. 
- Image segmentation involves dividing an image into regions or segments with similar characteristics. 
- In this case, we'll use the algorithm to cluster similar pixel colors in an image.

```python
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.preprocessing import StandardScaler

# Load an image (you can replace this with your own image)
image = imread('example_image.jpg')

# Flatten the image to create a dataset of RGB color values
data = image.reshape((-1, 3))

# Standardize the data
data = StandardScaler().fit_transform(data)

# Estimate bandwidth (bandwidth selection)
bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=500)

# Create and fit the Mean Shift model
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(data)

# Get cluster labels
labels = ms.labels_

# Get cluster centers
cluster_centers = ms.cluster_centers_

# Reshape the labels to the shape of the original image
segmented_image = labels.reshape(image.shape[:2])

# Visualize the segmented image
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image, cmap='tab20')
plt.title("Mean Shift Segmentation")
plt.axis('off')

plt.show()
```

In this example:

- We load an image and flatten it to create a dataset of RGB color values.
- We standardize the data to ensure that each feature has a similar scale.
- We estimate the bandwidth parameter using the `estimate_bandwidth` function.
- We create and fit the Mean Shift model to cluster similar pixel colors.
- We reshape the cluster labels to the shape of the original image to create a segmented image.
- Finally, we visualize both the original image and the segmented image.

## Estimating the number of clusters with Mean Shift algorithm

- You can use the Mean Shift algorithm to estimate the number of clusters in a complex dataset where the number of clusters is not known in advance.

We'll use the `make_spiral` function from Scikit-Learn to generate this dataset and then apply Mean Shift to estimate the number of clusters:

```python
import numpy as np
from sklearn.datasets import make_s_curve
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate a 3D dataset with intertwined spirals
X, _ = make_s_curve(n_samples=1000, noise=0.05, random_state=42)

# Estimate bandwidth (bandwidth selection)
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

# Create and fit the Mean Shift model
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)

# Get cluster labels and cluster centers
labels = ms.labels_
cluster_centers = ms.cluster_centers_

# Plot the estimated clusters in 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

unique_labels = np.unique(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    cluster = X[labels == label]
    ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], c=[color], label=f'Cluster {label}')

ax.set_title("Mean Shift Clustering")
ax.legend()
plt.show()
```

In this example:

- We generate a complex 3D dataset with intertwined spirals using the `make_s_curve` function.

- We estimate the bandwidth parameter using the `estimate_bandwidth` function.

- We create and fit the Mean Shift model to the dataset.

- We visualize the estimated clusters in 3D, with each cluster represented by a unique color.

Mean Shift will automatically identify and estimate the number of clusters in the complex dataset. In this case, it should identify two intertwined spirals as separate clusters. You can adjust the parameters and dataset characteristics as needed for other complex datasets.

### Let's use a financial price dataset to estimate the number of clusters with the Mean Shift algorithm. 

- We'll use historical stock price data for multiple companies and apply Mean Shift to identify potential clusters based on similar price movements. 
- We'll use a simplified example with randomly generated data for illustration:

```python
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

# Generate synthetic financial price data for five companies
np.random.seed(0)
num_samples = 200
num_features = 1
data = np.random.randn(num_samples, num_features)

# Scale the data to simulate different price ranges
data = data * np.array([10, 20, 5, 15, 30])

# Estimate bandwidth (bandwidth selection)
bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=500)

# Create and fit the Mean Shift model
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(data)

# Get cluster labels and cluster centers
labels = ms.labels_
cluster_centers = ms.cluster_centers_

# Plot the estimated clusters
plt.figure(figsize=(10, 6))

unique_labels = np.unique(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    cluster = data[labels == label]
    plt.scatter(cluster, np.zeros_like(cluster), c=[color], label=f'Cluster {label}')

plt.title("Mean Shift Clustering of Financial Price Data")
plt.xlabel("Price")
plt.legend()
plt.grid()
plt.show()
```

In this example:

- We generate synthetic financial price data for five companies, each with its own price range.

- We estimate the bandwidth parameter using the `estimate_bandwidth` function.

- We create and fit the Mean Shift model to the financial price data.

- We visualize the estimated clusters, where each cluster represents a group of companies with similar price movements.

This example illustrates how Mean Shift can be applied to financial price data to identify clusters of companies with similar price behaviors, which could be useful for portfolio management or trend analysis. You can replace the synthetic data with real financial price data for more meaningful insights.

## Silhouette scores

- A metric used to measure the quality of clusters created by a clustering algorithm, such as K-Means or Mean Shift. - It provides a way to quantify how well-separated the clusters are, with higher values indicating better-defined clusters. 
- The Silhouette Score ranges from -1 to 1, with the following interpretation:
  - A score near 1 indicates that the data points are well-clustered, and they are far from the neighboring clusters.
  - A score of 0 indicates overlapping clusters, where data points on cluster boundaries may be assigned to multiple clusters.
  - A score near -1 indicates that data points have been assigned to the wrong clusters, and they are closer to neighboring clusters than to their assigned cluster.

The Silhouette Score is calculated for each data point and is based on two measures:

1. **a(i)**: The average distance from the i-th data point to other data points in the same cluster. This measures the cohesion of the data point within its cluster.

2. **b(i)**: The smallest average distance from the i-th data point to data points in a different cluster, minimized over clusters. This measures the separation of the data point from other clusters.

The Silhouette Score for the i-th data point is defined as:

\[silhouette\_score(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}\]

The overall Silhouette Score for a clustering is the mean of the Silhouette Scores for all data points. A higher overall score indicates that the clustering is more appropriate.

In practice, you can use the Silhouette Score to compare different clustering results or to find the optimal number of clusters for a dataset. It helps in assessing the compactness and separation of clusters and can guide you in choosing the best clustering approach.

## Silhouette Score example

- Let's use a complex dataset and calculate the Silhouette Score to evaluate the quality of the clusters. 
- We'll use the Iris dataset, which is a commonly used dataset in machine learning and clustering tasks. 
- In this example, we'll perform K-Means clustering and calculate the Silhouette Score to assess the clustering quality:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
data = load_iris()
X = data.data  # Features
y = data.target  # True labels

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize a list to store silhouette scores
silhouette_scores = []

# Range of cluster numbers to consider
cluster_range = range(2, 11)

# Perform K-Means clustering for different cluster numbers
for num_clusters in cluster_range:
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate silhouette score for the current clustering
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot the Silhouette Score vs. Number of Clusters
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.grid(True)
plt.show()
```

In this example:

- We load the Iris dataset, which contains four features (sepal length, sepal width, petal length, and petal width) for three different species of iris flowers.

- We standardize the data using `StandardScaler` to ensure that each feature has a similar scale.

- We perform K-Means clustering for a range of cluster numbers (from 2 to 10) and calculate the Silhouette Score for each clustering.

- We plot the Silhouette Score vs. the number of clusters to visualize how the score changes as we vary the number of clusters.

- The plot will show how the Silhouette Score changes with different numbers of clusters. 
- You should look for the number of clusters that maximizes the Silhouette Score because it indicates the best clustering solution in terms of cluster separation and cohesion. 
- In this example, the optimal number of clusters can be determined based on the highest Silhouette Score.

## Estimating the quality of clustering with silhouette scores

- To estimate the quality of clustering using Silhouette Scores. In this example, we'll generate a synthetic dataset with known clusters and then evaluate the clustering quality using Silhouette Scores. 
- This will help demonstrate how Silhouette Scores can be used to assess the clustering performance.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Generate a synthetic dataset with three well-separated clusters
data, true_labels = make_blobs(n_samples=300, centers=3, random_state=42)

# Initialize a list to store silhouette scores
silhouette_scores = []

# Range of cluster numbers to consider
cluster_range = range(2, 11)

# Perform K-Means clustering for different cluster numbers
for num_clusters in cluster_range:
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    
    # Calculate silhouette score for the current clustering
    silhouette_avg = silhouette_score(data, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot the Silhouette Score vs. Number of Clusters
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.grid(True)
plt.show()
```

In this example:

- We generate a synthetic dataset with three well-separated clusters using the `make_blobs` function. The `true_labels` variable contains the ground truth cluster assignments.

- We initialize an empty list to store Silhouette Scores.

- We perform K-Means clustering for a range of cluster numbers (from 2 to 10) and calculate the Silhouette Score for each clustering.

- We plot the Silhouette Score vs. the number of clusters to visualize how the score changes as we vary the number of clusters.

Since we generated the data with known clusters, we can compare the Silhouette Scores with the ground truth labels to assess the clustering quality. In practice, when working with real datasets where true labels are not available, the Silhouette Score can still be used as an indicator of clustering quality, and the goal is to choose the number of clusters that maximizes this score.

## Gaussian Mixture Models

- A probabilistic model used for clustering and density estimation. 
- Unlike K-Means, which assumes that data points belong to a single cluster, GMMs allow data points to belong to multiple clusters with varying degrees of membership. 
- GMMs are based on a mixture of Gaussian distributions, where each Gaussian component represents a cluster.

1. **Mixture of Gaussians:** A GMM models the data as a weighted sum of multiple Gaussian distributions (also known as Gaussian components or clusters). Each Gaussian component has its own mean and covariance matrix.

2. **Parameters:** The parameters of a GMM include the means, covariances, and mixing coefficients (weights) for each Gaussian component. These parameters are learned from the data using techniques like the Expectation-Maximization (EM) algorithm.

3. **Probability Density Function:** The probability density function (PDF) of a GMM is a linear combination of the PDFs of its Gaussian components, weighted by the mixing coefficients. Mathematically, it can be expressed as:
   
   \[P(x) = \sum_{i=1}^{K} \pi_i \cdot \mathcal{N}(x | \mu_i, \Sigma_i)\]

   - \(K\) is the number of Gaussian components.
   - \(\pi_i\) is the mixing coefficient of the \(i\)-th component (\(\sum_{i=1}^{K} \pi_i = 1\)).
   - \(\mathcal{N}(x | \mu_i, \Sigma_i)\) is the Gaussian distribution with mean \(\mu_i\) and covariance \(\Sigma_i\).

4. **Clustering:** In clustering tasks, GMMs assign data points to one or more clusters based on their probabilities of belonging to each component. Data points can belong to multiple clusters with different probabilities.

5. **Density Estimation:** GMMs can be used for density estimation, where they model the underlying probability distribution of the data. This can be useful for generative modeling and anomaly detection.

6. **EM Algorithm:** The EM algorithm is commonly used to estimate the parameters of GMMs. It alternates between the Expectation (E) step, where the probabilities of data points belonging to each component are computed, and the Maximization (M) step, where the parameters of each component are updated to maximize the likelihood of the data.

7. **Initialization:** GMMs require careful initialization of parameters, often using methods like K-Means initialization or random initialization, followed by EM optimization.

**In summary**
- Gaussian Mixture Models are versatile and can capture complex data distributions, making them suitable for a wide range of applications, including clustering, density estimation, image segmentation, and generative modeling.
- They are particularly useful when data points may belong to multiple overlapping clusters.

## Building a classifier based on Gaussian Mixture Models

- Typically involves modeling the probability distribution of each class using GMMs and then using these models to classify new data points. 
- In this example, we'll create a classifier based on GMMs using a synthetic dataset with multiple classes. 
- We'll generate a synthetic dataset with known classes, fit GMMs to each class, and then classify new data points based on the GMM likelihoods.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a synthetic dataset with three classes
data, true_labels = make_blobs(n_samples=300, centers=3, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, true_labels, test_size=0.2, random_state=42)

# Fit Gaussian Mixture Models (GMMs) to each class
gmm_models = []
for class_label in np.unique(y_train):
    # Select data points for the current class
    class_data = X_train[y_train == class_label]
    
    # Fit a GMM to the data of the current class
    gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
    gmm.fit(class_data)
    
    # Add the trained GMM to the list of models
    gmm_models.append(gmm)

# Classify new data points using GMM likelihoods
predicted_labels = []
for data_point in X_test:
    likelihoods = [gmm.score_samples(data_point.reshape(1, -1))[0] for gmm in gmm_models]
    predicted_label = np.argmax(likelihoods)
    predicted_labels.append(predicted_label)

# Calculate accuracy
accuracy = accuracy_score(y_test, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Plot the true labels and predicted labels
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=predicted_labels, cmap='viridis', marker='o', s=50, label='Predicted Labels')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='Set1', marker='x', s=50, label='True Labels')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('GMM Classifier Results')
plt.legend()
plt.grid(True)
plt.show()
```

In this example:

- We generate a synthetic dataset with three classes using the `make_blobs` function.

- We split the dataset into training and testing sets.

- For each class in the training set, we fit a GMM with one component (representing each class) using the `GaussianMixture` class from Scikit-Learn.

- To classify new data points, we compute the likelihoods of each data point belonging to each class's GMM using the `score_samples` method.

- We classify each data point based on the GMM with the highest likelihood.

- We calculate the accuracy of the classifier using the true labels and predicted labels.

- Finally, we visualize the true labels and predicted labels on a scatter plot.

## Affinity Propagation model

- A clustering algorithm used in machine learning and data analysis. 
- Unlike traditional clustering algorithms like K-Means or Hierarchical Clustering, Affinity Propagation does not require specifying the number of clusters in advance. 
- Instead, it identifies clusters automatically based on similarity measures between data points.

1. **Similarity Matrix:** Affinity Propagation starts with a similarity matrix that quantifies the similarity between pairs of data points. The similarity can be computed using various metrics, such as Euclidean distance, cosine similarity, or other domain-specific measures.

2. **Responsibility and Availability:** The algorithm iteratively updates two matrices:
   - **Responsibility Matrix (R):** R(i, k) measures the suitability of data point k to be the exemplar (cluster center) for data point i. It reflects how well data point k can represent data point i.
   - **Availability Matrix (A):** A(i, k) represents the accumulated evidence that data point i should choose data point k as its exemplar. It considers how appropriate data point k is to be the exemplar for other data points.

3. **Message Passing:** In each iteration, data points exchange "messages" (information) with each other to update the Responsibility and Availability matrices. The messages are computed based on the current values of these matrices and the similarity matrix.

4. **Exemplars and Clustering:** Data points select their exemplars based on the updated Responsibility and Availability matrices. The exemplars are data points that are most likely to represent clusters. Data points that share the same exemplar are considered part of the same cluster.

5. **Convergence:** The algorithm iteratively updates the Responsibility and Availability matrices until convergence. Convergence is typically achieved when the matrices stabilize, and the clusters remain relatively unchanged between iterations.

6. **Number of Clusters:** Affinity Propagation does not require specifying the number of clusters in advance. The algorithm automatically determines the number of clusters based on the data and similarity matrix.

- Affinity Propagation has several applications in various domains, including image segmentation, natural language processing, and bioinformatics. 
- It is particularly useful when the number of clusters is unknown or when clusters have varying sizes and shapes. 
- However, it can be computationally expensive, especially for large datasets, and its performance may depend on the choice of similarity metric and parameters.

## Finding subgroups in stock market using Affinity Propagation model

- Let's use the Affinity Propagation model to find subgroups within a complex dataset. 
- In this example, we'll apply Affinity Propagation to a synthetic dataset with subgroups and visualize the identified clusters.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score

# Generate a synthetic dataset with subgroups
data, true_labels = make_blobs(n_samples=300, centers=6, cluster_std=1.0, random_state=42)

# Apply Affinity Propagation
affinity_propagation = AffinityPropagation(damping=0.7)
cluster_labels = affinity_propagation.fit_predict(data)

# Calculate the silhouette score to evaluate the quality of clusters
silhouette_avg = silhouette_score(data, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.2f}")

# Visualize the clusters
plt.figure(figsize=(10, 6))
unique_labels = np.unique(cluster_labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    cluster = data[cluster_labels == label]
    plt.scatter(cluster[:, 0], cluster[:, 1], c=[color], label=f'Cluster {label}')

plt.title('Affinity Propagation Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
```

In this example:

- We generate a synthetic dataset with subgroups using the `make_blobs` function. The `centers` parameter is set to 6 to create six distinct subgroups.

- We apply the Affinity Propagation clustering algorithm to the dataset using the `AffinityPropagation` class from Scikit-Learn. The `damping` parameter is set to control the extent of damping during message passing.

- We calculate the Silhouette Score to evaluate the quality of the identified clusters. A higher Silhouette Score indicates better separation between clusters.

- Finally, we visualize the identified clusters using a scatter plot, with each cluster represented by a unique color.

- Affinity Propagation is capable of finding subgroups within complex datasets, and it automatically determines the number of clusters based on the data and similarity measures. 
- You can adjust the damping parameter and other hyperparameters to fine-tune the clustering results based on your dataset's characteristics.

## Segmenting the market based on shopping patterns

