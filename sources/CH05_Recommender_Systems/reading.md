# Recommender Systems

- Creating a training pipeline
- Extracting the nearest neighbors
- Building a K Nearest Neighbors classifier
- Computing similarity scores
- Finding similar users using collaborative filtering
- Building a movie recommendation system

## Table of Contents

- [Recommender Systems](#recommender-systems)
  - [Table of Contents](#table-of-contents)
  - [What are Recommender Systems](#what-are-recommender-systems)
  - [Creating a training pipeline](#creating-a-training-pipeline)
    - [overview](#overview)
    - [in sklearn](#in-sklearn)
    - [sklearn.pipeline class](#sklearnpipeline-class)
  - [Extracting the nearest neighbors](#extracting-the-nearest-neighbors)
  - [K-Nearest Neighbors classifier](#k-nearest-neighbors-classifier)
    - [Knn with scikit-learn](#knn-with-scikit-learn)
    - [Knn example](#knn-example)
  - [Computing similarity scores](#computing-similarity-scores)
  - [collaborative filtering](#collaborative-filtering)
  - [recommendation system](#recommendation-system)
  - [recommendation system example](#recommendation-system-example)
  - [Code example](#code-example)
  
## What are Recommender Systems

- Also known as recommendation systems or recommenders, are a class of machine learning and information retrieval techniques designed to provide personalized suggestions or recommendations to users. 
- These systems are widely used in various applications to help users discover items or content that are likely to be of interest to them. 
- Recommender systems are especially prevalent in e-commerce, content streaming, social media, and online advertising platforms. 
- play a vital role in enhancing user experience, increasing engagement, and driving sales and user satisfaction in many online platforms
- help users discover new products, movies, music, articles, and other items tailored to their preferences, ultimately leading to improved user retention and business revenue.
- There are primarily two types of recommender systems:

1. **Collaborative Filtering:**
   - Collaborative filtering is based on the idea that users who have shown similar preferences or behavior in the past will likely have similar preferences in the future.
   - It relies on user-item interaction data, such as ratings, reviews, or purchase history, to make recommendations.
   - Collaborative filtering techniques include user-based, item-based, and matrix factorization methods.

2. **Content-Based Filtering:**
   - Content-based filtering recommends items to users based on the characteristics or features of the items themselves and the user's past preferences.
   - It relies on item descriptions, metadata, or content attributes to make recommendations.
   - Content-based filtering methods involve comparing the content features of items to the user's profile or preferences.

Additionally, there are hybrid recommender systems that combine collaborative filtering and content-based filtering to leverage the strengths of both approaches. These systems aim to provide more accurate and diverse recommendations.

Here are some key components and concepts related to recommender systems:

- **User Profile:** A user's profile is a representation of their preferences, behaviors, or characteristics. It can include historical interactions, ratings, demographic information, and explicit or implicit feedback.

- **Item Profile:** An item's profile consists of its attributes, features, or content descriptors. These can include textual descriptions, genre tags, and other metadata.

- **Similarity Metrics:** Recommender systems use various similarity metrics to measure the similarity between users, items, or their profiles. Common metrics include cosine similarity, Pearson correlation, and Jaccard similarity.

- **Recommendation Algorithms:** There are various recommendation algorithms, including user-based and item-based collaborative filtering, matrix factorization methods (e.g., Singular Value Decomposition), and content-based filtering algorithms (e.g., TF-IDF or neural networks).

- **Cold Start Problem:** The cold start problem occurs when a recommender system needs to make recommendations for new users or items with limited historical data. Hybrid models and content-based filtering can help address this problem.

- **Evaluation Metrics:** To assess the quality of recommendations, recommender systems use evaluation metrics such as Mean Absolute Error (MAE), Root Mean Square Error (RMSE), Precision, Recall, and F1-score.

- **Scalability:** Scalability is a critical consideration, as recommender systems need to handle large datasets and provide real-time recommendations. Distributed computing and algorithm optimizations are often employed.

## Creating a training pipeline

### overview

- In the context of AI and machine learning (ML), a training pipeline, often referred to as a machine learning **pipeline** or **workflow**, is a structured and automated series of steps designed to process and transform data, train machine learning models, and prepare them for deployment. 
- Training pipelines are crucial for managing the end-to-end process of developing, training, and deploying ML models efficiently and effectively. Here are the key components and steps typically found in a training pipeline:

1. **Data Ingestion:**
   - The pipeline begins with the collection and ingestion of raw data from various sources, such as databases, APIs, files, or streaming data.

2. **Data Preprocessing:**
   - Raw data is often messy and needs preprocessing. This step includes tasks like cleaning, handling missing values, encoding categorical variables, scaling, and feature engineering.

3. **Data Splitting:**
   - The data is split into training, validation, and testing sets to assess model performance and avoid overfitting.

4. **Feature Selection/Extraction:**
   - Features are selected or extracted to create meaningful input data for the models. This step may involve dimensionality reduction techniques like Principal Component Analysis (PCA).

5. **Model Selection:**
   - Choose appropriate ML algorithms or models based on the problem type (classification, regression, clustering, etc.) and domain knowledge.

6. **Hyperparameter Tuning:**
   - Optimize model hyperparameters through techniques like grid search, random search, or Bayesian optimization.

7. **Model Training:**
   - Train the selected model(s) on the training data using the best hyperparameters.

8. **Model Evaluation:**
   - Assess the model's performance on the validation dataset using evaluation metrics like accuracy, precision, recall, F1-score, or Mean Squared Error (MSE).

9. **Model Selection (Optional):**
   - If multiple models were trained, select the best-performing model based on validation results.

10. **Model Testing:**
    - Evaluate the chosen model(s) on the test dataset to estimate how well it will perform in real-world scenarios.

11. **Model Deployment:**
    - If the model meets performance criteria, deploy it to a production environment where it can make predictions on new data.

12. **Monitoring and Maintenance:**
    - Continuously monitor the deployed model's performance, drift, and potential issues. Retrain the model periodically with new data to keep it up to date.

13. **Documentation and Reporting:**
    - Maintain documentation of the pipeline's components, data sources, preprocessing steps, models, and results for future reference and auditing.

14. **Automation and Orchestration:**
    - Automate the entire pipeline to enable reproducibility and scalability. Use orchestration tools like Apache Airflow or Kubernetes for managing workflow scheduling and dependencies.

15. **Logging and Error Handling:**
    - Implement logging and error-handling mechanisms to track the pipeline's progress, catch issues, and facilitate debugging.

16. **Deployment to Production:**
    - Deploy the trained model(s) to production, which may involve creating APIs or integrating the model with other software systems.

### in sklearn

- Creating a training pipeline for a recommender system using scikit-learn involves several steps, including data preprocessing, model selection, training, and evaluation.
  
1. **Data Preparation:**
   - Load and preprocess the user-item interaction data, which typically includes user IDs, item IDs, and user-item interaction scores (e.g., ratings).
   - Perform any necessary data cleaning, such as handling missing values or removing duplicates.

2. **Splitting the Data:**
   - Split the data into training and testing sets. Common splitting ratios are 80-20 or 70-30, depending on the dataset size.

3. **Feature Engineering:**
   - Create user and item feature representations, if available. These features can include demographic information, item attributes, or user behavior history.
   - Perform one-hot encoding or feature embedding as needed.

4. **Model Selection:**
   - Choose an appropriate collaborative filtering algorithm from scikit-learn or other libraries. Common choices include:
     - User-based collaborative filtering (`sklearn.neighbors.NearestNeighbors`)
     - Item-based collaborative filtering (`sklearn.neighbors.NearestNeighbors`)
     - Matrix factorization (e.g., Singular Value Decomposition with `sklearn.decomposition.TruncatedSVD`)
     - Alternating Least Squares (ALS) matrix factorization (using libraries like Spark MLlib)
     - Neural collaborative filtering models (using libraries like TensorFlow or PyTorch)

5. **Hyperparameter Tuning:**
   - If applicable, perform hyperparameter tuning using techniques like cross-validation to find the best model hyperparameters.

6. **Model Training:**
   - Train the selected model on the training data using the chosen algorithm and hyperparameters.

7. **Model Evaluation:**
   - Evaluate the trained model's performance on the testing data using appropriate evaluation metrics. Common metrics include Mean Absolute Error (MAE), Root Mean Square Error (RMSE), Precision, Recall, and F1-score.

8. **Deployment:**
   - If the model meets performance criteria, deploy it to make real-time recommendations.
   - For deployment, consider using a production-ready framework like Flask or FastAPI to create an API endpoint for making recommendations.

9. **Monitoring and Maintenance:**
   - Continuously monitor the model's performance in the production environment.
   - Implement mechanisms for retraining the model periodically with new data to keep recommendations up to date.

Here's a simplified code example using scikit-learn's `NearestNeighbors` for user-based collaborative filtering:

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Assuming you have user-item interaction data in the form of a user-item matrix
user_item_matrix = np.array([[1, 0, 3, 4, 0],
                             [0, 0, 2, 0, 0],
                             [4, 0, 0, 1, 2],
                             [3, 0, 4, 0, 0]])

# Create a NearestNeighbors model (user-based)
knn_model = NearestNeighbors(n_neighbors=2, metric='cosine', algorithm='brute')
knn_model.fit(user_item_matrix)

# Make recommendations for a user (e.g., user 0)
user_index = 0
distances, indices = knn_model.kneighbors([user_item_matrix[user_index]], n_neighbors=3)

# Recommend items based on similar users
recommended_items = user_item_matrix[indices[0]]
```

### sklearn.pipeline class

- The `sklearn.pipeline` module in scikit-learn provides a convenient and powerful way to create and manage machine learning pipelines. 
- A machine learning pipeline is a sequence of data processing steps, where each step can include data preprocessing, feature extraction, feature selection, and model training. 
- Pipelines are particularly useful for structuring and automating the machine learning workflow. 
- The `Pipeline` class in scikit-learn allows you to define and execute such workflows efficiently.

**Components of `sklearn.pipeline` module:**

1. **Pipeline:** The `Pipeline` class itself represents a sequence of data processing steps, where each step is specified as a tuple with a name and an estimator (transformer or model). The `Pipeline` class ensures that the steps are executed in the correct order. Pipelines can be used for both classification and regression tasks.

2. **FeatureUnion:** The `FeatureUnion` class allows you to combine the results of multiple transformer objects (e.g., feature extraction methods) into a single set of features. This is useful when you want to apply different feature extraction techniques to the same input data and then concatenate their outputs.

**Key Features and Benefits:**

- **Convenience:** Pipelines make it easier to organize, document, and maintain complex machine learning workflows. You can define a sequence of data transformations and model training in a single object.

- **Consistency:** Pipelines ensure that the data processing steps are executed in the correct order, preventing common mistakes like data leakage.

- **Easier Cross-Validation:** When performing cross-validation or grid search, pipelines are compatible with scikit-learn's `cross_val_score` and `GridSearchCV`, making it straightforward to optimize hyperparameters.

- **Code Reusability:** Pipelines encourage code reusability because you can define and reuse the same pipeline for multiple datasets or tasks.

- **Scalability:** Pipelines are useful when dealing with large datasets or complex workflows, as they allow you to efficiently manage data processing and model training.

**Creating a Simple Pipeline:**

Here's an example of how to create a simple classification pipeline using scikit-learn's `Pipeline` class:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Define the individual steps as tuples (name, estimator)
steps = [
    ('scaler', StandardScaler()),   # Standardize features
    ('pca', PCA(n_components=2)),  # Apply PCA for dimensionality reduction
    ('svm', SVC(kernel='rbf'))     # Support Vector Machine classifier
]

# Create a pipeline by passing the steps
pipeline = Pipeline(steps)

# Fit the pipeline to the data and make predictions
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

In this example:

- We define three steps in the pipeline: standardizing the features, applying Principal Component Analysis (PCA), and using a Support Vector Machine (SVM) classifier.

- The pipeline ensures that these steps are executed in the specified order when you call `fit()` or `predict()`.

- You can then fit the pipeline to your training data and use it to make predictions.

## Extracting the nearest neighbors

- A fundamental technique used in recommender systems, clustering, anomaly detection, and various other machine learning tasks. 
- In the context of recommender systems, nearest neighbors are used to find items or users that are similar to a given item or user, allowing you to make personalized recommendations.
- Here, we'll discuss the concept of extracting nearest neighbors in detail, including advanced best practices and provide an example using a complex dataset.

**Concept of Extracting Nearest Neighbors:**

- The idea behind extracting nearest neighbors is to identify data points that are close or similar to a target data point in a feature space. 
- The term "neighbor" typically refers to the k-nearest neighbors (k-NN), where k is the number of neighbors to be retrieved. 
- Nearest neighbors can be determined using various **distance** or **similarity** metrics, such as **Euclidean** distance, **cosine** similarity, or **Jaccard** similarity, depending on the **type** of data and the **problem** at hand.

**Advanced Best Practices:**

1. **Feature Selection and Scaling:** Before extracting nearest neighbors, it's crucial to perform feature selection and scaling to ensure that all features contribute equally to the distance calculations. Use techniques like PCA for dimensionality reduction and standardization to scale features.

2. **Distance Metric Selection:** Choose an appropriate distance or similarity metric that suits your data and problem. For example, Euclidean distance works well for continuous numerical data, while cosine similarity is suitable for text-based or sparse data.

3. **Efficient Data Structures:** For large datasets, consider using data structures like KD-trees or Ball trees to speed up nearest neighbor searches. These structures can significantly improve computational efficiency compared to brute-force searching.

4. **Optimal k:** Determine the optimal value of k based on cross-validation or grid search. A smaller k may capture local patterns but be noisy, while a larger k may provide more general recommendations but miss local nuances.

5. **Regularization:** Implement regularization techniques like L2 regularization to mitigate the effects of noisy or irrelevant features in the distance calculations.

**Example with a Complex Dataset:**

Let's demonstrate the extraction of nearest neighbors using scikit-learn on a complex dataset. We'll use the famous "Iris" dataset, which contains measurements of iris flowers:

```python
from sklearn.datasets import load_iris
from sklearn.neighbors import NearestNeighbors

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Create a NearestNeighbors model with a specified distance metric (Euclidean)
k = 3
nn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')

# Fit the model to the data
nn_model.fit(X)

# Define a query point for which we want to find the nearest neighbors
query_point = [5.1, 3.5, 1.4, 0.2]

# Find the k-nearest neighbors of the query point
distances, indices = nn_model.kneighbors([query_point])

# Print the indices of nearest neighbors and their distances
print("Indices of Nearest Neighbors:", indices)
print("Distances to Nearest Neighbors:", distances)
```

In this example:

- We load the Iris dataset and create a `NearestNeighbors` model with Euclidean distance as the distance metric.

- We fit the model to the data.

- We define a query point (representing an iris flower) for which we want to find the k-nearest neighbors.

- We use the `kneighbors` method to find the indices and distances of the k-nearest neighbors to the query point.

## K-Nearest Neighbors classifier

1. **Load the Dataset:**
   - Start by loading your dataset, which should include both features (attributes) and labels (the target variable you want to predict). Ensure that the dataset is properly formatted.

2. **Data Preprocessing:**
   - If necessary, perform data preprocessing steps such as handling missing values, encoding categorical features, and scaling numerical features. Data preprocessing ensures that the data is in a suitable format for the K-NN algorithm.

3. **Split the Dataset:**
   - Divide your dataset into two subsets: a training set and a testing (or validation) set. A common split ratio is 80% for training and 20% for testing, but you can adjust this based on your dataset size and requirements.

4. **Choose the Number of Neighbors (k):**
   - Decide on the number of nearest neighbors (k) that the algorithm should consider when making predictions. You can experiment with different values of k to find the optimal one for your dataset.

5. **Select a Distance Metric:**
   - Choose a suitable distance metric (e.g., Euclidean distance, Manhattan distance, cosine similarity) to measure the similarity between data points. The choice of metric depends on the nature of your data and the problem you're solving.

6. **Initialize the K-NN Classifier:**
   - Create an instance of the K-NN classifier, specifying the value of k and the chosen distance metric. In scikit-learn, this can be done using the `KNeighborsClassifier` class.

7. **Fit the Model:**
   - Train the K-NN classifier by fitting it to the training data. This step involves storing the training data and labels so that the algorithm can use them to make predictions.

8. **Make Predictions:**
   - Use the trained K-NN classifier to make predictions on the testing (or validation) data. The algorithm finds the k-nearest neighbors of each test data point in the training set and assigns a label based on majority voting among these neighbors.

9. **Evaluate the Model:**
   - Assess the performance of the K-NN classifier by calculating various evaluation metrics, such as accuracy, precision, recall, F1-score, and confusion matrix. These metrics help you understand how well the model is performing on unseen data.

10. **Hyperparameter Tuning:**
    - Experiment with different values of k and distance metrics to optimize the model's performance. You can use techniques like cross-validation and grid search to find the best hyperparameters.

11. **Deploy the Model (Optional):**
    - If the K-NN classifier meets your performance criteria, you can deploy it for making predictions on new, unseen data in a production environment.

12. **Monitoring and Maintenance (Production):**
    - In a production environment, monitor the model's performance and consider retraining it periodically with fresh data to ensure it remains accurate and up to date.

### Knn with scikit-learn

- A focus on using scikit-learn library functions:

1. **Hyperparameter Tuning with Grid Search (Optional):**
    - To fine-tune hyperparameters such as the number of neighbors (k) and the distance metric, you can use scikit-learn's `GridSearchCV`. This tool performs an exhaustive search over a predefined parameter grid and helps you find the best combination of hyperparameters through cross-validation.

   ```python
   from sklearn.model_selection import GridSearchCV

   # Define a parameter grid for hyperparameter tuning
   param_grid = {
       'n_neighbors': [3, 5, 7],           # Different values of k
       'metric': ['euclidean', 'manhattan']  # Different distance metrics
   }

   # Create a K-NN classifier
   knn_classifier = KNeighborsClassifier()

   # Perform grid search with cross-validation
   grid_search = GridSearchCV(knn_classifier, param_grid, cv=5)
   grid_search.fit(X_train, y_train)

   # Get the best hyperparameters
   best_k = grid_search.best_params_['n_neighbors']
   best_metric = grid_search.best_params_['metric']
   ```

   After running grid search, you can extract the best values of k and the distance metric based on cross-validation performance.

2. **Final Model Training:**
    - Once you've identified the optimal hyperparameters, retrain the K-NN classifier using these values on the entire training dataset.

   ```python
   # Create a K-NN classifier with the best hyperparameters
   final_knn_classifier = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric)

   # Fit the final model to the entire training dataset
   final_knn_classifier.fit(X_train, y_train)
   ```

3. **Evaluate the Final Model:**
    - Use the final trained K-NN classifier to make predictions on the testing (or validation) data and evaluate its performance using metrics like accuracy, precision, recall, F1-score, and confusion matrix.

   ```python
   # Make predictions using the final model
   y_pred = final_knn_classifier.predict(X_test)

   # Evaluate the final model's performance
   accuracy = accuracy_score(y_test, y_pred)
   classification_report = classification_report(y_test, y_pred)
   ```

4. **Deployment (Optional):**
    - If the final K-NN classifier meets your performance requirements, you can deploy it in a production environment to make predictions on new, unseen data.

5. **Monitoring and Maintenance (Production):**
    - In a production environment, monitor the model's performance regularly and consider retraining it with updated data to ensure continued accuracy.

### Knn example

- We'll use the Iris dataset, which is a popular dataset for classification tasks. 
- The goal is to classify iris flowers into three species based on their measurements.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a K-NN classifier with a specified number of neighbors (e.g., k=3)
k = 3
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Fit the classifier to the training data
knn_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn_classifier.predict(X_test)

# Evaluate the classifier's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
```

## Computing similarity scores

Computing similarity scores is a fundamental concept in various fields, including information retrieval, recommendation systems, and clustering. Similarity scores quantify how alike two data points, items, or documents are based on their attributes or features. The choice of similarity metric depends on the type of data and the problem you're trying to solve. Here, we'll discuss the theory behind computing similarity scores and provide scikit-learn code examples for some common similarity metrics.

**Theory of Computing Similarity Scores:**

There are several similarity metrics commonly used to compute similarity scores. Here are some of the most widely used ones:

1. **Cosine Similarity:**
   - Cosine similarity measures the cosine of the angle between two vectors in a multidimensional space. It is often used for text data and is particularly useful when the data is sparse (e.g., text documents represented as term frequency-inverse document frequency vectors).
   - Cosine similarity ranges from -1 (perfectly dissimilar) to 1 (perfectly similar), with 0 indicating no similarity.

2. **Euclidean Distance:**
   - Euclidean distance calculates the straight-line distance between two data points in a Euclidean space. It is suitable for continuous numerical data.
   - Smaller Euclidean distances indicate higher similarity.

3. **Manhattan Distance (City Block Distance):**
   - Manhattan distance measures the distance between two points by summing the absolute differences of their coordinates. It is often used when movement is constrained to grid-based paths.
   - Similar to Euclidean distance, smaller Manhattan distances indicate higher similarity.

4. **Jaccard Similarity:**
   - Jaccard similarity is used for comparing sets or binary data. It computes the size of the intersection of two sets divided by the size of their union.
   - Jaccard similarity ranges from 0 (no similarity) to 1 (perfect similarity).

5. **Pearson Correlation Coefficient:**
   - Pearson correlation measures the linear relationship between two variables. It is used for continuous numerical data and assesses how well the data points fit a straight line.
   - Pearson correlation ranges from -1 (perfectly negatively correlated) to 1 (perfectly positively correlated), with 0 indicating no correlation.

**Scikit-Learn Code Examples:**

Let's provide scikit-learn code examples for computing similarity scores using Cosine Similarity and Euclidean Distance. We'll use synthetic data for illustration:

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Sample data (2D array)
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# Compute cosine similarity
cosine_sim = cosine_similarity(data)

# Compute Euclidean distances
euclidean_dist = euclidean_distances(data)

print("Cosine Similarity:")
print(cosine_sim)
print("\nEuclidean Distances:")
print(euclidean_dist)
```

In this code:

- We create a sample data matrix `data` with three rows (data points) and three columns (features).

- We use scikit-learn's `cosine_similarity` function to compute the cosine similarity matrix `cosine_sim` for the data.

- We use `euclidean_distances` to compute the Euclidean distance matrix `euclidean_dist`.

## collaborative filtering

Computing similarity scores is a fundamental concept in various fields, including information retrieval, recommendation systems, and clustering. Similarity scores quantify how alike two data points, items, or documents are based on their attributes or features. The choice of similarity metric depends on the type of data and the problem you're trying to solve. Here, we'll discuss the theory behind computing similarity scores and provide scikit-learn code examples for some common similarity metrics.

**Theory of Computing Similarity Scores:**

There are several similarity metrics commonly used to compute similarity scores. Here are some of the most widely used ones:

1. **Cosine Similarity:**
   - Cosine similarity measures the cosine of the angle between two vectors in a multidimensional space. It is often used for text data and is particularly useful when the data is sparse (e.g., text documents represented as term frequency-inverse document frequency vectors).
   - Cosine similarity ranges from -1 (perfectly dissimilar) to 1 (perfectly similar), with 0 indicating no similarity.

2. **Euclidean Distance:**
   - Euclidean distance calculates the straight-line distance between two data points in a Euclidean space. It is suitable for continuous numerical data.
   - Smaller Euclidean distances indicate higher similarity.

3. **Manhattan Distance (City Block Distance):**
   - Manhattan distance measures the distance between two points by summing the absolute differences of their coordinates. It is often used when movement is constrained to grid-based paths.
   - Similar to Euclidean distance, smaller Manhattan distances indicate higher similarity.

4. **Jaccard Similarity:**
   - Jaccard similarity is used for comparing sets or binary data. It computes the size of the intersection of two sets divided by the size of their union.
   - Jaccard similarity ranges from 0 (no similarity) to 1 (perfect similarity).

5. **Pearson Correlation Coefficient:**
   - Pearson correlation measures the linear relationship between two variables. It is used for continuous numerical data and assesses how well the data points fit a straight line.
   - Pearson correlation ranges from -1 (perfectly negatively correlated) to 1 (perfectly positively correlated), with 0 indicating no correlation.

**Scikit-Learn Code Examples:**

Let's provide scikit-learn code examples for computing similarity scores using Cosine Similarity and Euclidean Distance. We'll use synthetic data for illustration:

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Sample data (2D array)
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# Compute cosine similarity
cosine_sim = cosine_similarity(data)

# Compute Euclidean distances
euclidean_dist = euclidean_distances(data)

print("Cosine Similarity:")
print(cosine_sim)
print("\nEuclidean Distances:")
print(euclidean_dist)
```

In this code:

- We create a sample data matrix `data` with three rows (data points) and three columns (features).

- We use scikit-learn's `cosine_similarity` function to compute the cosine similarity matrix `cosine_sim` for the data.

- We use `euclidean_distances` to compute the Euclidean distance matrix `euclidean_dist`.

## recommendation system

Building a recommendation system with a complex dataset typically involves more advanced techniques to provide meaningful recommendations to users. One common approach is collaborative filtering, which relies on user behavior and preferences to generate recommendations. In this example, we'll use the MovieLens dataset, which is a popular dataset for recommendation system tasks. We'll implement a collaborative filtering recommendation system using matrix factorization with Singular Value Decomposition (SVD) and scikit-learn. Here are the steps:

**Step 1: Load and Explore the Dataset:**
- Load the MovieLens dataset, which includes user ratings for movies.
- Explore the dataset to understand its structure and features.

**Step 2: Data Preprocessing:**
- Preprocess the dataset to ensure it's in a suitable format for recommendation.
- You may need to handle missing values, encode categorical data, and filter out less relevant information.

**Step 3: Create the User-Item Interaction Matrix:**
- Transform the dataset into a user-item interaction matrix, where rows represent users, columns represent items (movies), and the values represent user ratings.

**Step 4: Matrix Factorization with SVD:**
- Use Singular Value Decomposition (SVD) to factorize the user-item interaction matrix into three matrices: U (user features), Σ (diagonal matrix of singular values), and V^T (item features).
- Determine the number of latent factors (dimensions) to use, which impacts the quality of recommendations.

**Step 5: Generate Recommendations:**
- Multiply the U and Σ matrices to create a reduced user-feature matrix.
- Calculate the dot product of the reduced user-feature matrix and the V^T matrix to generate predicted ratings for each user-item pair.
- Recommend items with the highest predicted ratings for each user.

**Step 6: Evaluate the Recommendation System:**
- Split the dataset into a training set and a test set to evaluate the recommendation system's performance.
- Use evaluation metrics such as Mean Squared Error (MSE) or Root Mean Squared Error (RMSE) to assess the quality of recommendations.

**Step 7: Deploy and Optimize (Optional):**
- If the recommendation system meets your requirements, you can deploy it in a production environment.
- Monitor the system's performance and consider optimizing it by experimenting with different hyperparameters or using more advanced techniques like collaborative filtering with matrix factorization variants (e.g., Alternating Least Squares).

Here's a simplified Python code snippet for building a recommendation system using SVD with scikit-learn (note that real-world recommendation systems often involve more data preprocessing and fine-tuning):

```python
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the MovieLens dataset (example: MovieLens 100k)
data = pd.read_csv('movie_ratings.csv')  # Replace with the actual dataset

# Create the user-item interaction matrix
user_item_matrix = data.pivot_table(index='user_id', columns='movie_id', values='rating')

# Split the data into training and test sets
train_data, test_data = train_test_split(user_item_matrix, test_size=0.2, random_state=42)

# Perform matrix factorization with SVD
n_latent_factors = 10  # Number of latent factors (hyperparameter)
svd = TruncatedSVD(n_components=n_latent_factors)
latent_matrix = svd.fit_transform(train_data)

# Generate recommendations (user-item ratings)
predicted_ratings = pd.DataFrame(svd.inverse_transform(latent_matrix), columns=train_data.columns)

# Evaluate the recommendation system
mse = mean_squared_error(test_data.fillna(0), predicted_ratings.fillna(0))
rmse = np.sqrt(mse)

print("Root Mean Squared Error (RMSE):", rmse)
```

In this example, replace `'movie_ratings.csv'` with the path to your MovieLens dataset or any relevant dataset. This simplified example provides a basic understanding of how to build a recommendation system using matrix factorization with SVD. Real-world recommendation systems may involve more complex algorithms, handling user interactions, and integrating with web applications.

## recommendation system example

**Step 1: Data Collection:**
- Gather historical stock price data, financial reports, news sentiment data, and any other relevant financial data sources. You can obtain this data from financial data providers, APIs, or web scraping.

**Step 2: Data Preprocessing:**
- Clean and preprocess the collected data. This includes handling missing values, removing outliers, and aligning data from different sources.

**Step 3: Feature Engineering:**
- Create meaningful features from the raw data. For stock recommendation, features might include historical price movements, trading volumes, financial ratios, and sentiment scores from news articles.

**Step 4: Build a Predictive Model:**
- Choose a machine learning or deep learning algorithm suitable for stock prediction. Common choices include time series forecasting models (e.g., ARIMA, LSTM), regression models, and ensemble methods.

**Step 5: Train and Test the Model:**
- Split the dataset into training and testing sets to evaluate the model's performance. Use metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), or others relevant to regression tasks to assess model accuracy.

**Step 6: Generate Stock Recommendations:**
- Once you have a trained model, you can use it to predict future stock prices or trends. Based on these predictions, you can generate buy, sell, or hold recommendations for specific stocks.

**Step 7: Risk Management:**
- Incorporate risk management techniques into your recommendation system. Consider factors like portfolio diversification and risk tolerance when providing recommendations to users.

**Step 8: Backtesting:**
- Perform backtesting to evaluate how well the recommendations would have performed in the past. Backtesting helps validate the effectiveness of your model and strategy.

**Step 9: Deployment (Optional):**
- If you plan to deploy the stock recommendation system for real-time use, create an interface or API through which users can access recommendations. Ensure that your system continuously updates its models with fresh data.

**Step 10: Monitoring and Optimization:**
- Continuously monitor the performance of your stock recommendation system in real-world scenarios. Implement mechanisms to retrain models and adapt to changing market conditions.

## Code example

Creating a full-fledged stock recommendation system is a complex task that often requires substantial resources and domain knowledge. However, I can provide you with a simplified example of how you can use Python libraries to retrieve stock data, analyze it, and make a basic recommendation based on a simple strategy. Please note that this example is for educational purposes and should not be used for actual stock trading.

For this example, we'll use Python, the `pandas` library for data manipulation, and `yfinance` to fetch stock data from Yahoo Finance. We'll create a basic moving average crossover strategy, which is a simple approach.

Ensure you have the necessary libraries installed before running this code:

```bash
pip install pandas yfinance
```

Here's the sample code:

```python
import yfinance as yf
import pandas as pd

# Define the stock symbol and date range
stock_symbol = "AAPL"  # Apple Inc.
start_date = "2020-01-01"
end_date = "2021-12-31"

# Fetch historical stock data from Yahoo Finance
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Calculate short-term (e.g., 50-day) and long-term (e.g., 200-day) moving averages
short_window = 50
long_window = 200

stock_data['SMA50'] = stock_data['Close'].rolling(window=short_window).mean()
stock_data['SMA200'] = stock_data['Close'].rolling(window=long_window).mean()

# Create a "Signal" column based on the moving average crossover strategy
stock_data['Signal'] = 0  # Initialize with no signal
stock_data.loc[stock_data['SMA50'] > stock_data['SMA200'], 'Signal'] = 1  # Buy signal

# Display the stock data with signals
print(stock_data.tail(10))  # Print the last 10 rows of data

# Check the latest signal
latest_signal = stock_data['Signal'].iloc[-1]

if latest_signal == 1:
    print(f"Recommendation for {stock_symbol}: Buy")
else:
    print(f"Recommendation for {stock_symbol}: Hold or Sell")
```