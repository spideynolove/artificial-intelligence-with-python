# Probabilistic Reasoning

- Handling time-series data with Pandas
- Slicing time-series data
- Operating on time-series data
- Extracting statistics from time-series data
- Generating data using Hidden Markov Models
- Identifying alphabet sequences with Conditional Random Fields
- Stock market analysis

## Overview

### Sequential learning models

Sequence learning models, also known as sequence models or sequential models, are a class of machine learning models designed to handle sequences of data. Sequences are ordered collections of elements, and these models are particularly well-suited for tasks where the order of elements in the input data is crucial. Sequence learning models are commonly used in various fields, including natural language processing (NLP), speech recognition, time series analysis, and bioinformatics.

1. **Recurrent Neural Networks (RNNs):**
   - **Description:** RNNs are a type of neural network designed for sequential data. They have connections that form cycles, allowing them to maintain a hidden state that captures information about previous elements in the sequence. This hidden state is updated at each time step, making RNNs suitable for tasks involving dependencies over time.
   - **Applications:** Natural language processing, time series prediction, speech recognition.

2. **Long Short-Term Memory Networks (LSTMs) and Gated Recurrent Units (GRUs):**
   - **Description:** These are specialized types of RNNs that address the vanishing gradient problem, which can hinder the training of traditional RNNs. LSTMs and GRUs use mechanisms to selectively update and forget information in the hidden state, allowing them to capture long-term dependencies more effectively.
   - **Applications:** Similar to RNNs, with improved performance on long sequences.

3. **Transformer Models:**
   - **Description:** Transformers are a type of sequence-to-sequence model that does not rely on recurrent connections. Instead, they use self-attention mechanisms to weigh the importance of different elements in the sequence. Transformers have been highly successful in NLP tasks and have also been adapted for other sequential data types.
   - **Applications:** Natural language processing, image captioning, machine translation.

4. **Hidden Markov Models (HMMs):**
   - **Description:** HMMs are probabilistic models that represent a system evolving over time in a sequence of hidden states. Observations are generated from these hidden states with associated probabilities. HMMs are commonly used in applications where the underlying system is assumed to be a Markov process.
   - **Applications:** Speech recognition, part-of-speech tagging, bioinformatics.

5. **Conditional Random Fields (CRFs):**
   - **Description:** CRFs are a type of discriminative probabilistic model that can be used for labeling sequential data. Unlike HMMs, which model joint probabilities, CRFs model conditional probabilities given the observed data. They are often used for tasks like sequence labeling.
   - **Applications:** Named entity recognition, part-of-speech tagging.

6. **Bidirectional RNNs:**
   - **Description:** Bidirectional RNNs process sequences in both forward and backward directions. This allows them to consider information from both past and future elements in the sequence at each time step.
   - **Applications:** Natural language processing, sentiment analysis.

### sequence learning models example using pytorch

Creating sequence learning models using PyTorch often involves using recurrent neural networks (RNNs) or transformer-based architectures. Below is a simple example of a sequence learning model using PyTorch for sequence classification. Using an LSTM (Long Short-Term Memory) network.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define a simple LSTM-based sequence model
class SequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SequenceModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out

# Generate synthetic data
sequence_length = 10
input_size = 5
output_size = 2
hidden_size = 10
batch_size = 32

# Create synthetic input sequences and labels
data = torch.randn(batch_size, sequence_length, input_size)
labels = torch.randint(0, output_size, (batch_size,))

# Create DataLoader
dataset = TensorDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate the model, loss function, and optimizer
model = SequenceModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Inference
model.eval()
with torch.no_grad():
    new_data = torch.randn(5, sequence_length, input_size)
    predictions = model(new_data)
    predicted_labels = torch.argmax(predictions, dim=1)
    print("Predicted Labels:", predicted_labels.numpy())
```

### How to handle time-series data in Pandas

Handling time-series data in Pandas involves leveraging the built-in capabilities of the Pandas library, which provides specialized data structures and functions for working with time-related data. 

**1. Import the Necessary Libraries:**

```python
import pandas as pd
```

**2. Create or Load Time-Series Data:**

You can create a time-series DataFrame or load time-series data from a file. Pandas provides the `pd.to_datetime()` function to convert strings to datetime objects.

```python
# Example: Creating a time-series DataFrame
date_rng = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
df = pd.DataFrame(date_rng, columns=['date'])
```

**3. Set the Datetime Column as the Index:**

Set the datetime column as the index of the DataFrame using the `set_index()` method.

```python
df.set_index('date', inplace=True)
```

**4. Accessing and Slicing Time-Series Data:**

You can use the datetime index to access and slice data based on time.

```python
# Accessing data for a specific date
print(df.loc['2023-01-03'])

# Slicing data for a date range
print(df['2023-01-03':'2023-01-07'])
```

**5. Resampling Time-Series Data:**

Resampling allows you to change the frequency of the time series. Use the `resample()` method to aggregate or downsample data.

```python
# Resample to weekly frequency and calculate the mean
weekly_data = df.resample('W').mean()
```

**6. Shifting and Lagging:**

Shift the values of a time series using the `shift()` method to create lag features.

```python
# Create a lagged column
df['lagged_value'] = df['value'].shift(1)
```

**7. Rolling Windows:**

Calculate rolling statistics, such as moving averages, using the `rolling()` method.

```python
# Calculate a 3-day moving average
df['rolling_average'] = df['value'].rolling(window=3).mean()
```

**8. Handling Time Zones:**

Pandas supports working with time zones. You can set and convert time zones using the `tz_localize()` and `tz_convert()` methods.

```python
# Set time zone
df = df.tz_localize('UTC')

# Convert time zone
df = df.tz_convert('US/Eastern')
```

### how to extract various stats from time-series data on a rolling basis

To extract various statistics from time-series data on a rolling basis, you can use the rolling window functionality provided by Pandas along with aggregation functions. The following steps demonstrate how to compute rolling statistics using a rolling window:

```python
import pandas as pd
import numpy as np

# Create or load time-series data
date_rng = pd.date_range(start='2023-01-01', end='2023-01-15', freq='D')
data = np.random.randn(len(date_rng))
df = pd.DataFrame(data, index=date_rng, columns=['value'])

# Calculate rolling mean, standard deviation, and sum
window_size = 3
df_rolling = df.rolling(window=window_size)

# Rolling mean
df['rolling_mean'] = df_rolling.mean()

# Rolling standard deviation
df['rolling_std'] = df_rolling.std()

# Rolling sum
df['rolling_sum'] = df_rolling.sum()

print(df)
```

In this example, we've created a DataFrame with a random time-series, and we're calculating rolling mean, standard deviation, and sum using a window size of 3.

Here are the key points:

1. **Create or Load Time-Series Data:** Initialize your time-series data, either by creating a DataFrame or loading it from an external source.

2. **Specify the Rolling Window:** Determine the size of the rolling window by setting the `window` parameter in the `rolling()` method.

3. **Apply Aggregation Functions:** Use various aggregation functions provided by Pandas (e.g., `mean()`, `std()`, `sum()`) to calculate the desired statistics within the rolling window.

4. **Store Results:** Assign the calculated rolling statistics to new columns in the DataFrame.

### Hidden Markov Models

Hidden Markov Models (HMMs) are widely used for modeling time-series data with underlying hidden states. 

```bash
pip install hmmlearn
```

Now, you can use the following code to generate synthetic data using an HMM:

```python
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Define the HMM parameters
n_components = 2  # Number of hidden states
n_observation_features = 1  # Number of features in the observations
model = hmm.GaussianHMM(n_components=n_components, covariance_type="full")

# Set initial parameters
model.startprob_ = np.array([0.5, 0.5])  # Initial state probabilities
model.transmat_ = np.array([[0.7, 0.3], [0.3, 0.7]])  # State transition matrix

# Set means and covariances for each state
model.means_ = np.array([[0.0], [5.0]])  # Means of the Gaussian distributions
model.covars_ = np.array([[[1.0]], [[1.0]]])  # Covariance matrices

# Generate synthetic data
num_samples = 100
hidden_states, observations = model.sample(num_samples)

# Plot the generated data
plt.figure(figsize=(10, 6))
plt.plot(observations, label="Observations", marker="o")
plt.plot(hidden_states, label="Hidden States", linestyle="--", marker="x", markersize=8)
plt.title("Generated Data from Hidden Markov Model")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.show()
```

Note: The `hmmlearn` library is used here for simplicity, but for more complex scenarios and applications, consider using other libraries like `pyro` for PyTorch-based HMMs or `pomegranate` for more advanced HMM functionalities.

### Conditional Random Fields

Conditional Random Fields (CRFs) are a type of probabilistic graphical model used for modeling structured data, particularly in the context of sequence labeling and segmentation tasks. They are discriminative models that can be used to predict the labels of a sequence of observations given the input features. 

### Key Concepts:

1. **Sequential Data Modeling:**
   - CRFs are designed for modeling sequential or structured data, such as natural language sentences, speech signals, or biological sequences.

2. **Graphical Model:**
   - CRFs are a type of graphical model, where nodes represent random variables, and edges represent dependencies between them.

3. **Features and Labels:**
   - Each observation in the sequence is associated with a set of features, and the goal is to predict the corresponding label. Features can capture information about the current observation and its context.

4. **Conditional Probability:**
   - CRFs model the conditional probability of label sequences given the input features. The conditional nature of CRFs distinguishes them from generative models like Hidden Markov Models (HMMs).

5. **Discriminative Model:**
   - CRFs are discriminative models, meaning they directly model the conditional distribution of labels given the input, rather than modeling the joint distribution of observations and labels.

6. **Inference:**
   - Inference in CRFs involves finding the most likely sequence of labels given the observed features. This is typically done using dynamic programming algorithms, such as the Viterbi algorithm.

### Applications:

1. **Sequence Labeling:**
   - Named Entity Recognition (NER), part-of-speech tagging, and chunking are common applications where CRFs excel. They can capture dependencies between adjacent labels in a sequence.

2. **Segmentation:**
   - CRFs can be used for image segmentation tasks, where the goal is to label each pixel in an image based on its features and the context of neighboring pixels.

3. **Speech Recognition:**
   - CRFs can be applied to phoneme or word segmentation in speech signals.

### Implementation:

- The scikit-learn library in Python provides an implementation of CRFs for sequence labeling tasks. You can use the `sklearn_crfsuite` package, which is an interface to the CRFsuite library.

  ```python
  from sklearn_crfsuite import CRF
  ```

- Training a CRF involves providing labeled sequences and their corresponding features. The model learns the parameters that maximize the conditional likelihood of the labels given the features.

### Example Code:

Here's a simplified example using scikit-learn CRFsuite for named entity recognition (NER):

```python
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score
from sklearn.model_selection import train_test_split

# Assume 'X' contains input features, and 'y' contains corresponding labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the CRF model
crf = CRF()
crf.fit(X_train, y_train)

# Make predictions
y_pred = crf.predict(X_test)

# Evaluate the model
f1_score = flat_f1_score(y_test, y_pred, average='weighted')
print(f"F1 Score: {f1_score}")
```

## Understanding sequential data

