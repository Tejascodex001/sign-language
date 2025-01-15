import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from collections import Counter

# Load the dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))
data_raw = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Standardize feature lengths
expected_length = 84
data = []
for i, sample in enumerate(data_raw):
    if len(sample) < expected_length:
        sample = sample + [0] * (expected_length - len(sample))
    elif len(sample) > expected_length:
        sample = sample[:expected_length]
    data.append(sample)

# Convert to NumPy array
data = np.asarray(data)

# Ensure labels match the filtered data
labels = labels[:len(data)]

# Check class distribution
class_counts = Counter(labels)
print("Class distribution before filtering:", class_counts)

# Filter out classes with fewer than 2 samples
filtered_data = []
filtered_labels = []
for i, label in enumerate(labels):
    if class_counts[label] > 1:
        filtered_data.append(data[i])
        filtered_labels.append(label)

filtered_data = np.asarray(filtered_data)
filtered_labels = np.asarray(filtered_labels)

# Check class distribution after filtering
class_counts = Counter(filtered_labels)
print("Class distribution after filtering:", class_counts)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    filtered_data, filtered_labels, test_size=0.2, shuffle=True, stratify=filtered_labels
)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'random_state': [42]
}

# Initialize the RandomForestClassifier
model = RandomForestClassifier()

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Evaluate the best model
y_predict = best_model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(f"{score * 100:.2f}% of samples were classified correctly!")

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': best_model}, f)
