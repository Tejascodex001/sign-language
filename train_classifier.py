import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))
data_raw = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Standardize feature lengths
expected_length = 84
data = []
for i, sample in enumerate(data_raw):
    if len(sample) < expected_length:
        # Pad with zeros if too short
        sample = sample + [0] * (expected_length - len(sample))
    elif len(sample) > expected_length:
        # Truncate if too long
        sample = sample[:expected_length]
    data.append(sample)

# Convert to NumPy array
data = np.asarray(data)

# Ensure labels match the filtered data
labels = labels[:len(data)]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate the model
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(f"{score * 100:.2f}% of samples were classified correctly!")

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
