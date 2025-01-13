import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

# Filter out classes with fewer than 2 samples
label_counts = Counter(labels)
filtered_data = []
filtered_labels = []
for i, label in enumerate(labels):
    if label_counts[label] > 1:
        filtered_data.append(data[i])
        filtered_labels.append(label)

filtered_data = np.asarray(filtered_data)
filtered_labels = np.asarray(filtered_labels)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    filtered_data, filtered_labels, test_size=0.2, shuffle=True, stratify=filtered_labels
)

# Load the trained model
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
    best_model = model_dict['model']

# Evaluate the best model
y_predict = best_model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(f"{score * 100:.2f}% of samples were classified correctly!")

# Generate and save confusion matrix
conf_matrix = confusion_matrix(y_test, y_predict)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

# Generate and save classification report
class_report = classification_report(y_test, y_predict, output_dict=True)
plt.figure(figsize=(10, 7))
sns.heatmap(pd.DataFrame(class_report).iloc[:-1, :].T, annot=True)
plt.title('Classification Report')
plt.savefig('classification_report.png')
plt.close()

# Save accuracy score as an image
plt.figure(figsize=(5, 5))
plt.text(0.5, 0.5, f'Accuracy: {score * 100:.2f}%', horizontalalignment='center', verticalalignment='center', fontsize=20)
plt.axis('off')
plt.savefig('accuracy_score.png')
plt.close()
