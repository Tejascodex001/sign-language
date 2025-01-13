import pickle
import numpy as np
import shap
import lime
import lime.lime_tabular
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
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

# LIME Explanation
explainer = lime.lime_tabular.LimeTabularExplainer(
    x_train, feature_names=[f'feature_{i}' for i in range(x_train.shape[1])], class_names=np.unique(y_train), discretize_continuous=True
)

# Explain a single prediction
i = 0  # Index of the instance to explain
exp = explainer.explain_instance(x_test[i], best_model.predict_proba, num_features=10)

# Save LIME explanation as an image
fig = exp.as_pyplot_figure()
fig.savefig('lime_explanation.png')
plt.close(fig)

# SHAP Explanation
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(x_test)

# Plot SHAP summary plot
shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
plt.savefig('shap_summary_plot.png')
plt.close()

# Plot SHAP dependence plot for a specific feature (use a valid feature index)
feature_index = 0  # Change this to a valid feature index if needed
if feature_index < x_test.shape[1]:
    shap.dependence_plot(feature_index, shap_values[0], x_test, show=False)
    plt.savefig(f'shap_dependence_plot_feature_{feature_index}.png')
    plt.close()
else:
    print(f"Feature index {feature_index} is out of bounds for the dataset with {x_test.shape[1]} features.")

# Generate and save confusion matrix
conf_matrix = confusion_matrix(y_test, best_model.predict(x_test))
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

# Generate and save classification report
class_report = classification_report(y_test, best_model.predict(x_test), output_dict=True)
plt.figure(figsize=(10, 7))
sns.heatmap(pd.DataFrame(class_report).iloc[:-1, :].T, annot=True)
plt.title('Classification Report')
plt.savefig('classification_report.png')
plt.close()

# Save accuracy score as an image
score = accuracy_score(y_test, best_model.predict(x_test))
plt.figure(figsize=(5, 5))
plt.text(0.5, 0.5, f'Accuracy: {score * 100:.2f}%', horizontalalignment='center', verticalalignment='center', fontsize=20)
plt.axis('off')
plt.savefig('accuracy_score.png')
plt.close()
