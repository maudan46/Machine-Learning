# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load data from the CSV file
patient_data = pd.read_csv("drug200.csv", delimiter=",")
patient_data.head()

# Select features and target variable
features = patient_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

# Encode categorical variables
label_encoder_sex = preprocessing.LabelEncoder()
label_encoder_sex.fit(['F', 'M'])
features[:, 1] = label_encoder_sex.transform(features[:, 1])

label_encoder_bp = preprocessing.LabelEncoder()
label_encoder_bp.fit(['LOW', 'NORMAL', 'HIGH'])
features[:, 2] = label_encoder_bp.transform(features[:, 2])

label_encoder_cholesterol = preprocessing.LabelEncoder()
label_encoder_cholesterol.fit(['NORMAL', 'HIGH'])
features[:, 3] = label_encoder_cholesterol.transform(features[:, 3])

# Select target variable
target = patient_data["Drug"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=3)

# Create and train the decision tree classifier
drug_tree_classifier = DecisionTreeClassifier(criterion="entropy", max_depth=4)
drug_tree_classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = drug_tree_classifier.predict(X_test)

# Evaluate the model accuracy
accuracy = metrics.accuracy_score(y_test, predictions)
print("Decision Tree's Accuracy: {:.2f}".format(accuracy))

# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(drug_tree_classifier, filled=True, feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'],
          class_names=np.unique(target), rounded=True)
plt.title("Decision Tree Visualization")
plt.show()
