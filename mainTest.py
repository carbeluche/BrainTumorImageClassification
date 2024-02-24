import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from mainTrain import x_test, y_test

model = load_model('BrainTumor10EpochsCategorical.h5')

image = cv2.imread('/Users/carbeluche/Downloads/BrainTumorImageClassification/pred/pred0.jpg')

img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img)

# print(img)

input_img = np.expand_dims(img, axis=0)
result = model.predict(input_img)
predicted_class = np.round(result)
print(predicted_class)

# Evaluate the model on the test set
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Create confusion matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Tumor', 'Tumor'],
            yticklabels=['No Tumor', 'Tumor'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('CONFUSION MATRIX', fontsize=24)
plt.show()


correctly_identified_tumor_images = conf_matrix[1, 1]
misclassified_tumor_images = conf_matrix[0, 1] + conf_matrix[1, 0]
print(f"The model correctly identified {correctly_identified_tumor_images} MRI images as brain tumors while misclassifying {misclassified_tumor_images} MRI images.")
print(f"The overall accuracy of the model on the test set is calculated at {accuracy * 100:.2f}%, highlighting its reliable classification capabilities.")
