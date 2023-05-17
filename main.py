import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("color_data.csv")

color_dict = {}
for color in ['красный', 'зеленый', 'синий', 'желтый', 'черный', 'белый']:
    color_dict[color] = list(data[data[color] == 1]['word'])

train_data = []
train_labels = []
for color, words in color_dict.items():
    train_data.extend(words)
    train_labels.extend([color] * len(words))

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(train_data)
y = np.array(train_labels)
model = SVC(kernel='linear')

model.fit(X, y)

scores = cross_val_score(model, X, y, cv=5)
mean_accuracy = np.mean(scores)

print("Средняя точность модели: {:.2f}".format(mean_accuracy))

with open("text.txt", "r", encoding='UTF8') as f:
    lines = f.readlines()

for line in lines:
    sentence = line.strip()
    if not sentence:
        continue
    X_test = vectorizer.transform([sentence])
    predicted_color = model.predict(X_test)[0]
    print("'{}': {}".format(sentence, predicted_color))

train_predictions = model.predict(X)
train_accuracy = accuracy_score(y, train_predictions)
train_precision = precision_score(y, train_predictions, average='weighted')
train_recall = recall_score(y, train_predictions, average='weighted')
train_f1 = f1_score(y, train_predictions, average='weighted')

print("\nМетрики на обучающих данных:")
print("Accuracy: {:.2f}".format(train_accuracy))
print("Precision: {:.2f}".format(train_precision))
print("Recall: {:.2f}".format(train_recall))
print("F1-Score: {:.2f}".format(train_f1))

test_predictions = model.predict(X)
confusion_mat = confusion_matrix(y, test_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, cmap='Blues', fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()