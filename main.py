import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(model, train_data, train_labels):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(train_data)
    y = np.array(train_labels)

    model.fit(X, y)

    scores = cross_val_score(model, X, y, cv=5)
    mean_accuracy = np.mean(scores)

    print("Средняя точность модели: {:.2f}".format(mean_accuracy))

    return model, vectorizer, X, y

def evaluate_model(model, vectorizer, X, y):
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
    return confusion_mat

def main():
    data = pd.read_csv("color_data.csv")

    color_dict = {}
    for color in ['красный', 'зеленый', 'синий', 'желтый', 'черный', 'белый']:
        color_dict[color] = list(data[data[color] == 1]['word'])

    train_data = []
    train_labels = []
    for color, words in color_dict.items():
        train_data.extend(words)
        train_labels.extend([color] * len(words))

    svm_model = SVC(kernel='linear')
    svm_model, svm_vectorizer, svm_X, svm_y = train_model(svm_model, train_data, train_labels)
    svm_confusion_mat = evaluate_model(svm_model, svm_vectorizer, svm_X, svm_y)

    plt.figure(figsize=(8, 6))
    sns.heatmap(svm_confusion_mat, annot=True, cmap='Blues', fmt='d', xticklabels=svm_model.classes_, yticklabels=svm_model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - SVM')
    plt.show()

    nb_model = MultinomialNB()
    nb_model, nb_vectorizer, nb_X, nb_y = train_model(nb_model, train_data, train_labels)
    nb_confusion_mat = evaluate_model(nb_model, nb_vectorizer, nb_X, nb_y)

    plt.figure(figsize=(8, 6))
    sns.heatmap(nb_confusion_mat, annot=True, cmap='Blues', fmt='d', xticklabels=nb_model.classes_, yticklabels=nb_model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Naive Bayes')
    plt.show()

    with open('text.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        sentence = line.strip()
        if not sentence:
            continue

        X_test_svm = svm_vectorizer.transform([sentence])
        predicted_color_svm = svm_model.predict(X_test_svm)[0]
        print("SVM - Predicted color for text '{}': {}".format(sentence, predicted_color_svm))

        X_test_nb = nb_vectorizer.transform([sentence])
        predicted_color_nb = nb_model.predict(X_test_nb)[0]
        print("Naive Bayes - Predicted color for text '{}': {}".format(sentence, predicted_color_nb))

if __name__ == "__main__":
    main()
