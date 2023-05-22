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

    print("Средняя точность модели {}: {:.2f}".format(model.__class__.__name__, mean_accuracy))

    return model, vectorizer, X, y

def evaluate_model(model, vectorizer, X, y):
    train_predictions = model.predict(X)
    train_accuracy = accuracy_score(y, train_predictions)
    train_precision = precision_score(y, train_predictions, average='weighted')
    train_recall = recall_score(y, train_predictions, average='weighted')
    train_f1 = f1_score(y, train_predictions, average='weighted')

    print("\nМетрики на обучающих данных:")
    print("Точность (Accuracy): {:.2f}".format(train_accuracy))
    print("Точность (Precision): {:.2f}".format(train_precision))
    print("Полнота (Recall): {:.2f}".format(train_recall))
    print("F1-мера (F1-Score): {:.2f}".format(train_f1))
    print("\n")

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

    svm_model = SVC(kernel='linear', probability=True)
    svm_model, svm_vectorizer, svm_X, svm_y = train_model(svm_model, train_data, train_labels)
    svm_confusion_mat = evaluate_model(svm_model, svm_vectorizer, svm_X, svm_y)

    plt.figure(figsize=(8, 6))
    sns.heatmap(svm_confusion_mat, annot=True, cmap='Blues', fmt='d', xticklabels=svm_model.classes_,yticklabels=svm_model.classes_)
    plt.xlabel('Предсказанное значение')
    plt.ylabel('Истинное значение')
    plt.title('Матрица ошибок - SVM')
    plt.show()

    nb_model = MultinomialNB()
    nb_model, nb_vectorizer, nb_X, nb_y = train_model(nb_model, train_data, train_labels)
    nb_confusion_mat = evaluate_model(nb_model, nb_vectorizer, nb_X, nb_y)

    plt.figure(figsize=(8, 6))
    sns.heatmap(nb_confusion_mat, annot=True, cmap='Blues', fmt='d', xticklabels=nb_model.classes_,yticklabels=nb_model.classes_)
    plt.xlabel('Предсказанное значение')
    plt.ylabel('Истинное значение')
    plt.title('Матрица ошибок - Naive Bayes')
    plt.show()

    with open('input.txt', 'r', encoding='utf-8') as input_file:
        input_lines = input_file.readlines()

    with open('text.txt', 'r', encoding='utf-8') as text_file:
        text_lines = text_file.readlines()

    for input_line, text_line in zip(input_lines, text_lines):
        input_sentence = input_line.strip()
        text_sentence = text_line.strip()
        if not input_sentence or not text_sentence:
            continue

        X_test_svm = svm_vectorizer.transform([text_sentence])
        svm_probs = svm_model.predict_proba(X_test_svm)
        max_prob_svm = np.max(svm_probs)
        predicted_color_svm = svm_model.predict(X_test_svm)[0]

        X_test_nb = nb_vectorizer.transform([text_sentence])
        nb_probs = nb_model.predict_proba(X_test_nb)
        max_prob_nb = np.max(nb_probs)
        predicted_color_nb = nb_model.predict(X_test_nb)[0]

        print("Текст: {}".format(input_sentence))

        if max_prob_svm > max_prob_nb:
            print("Модель с более высокой вероятностью: SVM")
            print("Предсказанный цвет (SVM): {}".format(predicted_color_svm))
            print("Вероятность (SVM): {:.2f}".format(max_prob_svm))
        else:
            print("Модель с более высокой вероятностью: Naive Bayes")
            print("Предсказанный цвет (Naive Bayes): {}".format(predicted_color_nb))
            print("Вероятность (Naive Bayes): {:.2f}".format(max_prob_nb))

        print()


if __name__ == "__main__":
    main()
