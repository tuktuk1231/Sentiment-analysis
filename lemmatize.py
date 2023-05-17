import re
import pymorphy2

morph = pymorphy2.MorphAnalyzer()
def lemmatize_text(text):
    words = re.findall("[А-Яа-я]+", text)
    lemmatized_words = []
    preserve_ne = False
    for word in words:
        parsed_word = morph.parse(word)[0]
        if preserve_ne:
            lemmatized_words.append('не')
            preserve_ne = False
        if parsed_word.tag.POS in {'NOUN', 'VERB', 'ADJF', 'ADJS'}:
            lemmatized_words.append(parsed_word.normal_form)
        elif re.match('не', word, re.IGNORECASE):
            preserve_ne = True
    lemmatized_sentence = " ".join(lemmatized_words)
    return lemmatized_sentence

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

sentences = re.split("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)

lemmatized_sentences = []
for sentence in sentences:
    lemmatized_sentence = lemmatize_text(sentence)
    lemmatized_sentences.append(lemmatized_sentence)

with open("text.txt", "w", encoding="utf-8") as f:
    for lemmatized_sentence in lemmatized_sentences:
        f.write(lemmatized_sentence + "\n")
