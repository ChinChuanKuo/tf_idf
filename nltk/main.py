import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

text_file = open("./Natural_Language_Processing_Text.txt")
text = text_file.read()

words = word_tokenize(text)
lemma = WordNetLemmatizer()
words_no_punc = []
for w in words:
    if w.isalpha():
        words_no_punc.append(lemma.lemmatize(w.lower()))

clean_words = []
stopwords = stopwords.words("english")
for w in words_no_punc:
    if w not in stopwords:
        clean_words.append(w)

fdist = FreqDist(clean_words)
print(fdist.most_common(10))
fdist.plot(10)