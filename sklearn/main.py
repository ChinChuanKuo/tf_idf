from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

text_file = open("./Natural_Language_Processing_Text.txt")
text = sent_tokenize(text_file.read())

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(text)
print(vectorizer.get_feature_names())