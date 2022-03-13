from sklearn.feature_extraction.text import TfidfVectorizer
from WAFS import wafs
import pandas as pd
from sklearn import svm
from sklearn import metrics
from Preprocessing import preprocess


df = pd.read_csv('messages.csv')
y = df.label
x_text = df.message
x, xt = preprocess(x_text, x_text)
print("Finished preprocessing")
tvec1 = TfidfVectorizer(max_features=500, ngram_range=(1, 3))
tvec1.fit(x)
x_tfidf = tvec1.transform(x).toarray()
x = pd.DataFrame(x_tfidf, columns=tvec1.get_feature_names())
print("Finished TFIDF")
# s_method=False -> with good word attack
# s_method=True -> with PGD only
PGD_only = True
print("PGD only:", PGD_only)
selected_features = wafs(x, y, 38, tvec1, x_text, PGD_only)
x = x[selected_features]
print("Finished WAFS")
clf = svm.SVC(kernel='linear')
clf.fit(x, y)
print("Accuracy:", metrics.accuracy_score(y, clf.predict(x)))
print(selected_features)