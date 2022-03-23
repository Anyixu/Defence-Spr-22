from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from WAFS import wafs
import pandas as pd
from sklearn import svm
from sklearn import metrics
from Preprocessing import preprocess
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector

df = pd.read_csv('messages.csv')
y = df.label
x_text = df.astype({'message':'str'}).message
print(x_text)
x, z, y, z2 = train_test_split(x_text, y, train_size=500, random_state=99)
tvec1 = TfidfVectorizer()
tvec1.fit(x)
print("Feaute length: ", len(tvec1.get_feature_names_out()))
x_tfidf = tvec1.transform(x).toarray()
x_feature = pd.DataFrame(x_tfidf, columns=tvec1.get_feature_names_out())
print("Finished TFIDF")
# s_method=False -> with good word attack
# s_method=True -> with PGD only
clf = svm.SVC(kernel='linear')
selector = SelectFromModel(max_features=500, estimator=LogisticRegression()).fit(x_feature, y)
x_feature = x_feature[selector.get_feature_names_out()]
print(x_feature)
PGD_only = False
print("PGD only:", PGD_only)
thread_num = 1
selected_features = wafs(x_feature, y, 38, tvec1, x, PGD_only, thread_num)
x = x[selected_features]
print("Finished WAFS")
clf = svm.SVC(kernel='linear')
clf.fit(x, y)
print("Accuracy:", metrics.accuracy_score(y, clf.predict(x)))
print(selected_features)