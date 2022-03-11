import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split
from statistics import mean
from WhiteBox_WAFS import whitebox


def estimate_s(x, y, vectorizer, text, s_method=False):
    SEED = 2000
    x_train, x_test, y_train, y_test = train_test_split(text, y, test_size=.2, random_state=SEED)
    x_train_features, x_test_features, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=SEED)
    result = mean(whitebox(x_train, x_test, x_train_features, x_test_features, y_train, y_test, x.columns, vectorizer,
                           PGDonly=s_method))
    return result


def wafs(data, target, k, vectorizer, text, s_method, lamda=0.5):
    initial_features = data.columns.tolist()
    best_features = []
    while len(initial_features) > 0 and len(best_features) < k:
        remaining_features = list(set(initial_features)-set(best_features))
        new_g = pd.Series(index=remaining_features, dtype='float64')
        new_s = pd.Series(index=remaining_features, dtype='float64')
        new_gs = pd.Series(index=remaining_features, dtype='float64')
        for new_column in remaining_features:
            model = svm.SVC(kernel='linear')
            # model.fit(data[best_features+[new_column]], target)
            new_g[new_column] = mean(cross_val_score(model, data[best_features+[new_column]], target, cv=5))
            # Revise bellow line when security scoring is finished
            new_s[new_column] = estimate_s(data[best_features+[new_column]], target, vectorizer, text, s_method=s_method)
            new_gs[new_column] = new_g[new_column] + lamda * new_s[new_column]
        lamda = lamda * (new_s.max() ** -1)
        best_features.append(new_gs.idxmax())
        print("curent features ", best_features)
        print("curent len ", len(best_features))
    return best_features







