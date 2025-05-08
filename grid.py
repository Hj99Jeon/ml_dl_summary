import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

wine = pd.read_csv('https://bit.ly/wine_csv_data')
# print(wine.head())
# print(wine.info())
# print(wine.describe())
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)
print(sub_input.shape, val_input.shape)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)
# print(dt.score(sub_input, sub_target))
# print(dt.score(val_input, val_target))

splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) # 10-fold 파라미터
scores = cross_validate(dt, train_input, train_target, cv=splitter) # default : 5-fold 교차 검증 / 훈련 셋만 섞음
print(np.mean(scores['test_score']))

params = {'min_impurity_decrease' : np.arange(0.0001, 0.001, 0.0001),
          'max_depth' : range(5, 20, 1),
          'min_samples_split' : range(2, 100, 10)
          }
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1) # -1, 모든 경우의 수를 대입
gs.fit(train_input, train_target)
# dt = gs.best_estimator_
# print(dt.score(train_input, train_target))
print(gs.best_params_)
print(np.max(gs.cv_results_['mean_test_score']))

# best_index = np.argmax(gs.cv_results_['mean_test_score'])
# print(gs.cv_results_['params'][best_index])

