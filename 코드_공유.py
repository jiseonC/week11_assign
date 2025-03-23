# 라이브러리 및 데이터 불러오기

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

wine = load_wine()

# feature로 사용할 데이터에서는 'target' 컬럼을 drop합니다.
# target은 'target' 컬럼만을 대상으로 합니다.
# X, y 데이터를 test size는 0.2, random_state 값은 42로 하여 train 데이터와 test 데이터로 분할합니다.

''' 코드 작성 바랍니다 '''
df = pd.DataFrame(data = wine.data, columns = wine.feature_names)
df['target'] = wine.target

target = df['target']
feature = df.drop(columns='target')

X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=42)

####### A 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

param_grid = {
    "criterion": ['gini', 'entropy']
    , "max_depth": [2, 5]
    , "min_samples_split": [2, 10]
    , "min_samples_leaf": [1, 2, 4]
}

# HPO 및 Fitting
clf_grid = DecisionTreeClassifier(random_state= 42)
# core
grid_search = GridSearchCV(clf_grid, param_grid, cv = 5, scoring='accuracy')
# HyperParameter를 찾고, 이걸 가지고 fitting 모두 수행
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best accuracy:" , grid_search.best_score_)

# HPO 만들어진 모델의 정확도 계산
best_model_dt = grid_search.best_estimator_
y_pred_tuned_dt = best_model_dt.predict(X_test)

# print(classification_report(y_test, y_pred_tuned_dt))

#######################################################################################
# Best model에서의 feature importance 값
feature_importance = best_model_dt.feature_importances_ 

# feature importance값을 데이터프레임화
feature_names = wine.feature_names
importance_df = pd.DataFrame({'Feature':feature_names, 'Importance':feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Display
# print(importance_df)

# 시각화
plt.figure(figsize=(10, 6))
plt.bar(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Feature')
plt.ylabel('importances')
plt.title('Feature Importance')
plt.xticks(rotation = 40)
# plt.gca().invert_yaxis() 
plt.show()



####### B 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''

