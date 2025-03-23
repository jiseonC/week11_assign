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



####### B 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, classification_report

# 레이블링 인코딩
label_encoder = LabelEncoder() 
y_train_encoded = label_encoder.fit_transform(y_train) 
y_test_encoded = label_encoder.transform(y_test)

xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train_encoded)

# 예측 및 레이블 디코딩
rf_y_pred_encoded = xgb_model.predict(X_test)
y_pred_xgb = label_encoder.inverse_transform(rf_y_pred_encoded)

params = {
    "max_depth" : [3, 5, 7, 9, 15],
    "learning_rate" : [0.1, 0.01, 0.001],
    "n_estimators": [50, 100, 200, 300]
}
# 하이퍼파라미터 최적화 
grid_search = GridSearchCV(estimator=xgb_model, param_grid=params, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train_encoded)

print("Best parameters:", grid_search.best_params_)
print("Best accuracy:" , grid_search.best_score_)

# 최적의 하이퍼파라미터의 학습
best_model_xgb = grid_search.best_estimator_

#테스트 데이터에 대한 예측
y_pred_encoded_xgb = best_model_xgb.predict(X_test)
y_pred_xgb = label_encoder.inverse_transform(y_pred_encoded_xgb)

# print(classification_report(y_test, y_pred_xgb))

#######################################################################################
# Best model에서의 feature importance 값
feature_importance = best_model_xgb.feature_importances_ 

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
