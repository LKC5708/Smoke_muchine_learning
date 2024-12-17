import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, roc_curve, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler  # StandardScaler 임포트 추가

# 1. 데이터 로드
data = pd.read_csv("./data/1.smoke_detection_iot.csv")

# 2. 데이터 전처리
columns_to_drop = ["Unnamed: 0", "UTC", "CNT"]
data = data.drop(columns=columns_to_drop, errors='ignore')

if data.isnull().sum().sum() > 0:
    data = data.dropna()

y = data["Fire Alarm"]
X = data.drop(columns=["Fire Alarm"], errors='ignore')

# 데이터 정규화 (필요 시)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. K-fold 교차 검증 (K=5)
model = RandomForestClassifier(random_state=42)

# K-fold 교차 검증 (K=5)
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')  # 정확도로 성능 평가

# 교차 검증 결과 출력
print(f"K-fold 교차 검증 정확도: {cv_scores}")
print(f"교차 검증 평균 정확도: {np.mean(cv_scores):.4f}")

# 4. 모델 학습 후 평가 (optional, 전체적인 평가를 위해)
# 5. 모델 평가 (기존의 평가 코드도 사용 가능)
model.fit(X_scaled, y)  # 전체 데이터로 학습

# 예측
y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)[:, 1]

accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)
roc_auc = roc_auc_score(y, y_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# 상세 리포트 출력
print("\nClassification Report:\n", classification_report(y, y_pred))

# 6. 결과 시각화
# Confusion Matrix
conf_matrix = confusion_matrix(y, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["No Fire", "Fire"], yticklabels=["No Fire", "Fire"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y, y_proba)
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")
plt.show()


# 상관관계 계산
correlation_matrix = data.corr()

# 히트맵 그리기
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Correlation Heatmap of Features")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")  # 히트맵 이미지 저장
plt.show()

# 7. 중요 피처 시각화
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar', color='skyblue')
plt.title("Feature Importances")
plt.ylabel("Importance")
plt.savefig("feature_importances.png")
plt.show()