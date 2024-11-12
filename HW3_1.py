import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 隨機生成 300 個範圍在 [0, 1000] 的整數 X
np.random.seed(42)
X = np.random.randint(0, 1001, size=(300, 1))

# 根據條件 500 < X < 800 創建標籤 Y
Y = (X > 500) & (X < 800)
Y = Y.astype(int)

# 將數據集分為 80% 訓練集和 20% 測試集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 初始化 Logistic Regression 和 SVM 模型
log_reg = LogisticRegression()
svm = SVC(probability=True)

# 訓練模型
log_reg.fit(X_train, Y_train)
svm.fit(X_train, Y_train)

# 在測試集上預測
log_reg_pred = log_reg.predict(X_test)
svm_pred = svm.predict(X_test)

# 計算準確率
log_reg_accuracy = accuracy_score(Y_test, log_reg_pred)
svm_accuracy = accuracy_score(Y_test, svm_pred)

# 用於平滑的決策邊界，生成範圍在 [0, 1000] 的連續數值
X_range = np.linspace(0, 1000, 1000).reshape(-1, 1)

# 計算 Logistic Regression 和 SVM 的預測概率
log_reg_prob = log_reg.predict_proba(X_range)[:, 1]
svm_prob = svm.predict_proba(X_range)[:, 1]

# 設置子圖
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左圖：Logistic Regression
axes[0].scatter(X_train, Y_train, color='gray', alpha=0.5, label='True Labels', zorder=5)
axes[0].scatter(X_test, log_reg_pred, color='orange', marker='x', label='Logistic Regression Predictions', zorder=10)
axes[0].plot(X_range, log_reg_prob, color='orange', linestyle='--', label='Decision Boundary')
axes[0].set_title('Logistic Regression')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Probability')
axes[0].legend()

# 右圖：SVM
axes[1].scatter(X_train, Y_train, color='gray', alpha=0.5, label='True Labels', zorder=5)
axes[1].scatter(X_test, svm_pred, color='purple', marker='o', label='SVM Predictions', zorder=10)
axes[1].plot(X_range, svm_prob, color='purple', linestyle='--', label='Decision Boundary')
axes[1].set_title('SVM')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Probability')
axes[1].legend()

# 顯示圖形
plt.tight_layout()
plt.show()

# 打印準確率
print(f"Logistic Regression Accuracy: {log_reg_accuracy:.2f}")
print(f"SVM Accuracy: {svm_accuracy:.2f}")
