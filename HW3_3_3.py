import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import LinearSVC

# Step 1: 生成橢圓形分布的資料集，中心在 (0, 0)，x1 方差為 15，x2 方差為 5
np.random.seed(0)
num_points = 600
mean = 0
variance_x1 = 15  # x1 的方差
variance_x2 = 5   # x2 的方差
x1 = np.random.normal(mean, np.sqrt(variance_x1), num_points)
x2 = np.random.normal(mean, np.sqrt(variance_x2), num_points)

# 計算每個點到原點的 "橢圓距離"
distances = np.sqrt((x1 / np.sqrt(variance_x1))**2 + (x2 / np.sqrt(variance_x2))**2)

# 根據橢圓距離進行分類：距離小於 2 的點標記為 Y=0，其他點標記為 Y=1
Y = np.where(distances < 2, 0, 1)

# Step 2: 使用高斯函數計算 x3
def gaussian_function(x1, x2):
    return np.exp(-0.1 * (x1**2 + x2**2))

x3 = gaussian_function(x1, x2)

# Step 3: 使用 LinearSVC 訓練分隔超平面，並調整 C 值
X = np.column_stack((x1, x2, x3))
clf = LinearSVC(random_state=0, max_iter=10000, C=10)  # 增加 C 值
clf.fit(X, Y)
coef = clf.coef_[0]
intercept = clf.intercept_

# 繪製三維散佈圖，顏色根據 Y 值分配，並加上分隔超平面
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1[Y==0], x2[Y==0], x3[Y==0], c='blue', marker='o', label='Y=0')
ax.scatter(x1[Y==1], x2[Y==1], x3[Y==1], c='red', marker='s', label='Y=1')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
ax.set_title('3D Scatter Plot with Y Color and Adjusted Separating Hyperplane')
ax.legend()

# 建立網格以繪製分隔超平面，並加上微調偏移量
xx, yy = np.meshgrid(np.linspace(min(x1), max(x1), 10),
                     np.linspace(min(x2), max(x2), 10))
offset = 0.1  # 可調整偏移量
zz = (-coef[0] * xx - coef[1] * yy - intercept + offset) / coef[2]
ax.plot_surface(xx, yy, zz, color='gray', alpha=0.5)

plt.show()
