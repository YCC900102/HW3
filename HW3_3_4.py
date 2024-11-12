import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import LinearSVC

# Step 1: 生成以 C1=(0,0) 為中心的橢圓形分布資料集，方差為 10
np.random.seed(0)
num_points = 600
mean = 0
variance = 10
c1_x = np.random.normal(mean, np.sqrt(variance), num_points)
c1_y = np.random.normal(mean, np.sqrt(variance), num_points)

# 計算 C1 資料點到中心的距離，距離小於 6 的標記為 Y=0，其他為 Y=1
distances_c1 = np.sqrt(c1_x**2 + c1_y**2)
Y_c1 = np.where(distances_c1 < 6, 0, 1)

# Step 2: 生成以 C2=(10,10) 為中心的第二組資料集，方差同樣為 10
c2_x = np.random.normal(10, np.sqrt(variance), num_points)
c2_y = np.random.normal(10, np.sqrt(variance), num_points)

# 計算 C2 資料點到中心的距離，距離小於 3 的標記為 Y=0，其他為 Y=1
distances_c2 = np.sqrt((c2_x - 10)**2 + (c2_y - 10)**2)
Y_c2 = np.where(distances_c2 < 3, 0, 1)

# 合併兩組資料集
x1 = np.concatenate((c1_x, c2_x))
x2 = np.concatenate((c1_y, c2_y))
Y = np.concatenate((Y_c1, Y_c2))

# Step 3: 使用高斯函數計算 x3
def gaussian_function(x1, x2):
    return np.exp(-0.1 * (x1**2 + x2**2))

x3 = gaussian_function(x1, x2)

# Step 4: 使用 LinearSVC 訓練分隔超平面，增加 C 值以改善平面位置
X = np.column_stack((x1, x2, x3))
clf = LinearSVC(random_state=0, max_iter=10000, C=10)
clf.fit(X, Y)
coef = clf.coef_[0]
intercept = clf.intercept_

# 繪製三維散佈圖，顏色根據 Y 值分配，並加上分隔超平面
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x1, x2, x3, c=Y, cmap=plt.cm.coolwarm)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
ax.set_title('3D Scatter Plot with Y Color and Separating Hyperplane')

# 建立網格以繪製分隔超平面
xx, yy = np.meshgrid(np.linspace(min(x1), max(x1), 10),
                     np.linspace(min(x2), max(x2), 10))
offset = 0.1  # 偏移量，微調超平面的位置
zz = (-coef[0] * xx - coef[1] * yy - intercept + offset) / coef[2]
ax.plot_surface(xx, yy, zz, color='lightblue', alpha=0.5)

# 顯示圖形
plt.colorbar(scatter)
plt.show()
