import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: 生成 600 個隨機點，以 (0, 0) 為中心，使用不同的方差來形成橢圓形分布
np.random.seed(0)
num_points = 600
mean = 0
variance_x1 = 15  # x1 的方差較大
variance_x2 = 5   # x2 的方差較小
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

# 繪製三維散佈圖，顏色根據 Y 值分配
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1[Y==0], x2[Y==0], x3[Y==0], c='blue', marker='o', label='Y=0')
ax.scatter(x1[Y==1], x2[Y==1], x3[Y==1], c='red', marker='s', label='Y=1')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
ax.set_title('3D Scatter Plot with Y Color - Elliptical Distribution')
ax.legend()
plt.show()
