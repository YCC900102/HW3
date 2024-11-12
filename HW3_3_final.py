import numpy as np
import plotly.graph_objects as go
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

# Step 4: 使用 LinearSVC 訓練分隔超平面，C 值設為 10
X = np.column_stack((x1, x2, x3))
clf = LinearSVC(random_state=0, max_iter=10000, C=10)
clf.fit(X, Y)
coef = clf.coef_[0]
intercept = clf.intercept_

# 建立 Plotly 三維散佈圖
fig = go.Figure()

# 添加資料點的散佈圖
fig.add_trace(go.Scatter3d(
    x=x1, y=x2, z=x3, mode='markers',
    marker=dict(size=5, color=Y, colorscale='Viridis', opacity=0.7),
    name='Data Points'
))

# 建立網格以繪製分隔超平面
xx, yy = np.meshgrid(np.linspace(min(x1), max(x1), 10),
                     np.linspace(min(x2), max(x2), 10))
offset = 0.1  # 微調偏移量
zz = (-coef[0] * xx - coef[1] * yy - intercept + offset) / coef[2]

# 添加分隔超平面的表面圖
fig.add_trace(go.Surface(
    x=xx, y=yy, z=zz, colorscale='Blues', opacity=0.5,
    name='Separating Hyperplane'
))

# 自訂圖表布局
fig.update_layout(scene=dict(
    xaxis_title='x1', yaxis_title='x2', zaxis_title='x3'
), title='3D Scatter Plot with Y Color and Separating Hyperplane')

# 顯示圖表
fig.show()
