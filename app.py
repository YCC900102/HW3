import numpy as np
import plotly.graph_objects as go
from sklearn.svm import LinearSVC
import streamlit as st

# Streamlit 標題
st.title("SVM 3D Scatter Plot with Separating Hyperplane")

# Step 1: Generate 600 random points centered at C1=(0,0) with variance 10
np.random.seed(0)
num_points = 600
mean = 0
variance = 10
c1_x = np.random.normal(mean, np.sqrt(variance), num_points)
c1_y = np.random.normal(mean, np.sqrt(variance), num_points)

# Calculate distances from C1
distances_c1 = np.sqrt(c1_x**2 + c1_y**2)

# Assign labels Y=0 for points within distance 6 from C1, Y=1 for the rest
Y_c1 = np.where(distances_c1 < 6, 0, 1)

# Step 2: Generate another dataset centered at C2=(10,10) with variance 10
c2_x = np.random.normal(10, np.sqrt(variance), num_points)
c2_y = np.random.normal(10, np.sqrt(variance), num_points)

# Calculate distances from C2
distances_c2 = np.sqrt((c2_x - 10)**2 + (c2_y - 10)**2)

# Assign labels Y=0 for points within distance 10 from C2, Y=1 for the rest
Y_c2 = np.where(distances_c2 < 3, 0, 1)

# Combine the two datasets
x1 = np.concatenate((c1_x, c2_x))
x2 = np.concatenate((c1_y, c2_y))
Y = np.concatenate((Y_c1, Y_c2))

# Step 3: Calculate x3 using a Gaussian function
def gaussian_function(x1, x2):
    return np.exp(-0.1 * (x1**2 + x2**2))

x3 = gaussian_function(x1, x2)

# Step 4: Train a LinearSVC to find a separating hyperplane
X = np.column_stack((x1, x2, x3))
clf = LinearSVC(random_state=0, max_iter=10000)
clf.fit(X, Y)
coef = clf.coef_[0]
intercept = clf.intercept_

# Create a 3D scatter plot with Plotly
fig = go.Figure()

# Add scatter plot for data points
fig.add_trace(go.Scatter3d(
    x=x1, y=x2, z=x3, mode='markers',
    marker=dict(size=5, color=Y, colorscale='Viridis', opacity=0.7),
    name='Data Points'
))

# Create a meshgrid to plot the separating hyperplane
xx, yy = np.meshgrid(np.linspace(min(x1), max(x1), 10),
                     np.linspace(min(x2), max(x2), 10))
zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]

# Add surface plot for the separating hyperplane
fig.add_trace(go.Surface(
    x=xx, y=yy, z=zz, colorscale='Blues', opacity=0.5,
    name='Separating Hyperplane'
))

# Customize layout
fig.update_layout(scene=dict(
    xaxis_title='x1', yaxis_title='x2', zaxis_title='x3'
), title='3D Scatter Plot with Y Color and Separating Hyperplane')

# Use Streamlit to display the Plotly figure
st.plotly_chart(fig, use_container_width=True)
