import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_moons

NUM_POINTS = 500

def save_plot(plt, question, name, attempt):
    output_dir = os.path.join('../output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f"Question{question}_{name}_{attempt}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

## Question A ##
def plot_data_unif(x_range, y_range, color, label, attempt):
    x = np.random.uniform(x_range[0], x_range[1], NUM_POINTS)
    y = np.random.uniform(y_range[0], y_range[1], NUM_POINTS)
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, color=color, alpha=0.6, edgecolors='w', label=label)
    plt.title(f'Uniform Distribution ({label})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    save_plot(plt, '1A', 'UNIF', attempt)

x_range = [-10, 2]
y_range = [18, 45]

plot_data_unif(x_range, y_range, 'blue', '1st', '1st')
plot_data_unif(x_range, y_range, 'red', '2nd', '2nd')

### Question B ###
def plot_data_gaus(centers, stds, colors, labels, attempt):
    plt.figure(figsize=(10, 5))
    for i in range(len(centers)):
        x = np.random.normal(centers[i][0], stds[i], NUM_POINTS)
        y = np.random.normal(centers[i][1], stds[i], NUM_POINTS)
        plt.scatter(x, y, color=colors[i], alpha=0.6, edgecolors='w', label=labels[i])
    plt.title('Gaussian Distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    save_plot(plt, '1B', 'GAUS', attempt)

centers = [[1, -1], [2, -2], [4, -4]]
stds = [0.5 * i for i in [1, 2, 4]]
colors = ['blue', 'red', 'green']
labels = ['Gaussian 1', 'Gaussian 2', 'Gaussian 3']

plot_data_gaus(centers, stds, colors, labels, '1st')
plot_data_gaus(centers, stds, colors, labels, '2nd')

### Question C ###
def plot_data_letters(points_func, label, attempt):
    points = points_func()
    x, y = points[:, 0], points[:, 1]
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, alpha=0.6, edgecolors='w', label=label)
    plt.title(f'Letters {label}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    save_plot(plt, '1C', 'LETTERS', attempt)

def generate_y(offset_x=0, offset_y=0):
    points = []
    for i in range(20):
        points.append((offset_x + 0, offset_y + i))
    for i in range(20):
        points.append((offset_x - i, offset_y + 20 + i))
    for i in range(20):
        points.append((offset_x + i, offset_y + 20 + i))
    points = np.array(points) + np.random.normal(0, 0.5, (len(points), 2))
    return points

def generate_s(offset_x=0, offset_y=0):
    points = []
    for i in range(20):
        points.append((offset_x + i, offset_y + 20))
    for i in range(20):
        points.append((offset_x + i, offset_y + 20 - i))
    for i in range(20):
        points.append((offset_x + i, offset_y + 0))
    points = np.array(points) + np.random.normal(0, 0.5, (len(points), 2))
    return points

def generate_yyys():
    y1_points = generate_y(-30, 0)
    y2_points = generate_y(0, 0)
    y3_points = generate_y(30, 0)
    s_points = generate_s(60, 0)
    return np.concatenate((y1_points, y2_points, y3_points, s_points), axis=0)

points_func = generate_yyys

plot_data_letters(points_func, 'YYY_S', '1st')
plot_data_letters(points_func, 'YYY_S', '2nd')

### Question D ###
def plot_horizontal_clumps(NUM_POINTS, color='black', attempt='1st'):
    x_positions = [0, 5, 0, 5]
    y_positions = [0, 0, 2, 2]

    plt.figure(figsize=(8, 6))

    for i in range(4):
        x = np.random.normal(x_positions[i], 0.5, NUM_POINTS)
        y = np.random.normal(y_positions[i], 0.1, NUM_POINTS)
        plt.scatter(x, y, color=color, alpha=0.6, edgecolors='w')

    plt.title('Four Horizontal Clumps')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.xlim(-2, 8)
    plt.ylim(-3, 6)
    save_plot(plt, '1D', 'CLUMPS', attempt)

plot_horizontal_clumps(NUM_POINTS, attempt='1st')
plot_horizontal_clumps(NUM_POINTS, attempt='2nd')

### Question E ###
def plot_data_moons_dense_separated(attempt):
    x, y = make_moons(n_samples=NUM_POINTS, noise=0.05)
    x[:, 0] *= 1.2
    plt.figure(figsize=(10, 5))
    plt.scatter(x[:, 0], x[:, 1], alpha=0.6, edgecolors='w')
    plt.title('Two Moons - Dense and Separated')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    save_plot(plt, '1E', 'MOONS', attempt)

plot_data_moons_dense_separated('1st')
plot_data_moons_dense_separated('2nd')

### Question F ###
def plot_data_moons_sparse(attempt):
    x, y = make_moons(n_samples=NUM_POINTS, noise=0.1)
    idx = np.random.choice(len(x), size=50, replace=False)
    x[idx] += np.random.normal(0, 1, (50, 2))
    plt.figure(figsize=(10, 5))
    plt.scatter(x[:, 0], x[:, 1], alpha=0.6, edgecolors='w')
    plt.title('Two Moons Sparse')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    save_plot(plt, '1F', 'MOONS_SPARSE', attempt)

plot_data_moons_sparse('1st')
plot_data_moons_sparse('2nd')