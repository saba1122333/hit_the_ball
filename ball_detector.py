import random
from collections import deque

import matplotlib.pyplot as plt

from balls_generator import *


def get_neighbors(points, point_idx, epsilon):
    #F ind points within epsilon distance of given point
    distances = np.linalg.norm(points - points[point_idx], axis=1)
    return np.where(distances <= epsilon)[0]


def expand_cluster(points, point_idx, neighbors, labels, cluster_id, epsilon, min_points):
    """Expand cluster from seed point"""
    labels[point_idx] = cluster_id
    queue = deque(neighbors)

    while queue:
        current = queue.popleft()
        if labels[current] == 0:  # Was noise
            labels[current] = cluster_id

        if labels[current] != -1:
            continue

        labels[current] = cluster_id
        current_neighbors = get_neighbors(points, current, epsilon)

        if len(current_neighbors) >= min_points:
            queue.extend(current_neighbors)


def dbscan(points, epsilon=2, min_points=30):
    """Run DBSCAN clustering"""
    points = np.array(points)
    labels = np.full(len(points), -1)
    ball = 0

    for i in range(len(points)):
        if labels[i] != -1:
            continue

        neighbors = get_neighbors(points, i, epsilon)
        if len(neighbors) < min_points:
            labels[i] = 0  # Noise
            continue

        ball += 1
        expand_cluster(points, i, neighbors, labels, ball, epsilon, min_points)

    return labels, ball


def extract_circles(image, points, labels, num_clusters):
    balls = []
    for cluster_id in range(1, num_clusters + 1):
        cluster_points = points[labels == cluster_id]
        if len(cluster_points) < 2:
            continue
        center = np.mean(cluster_points, axis=0)
        radius = np.mean(np.linalg.norm(cluster_points - center, axis=1))
        x, y = (int(center[0]), int(center[1]))
        color = image[y, x].tolist()  # BGR format
        color_rgb = tuple(c / 255 for c in color[::-1])  # Convert BGR to normalized RGB

        balls.append((int(center[0]), image.shape[1] - int(center[1]), int(radius), color_rgb))
    return balls


def detect_edges(image, threshold):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    rows, cols = gray.shape
    up = gray[:-2, 1:-1]  # Pixels above
    down = gray[2:, 1:-1]  # Pixels below
    left = gray[1:-1, :-2]  # Pixels to the left
    right = gray[1:-1, 2:]  # Pixels to the right
    center = gray[1:-1, 1:-1]  # Center pixels

    diff_up = np.abs(center - up)
    diff_down = np.abs(center - down)
    diff_left = np.abs(center - left)
    diff_right = np.abs(center - right)

    # Combine all differences
    max_diff = np.maximum.reduce([diff_up, diff_down, diff_left, diff_right])

    # Find where differences exceed threshold
    edge_mask = max_diff > threshold

    # Get coordinates of edge points
    y_coords, x_coords = np.where(edge_mask)

    # Adjust coordinates to account for border removal
    x_coords += 1
    y_coords += 1

    # Return as array of [x, y] coordinates
    plt.scatter(x_coords,image.shape[0]- y_coords)
    plt.show()

    return np.column_stack((x_coords, y_coords))


def detect_balls(image_path,is_performance_test_on = False, epsilon=5, min_points=5, edge_threshold=30):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image from path")

    # Detect edge points
    edge_points = detect_edges(image, edge_threshold)

    # Run DBSCAN clustering
    labels, num_clusters = dbscan(edge_points, epsilon, min_points)

    # Extract balls
    balls = extract_circles(image, edge_points, labels, num_clusters)

    shooting_ball = add_shooting_ball(image.copy(), balls)
    if not is_performance_test_on:
        plot_balls(image, balls, shooting_ball)

    return [image.shape[0], image.shape[1]], shooting_ball, balls


def plot_balls(image, balls,shooting_ball=None, figsize=(12, 4)):
    plt.figure(figsize=figsize)
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(132)
    plt.xlim(0, image.shape[1])
    plt.ylim(0, image.shape[0])
    # Plot detected balls
    for ball in balls:
        circle = plt.Circle((ball[0], ball[1]), ball[2], fill=True, color=ball[3])
        plt.gca().add_patch(circle)

    if shooting_ball is not None:
        plt.text(shooting_ball[0], shooting_ball[1] + 20, "shooting_ball",
                 fontsize=8,
                 color='red',

                 ha='center',
                 va='top'
                 )
        circle = plt.Circle((shooting_ball[0], shooting_ball[1]),
                            shooting_ball[2], fill=True, color=shooting_ball[3])
        plt.gca().add_patch(circle)
    plt.axis('equal')
    plt.grid(True)
    plt.title('Processed Image with detected Target')
    plt.show()
