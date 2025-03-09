import cv2
import numpy as np
import matplotlib.pyplot as plt


def check_collision(x, y, r, existing_balls):
    # Check if a new ball collides with any existing balls
    for ex, ey, er in existing_balls:
        distance = np.sqrt((x - ex) ** 2 + (y - ey) ** 2)
        if distance < (r + er + 0.1):
            return True
    return False


def is_in_corner(x, y, r, corner_size=1.5):
    # Check if ball is too close to any corner
    corners = [(0, 0), (0, 10), (10, 0), (10, 10)]
    for cx, cy in corners:
        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if distance < corner_size + r:
            return True
    return False


def generate_ball_image(num_balls=10, min_radius=0.1, max_radius=0.2, image_size=500):
    # Create blank image array
    image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255

    # Scale factors to convert from 10x10 space to image dimensions
    scale = image_size / 15

    existing_balls = []
    for _ in range(num_balls):
        radius = np.random.uniform(min_radius, max_radius)
        x = np.random.uniform(radius, 15 - radius)
        y = np.random.uniform(radius, 15 - radius)

        if not check_collision(x, y, radius, existing_balls) and not is_in_corner(x, y, radius):
            existing_balls.append((x, y, radius))

            # Convert coordinates to image space
            center = (int(x * scale), int(y * scale))
            radius_px = int(radius * scale)

            # Draw circle on image
            color = tuple(map(lambda x: int(x * 255), np.random.rand(3)))
            cv2.circle(image, center, radius_px, color, -1)

    return image


def add_shooting_ball(image, existing_balls, radius=0.2):
    scale = image.shape[0] / 15  # Use image size for correct scaling
    attempts = 0
    max_attempts = 100

    while attempts < max_attempts:
        shooting_x = np.random.uniform(radius, 15 - radius)
        shooting_y = np.random.uniform(radius, 15 - radius)

        # Convert existing balls to same scale
        scaled_existing = []
        for x, y, r, color in existing_balls:
            scaled_existing.append((x / scale, y / scale, r / scale))

        if not check_collision(shooting_x, shooting_y, radius, scaled_existing):
            center = (int(shooting_x * scale), int(shooting_y * scale))
            radius_px = int(radius * scale)

            return center[0], center[1], radius_px, [0, 0, 0]

        attempts += 1

    return None


def generate(num_balls=7, min_radius=0.4, max_radius=0.8):

    image = generate_ball_image(num_balls, min_radius, max_radius)
    cv2.imwrite(f'test_images/test_balls{5}.png', image)

