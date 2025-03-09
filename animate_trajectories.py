from matplotlib import pyplot as plt
from motion_model import *
from shooting_method import shooting_method
derivative_functions = [dx_dt, dy_dt, dvx_dt, dvy_dt]


def order_function(shooting_ball, ball):
    shooting_x, shooting_y, _, _ = shooting_ball
    ball_x, ball_y, ball_radius, _ = ball
    return np.sqrt((shooting_x - ball_x) ** 2 + (shooting_y - ball_y) ** 2) - ball_radius


def shooting_order(shooting_ball, balls):
    weights = [(i, order_function(shooting_ball, ball)) for i, ball in enumerate(balls)]
    weights.sort(key=lambda x: x[1])
    return [balls[i] for i, _ in weights]


def trajectory(dimension, shooting_ball, target, step_size, time_margin, angle,
               scheme_for_shooting_method, scheme_for_initial_velocities):
    length, height = dimension
    shooting_pos_x, shooting_pos_y, _, _ = shooting_ball
    target_x, target_y, _, _ = target

    shooting_ball_x, shooting_ball_y = shooting_pos_x / length, shooting_pos_y / height
    target_x, target_y = target_x / length, target_y / height

    t_vals, solution_matrix = shooting_method(derivative_functions,
                                              [shooting_ball_x, target_x],
                                              [shooting_ball_y, target_y], step_size, time_margin, angle,
                                              scheme_for_shooting_method, scheme_for_initial_velocities)

    x_coords = solution_matrix[0, :] * length
    y_coords = solution_matrix[1, :] * height
    return x_coords, y_coords


def trajectories(dimension, shooting_ball, balls, step_size, time_margin,angle,
                 scheme_for_shooting_method,scheme_for_initial_velocities):
    ordered_balls = shooting_order(shooting_ball, balls)
    motion_paths = []
    for target in ordered_balls:
        motion_paths.append(trajectory(dimension, shooting_ball, target, step_size, time_margin,angle,
                                       scheme_for_shooting_method,scheme_for_initial_velocities))
    return motion_paths, ordered_balls


def check_collision(shooting_pos, target, shooting_radius, target_radius):
    dist = np.sqrt((shooting_pos[0] - target[0]) ** 2 + (shooting_pos[1] - target[1]) ** 2)
    return dist <= (shooting_radius + target_radius)



def animate_trajectories(dimensions, shooting_ball, balls, step_frame, step_size, time_margin, angle, pause_time,
                         scheme_for_shooting_method, scheme_for_initial_velocities):
    length, height = dimensions
    motion_paths, ordered_balls = trajectories(dimensions, shooting_ball, balls, step_size, time_margin,angle,
                                               scheme_for_shooting_method,scheme_for_initial_velocities)
    remaining_balls = ordered_balls.copy()

    plt.figure(figsize=(10, 10))
    ball_circles = []

    for ball in ordered_balls:
        x, y, r, color = ball
        circle = plt.Circle((x, y), r, color=color)
        ball_circles.append(circle)
        plt.gca().add_patch(circle)

    shooting_circle = plt.Circle((shooting_ball[0], shooting_ball[1]),
                                 shooting_ball[2], color=shooting_ball[3])
    plt.gca().add_patch(shooting_circle)
    plt.grid(True)
    plt.plot(shooting_ball[0], shooting_ball[1],
             '*',
             markersize=15,
             color='red',
             linewidth=2,
             label='Aim Point'  # This adds a legend entry
             )
    plt.legend()

    i = 0
    while i < len(motion_paths):

        path = motion_paths[i]
        target = ordered_balls[i]
        x_coords, y_coords = path
        trajectory_line = plt.plot(x_coords, y_coords, '--', alpha=0.3)[0]

        hit_unintended = False
        hit_intended = False

        # Always animate along the path, even if target is gone
        for j in range(0, len(x_coords), step_frame):
            new_pos = (x_coords[j], y_coords[j])

            # Only check collisions if there are remaining balls
            if remaining_balls:
                for idx, ball in enumerate(remaining_balls):
                    if check_collision(new_pos, ball[:2], shooting_ball[2], ball[2]):
                        if ball != target:  # Hit unintended ball
                            shooting_circle.center = new_pos
                            plt.pause(0.1)
                            # Remove the unintended ball
                            ball_circles[ordered_balls.index(ball)].remove()
                            remaining_balls.remove(ball)
                            # Reset shooting ball position
                            shooting_circle.center = (shooting_ball[0], shooting_ball[1])
                            hit_unintended = True
                            break
                        elif target in remaining_balls:  # Hit intended target (if it still exists)
                            shooting_circle.center = new_pos
                            plt.pause(0.1)
                            # Remove the target ball
                            ball_circles[ordered_balls.index(ball)].remove()
                            remaining_balls.remove(ball)
                            hit_intended = True
                            break

            if hit_unintended or hit_intended:
                break

            shooting_circle.center = new_pos
            plt.title(
                f'Solved with: {scheme_for_shooting_method.__name__},'
                f' Initial Velocities found by {scheme_for_initial_velocities.__name__}, '
                f'Step_size: {step_size},'
                f' Time_margin: {time_margin},'
                f' Paths: {len(motion_paths)}, Path: {i + 1}')
            plt.pause(pause_time)

        shooting_circle.center = (shooting_ball[0], shooting_ball[1])

        if not hit_unintended:  # Move to next path if we didn't hit an unintended ball
            i += 1

        plt.pause(0.5)

        if not remaining_balls:
            break
    plt.pause(0.2)
    plt.close()
