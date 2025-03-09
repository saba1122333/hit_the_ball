from schemes import *
import time
from ball_detector import detect_balls
from schemes import rk4_scheme, euler_scheme
from animate_trajectories import animate_trajectories

# step_frame and pause_time are for animate_trajectories function it matches  solution based on step_size this is for
# animation purposes so animation should not be slow or fast to display results neatly

# step_sizes = [0.01, 0.1, 0.001, 0.0001]
# step_frames = [1, 1, 10, 100]
# pause_times = [0.01, 0.5, 0.01, 0.01]
# angles = [np.pi / 8, np.pi / 6, np.pi / 4, np.pi / 3]

# Adjust as needed to control the time margin for flight travel
flight_margins = [0.5, 1, 2]
schemes = [euler_scheme, rk4_scheme, implicit_rk_scheme]
# test1 = [8, 30]
# test2 = [4, 35]
# test3 = [4, 38]


path = 'test_images/my_test.png'


def performance_test():
    angle = np.pi / 3
    time_margin = 1

    step_size, step_frame, pause_time = 0.01, 1, 0.01
    dimensions, shooting_ball, balls = detect_balls(path, False)

    #
    start_time = time.time()
    animate_trajectories(dimensions, shooting_ball, balls, step_frame, step_size, time_margin, angle,
                         pause_time, implicit_rk_scheme, implicit_rk_scheme)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f'pure implicit scheme performed in {round(execution_time, 2)}s')

    start_time = time.time()
    animate_trajectories(dimensions, shooting_ball, balls, step_frame, step_size, time_margin, angle,
                         pause_time, implicit_rk_scheme, rk4_scheme)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f'mixed scheme performed in {round(execution_time, 2)}s where initial speeds was computed by'
          f' explicit and final trajectories computed with implicit scheme')

    start_time = time.time()
    animate_trajectories(dimensions, shooting_ball, balls, step_frame, step_size, time_margin, angle,
                         pause_time, rk4_scheme, implicit_rk_scheme)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f'mixed scheme performed in {round(execution_time, 2)}s where initials speed was computed by'
          f' implicit and final trajectories computed with explicit scheme')

    start_time = time.time()
    animate_trajectories(dimensions, shooting_ball, balls, step_frame, step_size, time_margin, angle,
                         pause_time, rk4_scheme, rk4_scheme)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f'pure explicit scheme performed in {round(execution_time, 2)}s')


def stability_test():
    angle = np.pi / 3
    time_margin = 1
    step_size, step_frame, pause_time = 0.01, 1, 0.01
    dimensions, shooting_ball, balls = detect_balls(path, False)

    animate_trajectories(dimensions, shooting_ball, balls, step_frame, step_size, time_margin, angle,
                         pause_time, implicit_rk_scheme, euler_scheme)

    step_size, step_frame, pause_time = 0.1, 1, 0.5

    animate_trajectories(dimensions, shooting_ball, balls, step_frame, step_size, time_margin, angle,
                         pause_time, implicit_rk_scheme, implicit_rk_scheme)

    animate_trajectories(dimensions, shooting_ball, balls, step_frame, step_size, time_margin, angle,
                         pause_time, implicit_rk_scheme, euler_scheme)

    animate_trajectories(dimensions, shooting_ball, balls, step_frame, step_size, time_margin, angle,
                         pause_time, implicit_rk_scheme, rk4_scheme)


def flight_test():
    angle = np.pi / 3
    step_size, step_frame, pause_time = 0.01, 1, 0.01
    dimensions, shooting_ball, balls = detect_balls(path, False)
    for flight_margin in flight_margins:
        animate_trajectories(dimensions, shooting_ball, balls, step_frame, step_size, flight_margin, angle,
                             pause_time, implicit_rk_scheme, rk4_scheme)


def finals_tests():
    angle = np.pi / 3
    time_margin = 1
    step_size, step_frame, pause_time = 0.01, 1, 0.01

    epsilon, min_pints = 8, 30  # for test 1
    path1 = 'test_images/1.jpg'
    dimensions, shooting_ball, balls = detect_balls(path1, False, epsilon, min_pints)
    animate_trajectories(dimensions, shooting_ball, balls, step_frame, step_size, time_margin, angle,
                         pause_time,rk4_scheme , rk4_scheme)
    path2 = 'test_images/2.jpg'
    epsilon, min_pints = 4, 35  # for test 2

    dimensions, shooting_ball, balls = detect_balls(path2, False, epsilon, min_pints)
    animate_trajectories(dimensions, shooting_ball, balls, step_frame, step_size, time_margin, angle,
                         pause_time, rk4_scheme, rk4_scheme)



def tester():
    finals_tests()
    # print(f'running: {flight_test.__name__}')
    # flight_test()
    # print(f'running: {performance_test.__name__}')
    # performance_test()
    # print(f'running: {stability_test.__name__}')
    stability_test()
