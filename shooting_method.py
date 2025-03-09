from schemes import rk4_scheme, newtons_method_for_nonlinear_systems

from motion_model import *

derivative_functions = [dx_dt, dy_dt, dvx_dt, dvy_dt]


def calculate_initial_guesses(dx, dy, time_margin, angle, g=9.81):
    distance = np.sqrt(dx ** 2 + dy ** 2)
    t_min = np.sqrt(2 * distance / g)
    t_end = t_min * time_margin
    speed = np.sqrt((distance * g) / np.sin(2 * angle))
    velocity = np.clip(speed, 0, 10000)
    return velocity * np.cos(angle), velocity * np.sin(angle), t_end


def construct_shooting_functions(derivative_functions, x_coord, y_coord, t_start, t_end, step_size,
                                 scheme_for_initial_velocities):
    shooting_pos_x, target_pos_x = x_coord
    shooting_pos_y, target_pos_y = y_coord


    def f_sx(velocities):
        sx, sy = velocities
        initial_conditions = [shooting_pos_x, shooting_pos_y, sx, sy]

        _, solution_matrix = scheme_for_initial_velocities(derivative_functions, t_start, t_end, initial_conditions, step_size)
        return solution_matrix[0, -1] - target_pos_x   # g(sx) = ux(b,s) - vbx

    def f_sy(velocities):
        sx, sy = velocities
        initial_conditions = [shooting_pos_x, shooting_pos_y, sx, sy]

        _, solution_matrix = scheme_for_initial_velocities(derivative_functions, t_start, t_end, initial_conditions, step_size)
        return solution_matrix[1, -1] - target_pos_y # g(sy) = uy(b,s) - vby

    return [f_sx, f_sy]


def shooting_method(derivative_functions, x_coord, y_coord, step_size, time_margin, angle, scheme_for_shooting_method,scheme_for_initial_velocities):
    shooting_pos_x, target_pos_x = x_coord
    shooting_pos_y, target_pos_y = y_coord

    dx = target_pos_x - shooting_pos_x
    dy = target_pos_y - shooting_pos_y
    t_start = 0

    initial_sx, initial_sy, t_end = calculate_initial_guesses(dx, dy, time_margin, angle)

    f_sx, f_sy = construct_shooting_functions(derivative_functions, x_coord, y_coord, t_start, t_end, step_size,
                                              scheme_for_initial_velocities)

    final_sx, final_sy = newtons_method_for_nonlinear_systems([f_sx, f_sy], step_size, [initial_sx, initial_sy])

    final_conditions = [shooting_pos_x, shooting_pos_y, final_sx, final_sy]
    return scheme_for_shooting_method(derivative_functions, t_start, t_end, final_conditions, step_size)
