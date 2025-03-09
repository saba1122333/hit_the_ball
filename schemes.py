import numpy as np


def newtons_method_for_nonlinear_systems(system, step_size, initial_guess, num_iter=10, tol=10 ** -7):
    n = len(system)
    jacobi = np.zeros((n, n))
    current = np.array(initial_guess)
    for _ in range(num_iter):
        current_eval = np.zeros(n)
        for j in range(n):
            current_eval[j] = -system[j](current)
            for i in range(n):
                tuned = current.copy()
                tuned[i] += step_size
                jacobi[j, i] = (system[j](tuned) - system[j](current)) / step_size
        try:
            s = np.linalg.solve(jacobi, current_eval)
        except np.linalg.LinAlgError:

            damped_jacobi = jacobi + 1e-6 * np.eye(n)
            s = np.linalg.solve(damped_jacobi, current_eval) * 0.1

        if np.linalg.norm(s) < tol:
            return current

        # to prevent big jumps when system is ill constructed
        s = np.clip(s, -100, 100)
        current += s

    return current


def rk4_scheme(derivatives, t_start, t_end, initial_conditions, step_size):
    # Calculate number of points needed
    num_steps = max(2, int(np.ceil((t_end - t_start) / step_size)))

    # Create evenly spaced time points
    t_vals = np.linspace(t_start, t_end, num_steps)

    # The actual step size between points
    actual_step_size = (t_end - t_start) / (num_steps - 1)

    num_eq = len(initial_conditions)
    y_vals = np.zeros((num_eq, num_steps))
    y_vals[:, 0] = initial_conditions

    for i in range(1, num_steps):
        t = t_vals[i - 1]
        y = y_vals[:, i - 1]
        try:
            k1 = np.array([f(t, y) for f in derivatives])
            k2 = np.array([f(t + actual_step_size / 2, y + actual_step_size / 2 * k1) for f in derivatives])
            k3 = np.array([f(t + actual_step_size / 2, y + actual_step_size / 2 * k2) for f in derivatives])
            k4 = np.array([f(t + actual_step_size, y + actual_step_size * k3) for f in derivatives])

            increment = (actual_step_size / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            y_vals[:, i] = y + np.clip(increment, -1e10, 1e10)

        except (OverflowError, FloatingPointError):
            k1 = np.array([f(t, y) for f in derivatives])
            y_vals[:, i] = y + actual_step_size * k1

    return t_vals, y_vals


def euler_scheme(derivatives, t_start, t_end, initial_conditions, step_size):
    # Calculate number of points needed
    num_steps = max(2, int(np.ceil((t_end - t_start) / step_size)))

    # Create evenly spaced time points
    t_vals = np.linspace(t_start, t_end, num_steps)

    # The actual step size between points
    actual_step_size = (t_end - t_start) / (num_steps - 1)

    num_eq = len(initial_conditions)
    y_vals = np.zeros((num_eq, num_steps))
    y_vals[:, 0] = initial_conditions

    for i in range(1, num_steps):
        t = t_vals[i - 1]
        y = y_vals[:, i - 1]
        try:
            # Compute derivatives for the current state
            k = np.array([f(t, y) for f in derivatives])

            # Euler step with increment clipping for numerical stability
            increment = actual_step_size * k
            y_vals[:, i] = y + np.clip(increment, -1e10, 1e10)

        except (OverflowError, FloatingPointError):
            # Fallback to simple Euler step if numerical issues occur
            k = np.array([f(t, y) for f in derivatives])
            y_vals[:, i] = y + actual_step_size * k

    return t_vals, y_vals





def implicit_rk_step(derivative_functions, t_current, y_current, step_size, previous_ks):

    # c = np.array([1 / 2, 2 / 3, 1 / 2, 1])  # Stage times
    # A = np.array([
    #     [1 / 2, 0, 0, 0],
    #     [1 / 6, 1 / 2, 0, 0],
    #     [-1 / 2, 1 / 2, 1 / 2, 0],
    #     [3 / 2, -3 / 2, 1 / 2, 1 / 2]
    # ])  # Coefficients for intermediate steps
    # b = np.array([3 / 2, -3 / 2, 1 / 2, 1 / 2])  # Weights for the final combination
    c = np.array([0, 1])  # Stage times
    A = np.array([
        [0, 0],
        [1 / 2, 1 / 2]
    ])  # Coefficients for intermediate steps
    b = np.array([1 / 2, 1 / 2])  # Weights for the final combination

    num_stages = len(A)
    num_equations = len(derivative_functions)
    system = []

    # system construction
    def system_constructor(stage_index, eq_index):

        def equation(k_values):
            # this implementation expects that k_values sorted by stage_index
            # rows represent k stages
            # columns represent equations
            k_matrix = k_values.reshape(num_stages, num_equations)
            stage_y = np.zeros(num_equations)
            stage_time = t_current + c[stage_index] * step_size

            for eq in range(num_equations):
                stage_y[eq] = y_current[eq] + step_size*np.dot(A[stage_index], k_matrix[:, eq])

            return k_matrix[stage_index, eq_index] - derivative_functions[eq_index](stage_time, stage_y)

        return equation

    for i in range(num_stages):
        for j in range(num_equations):
            system.append(system_constructor(i, j))

    initial_guess = previous_ks

    k_results = newtons_method_for_nonlinear_systems(system, step_size, initial_guess)

    k_results_matrix = k_results.reshape(num_stages, num_equations)
    y_next = np.zeros_like(y_current)
    for eq in range(num_equations):
        y_next[eq] = y_current[eq] + step_size * np.dot(b, k_results_matrix[:, eq])
    return y_next, k_results


def implicit_rk_scheme(derivative_functions, t_start, t_end, initial_conditions, step_size, number_of_stages=2):
    num_steps = int((t_end - t_start) / step_size) + 1
    num_equations = len(initial_conditions)
    t_values = np.zeros(num_steps)
    y_values = np.zeros((num_equations, num_steps))
    t_values[0] = t_start
    y_values[:, 0] = initial_conditions

    previous_ks = np.zeros(number_of_stages * num_equations)

    for stage in range(number_of_stages):
        for eq in range(num_equations):
            index = stage * num_equations + eq
            previous_ks[index] = derivative_functions[eq](t_start, initial_conditions)

    for i in range(1, num_steps):
        t_current = t_values[i - 1]
        y_current = y_values[:, i - 1]

        y_next, k_values = implicit_rk_step(derivative_functions, t_current, y_current, step_size, previous_ks)

        previous_ks = k_values

        y_values[:, i] = y_next
        t_values[i] = t_current + step_size

    return t_values, y_values
