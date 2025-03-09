import numpy as np

k = 0.47  # resistance  for sphire
m = 0.5  # mass of ball
g = 9.81

def dx_dt(t, state):
    x, y, vx, vy = state[0], state[1], state[2], state[3]
    return vx

def dy_dt(t, state):
    x, y, vx, vy = state[0], state[1], state[2], state[3]
    return vy


def dvx_dt(t, state):
    x, y, vx, vy = state[0], state[1], state[2], state[3]
    v = np.hypot(vx, vy)

    return -(k / m) * vx * v

def dvy_dt(t, state):
    x, y, vx, vy = state[0], state[1], state[2], state[3]
    v = np.hypot(vx, vy)
    return -g - (k / m) * vy * v
