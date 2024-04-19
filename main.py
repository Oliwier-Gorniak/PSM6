import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

L = np.pi
N = 10
Dx = L / N
c = 1
T = 10
dt = 0.05

points_on_string = np.linspace(0, L, N + 1)
initial_shape = np.sin(points_on_string)
initial_v = np.zeros(N + 1)

initial_conditions = np.concatenate([initial_shape, initial_v])


def wave_equation(time, state):
    shape = state[:N + 1]
    velocity = state[N + 1:]
    d_shape_dt = velocity
    d_vel_dt = np.zeros(N + 1)
    d_vel_dt[1:N] = (shape[:-2] - 2 * shape[1:-1] + shape[
                                                                                 2:]) / Dx ** 2
    return np.concatenate([d_shape_dt, d_vel_dt])


solution = solve_ivp(wave_equation, [0, T], initial_conditions,
                     t_eval=np.arange(0, T, dt), method='RK45')

time = solution.t
shape = solution.y[:N + 1]
v = solution.y[N + 1:]

Ek = np.sum(Dx * v ** 2 / 2, axis=0)
Ep = np.sum((np.diff(shape, axis=0) ** 2) / (2 * Dx), axis=0)
Ec = Ek + Ep

fig, ax = plt.subplots()
ax.plot(time, Ek, label='Kinetic Energy')
ax.plot(time, Ep, label='Potential Energy')
ax.plot(time, Ec, label='Total Energy')
ax.set_xlabel('Time')
ax.set_ylabel('Energy')
ax.legend()
plt.show()
