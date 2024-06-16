# Simulating Constrained Dynamics of a Double Pendulum with JAX

In this post, we'll delve into the fascinating world of simulating the constrained dynamics of mechanical systems, focusing on a double pendulum. We'll use JAX, a high-performance numerical computing library, to model the system and perform simulations. The code snippets provided will illustrate the process step-by-step, allowing you to follow along and understand the underlying theory.

## Introduction to Double Pendulum Dynamics

A double pendulum consists of two pendulums attached end-to-end. It exhibits rich and complex dynamics due to the nonlinear nature of its motion and the constraints imposed by the lengths of the rods connecting the masses.

### Parameters and Constraints

We start by defining the parameters of the double pendulum system, including masses, lengths, and gravitational acceleration. The constraints ensure that the pendulum maintains its geometric configuration throughout the simulation.

```python
import jax
import jax.numpy as jnp
from jax import jit, jacfwd, vmap
from time import perf_counter
import numpy as np
from visualize import visualize_double_pendulum

# Define parameters
masses = 1.0, 2.0
lengths = 0.8, 1.2
g = 9.81

# Define constraints
def constraints(q):
    """
    Compute the constraints for the double pendulum system.

    Parameters:
    q (array): The current state [x1, y1, x2, y2]

    Returns:
    phi (array): The constraints [phi1, phi2]
    """
    x1, y1, x2, y2 = q
    phi = jnp.array([x1**2 + y1**2 - lengths[0]**2,
                     (x2 - x1)**2 + (y2 - y1)**2 - lengths[1]**2])
    return phi
```

The constraints \(\phi\) are given by:

\[
\phi_1 = x_1^2 + y_1^2 - L_1^2 = 0
\]
\[
\phi_2 = (x_2 - x_1)^2 + (y_2 - y_1)^2 - L_2^2 = 0
\]

These constraints ensure that each mass stays on its respective rod.

## Free Dynamics and Mass Matrix

Next, we define the mass matrix and external forces acting on the system. The mass matrix is crucial for modeling the inertial properties of the pendulum.

```python
# Define free dynamics parameters
M = jnp.eye(4)
M = M.at[:2, :2].set(masses[0] * jnp.eye(2))
M = M.at[2:, 2:].set(masses[1] * jnp.eye(2))
h = jnp.array([0, masses[0] * g, 0, masses[1] * g])
```

The mass matrix \( \mathbf{M} \) is:

\[
\mathbf{M} = \begin{bmatrix}
m_1 & 0 & 0 & 0 \\
0 & m_1 & 0 & 0 \\
0 & 0 & m_2 & 0 \\
0 & 0 & 0 & m_2
\end{bmatrix}
\]

And the external forces \( \mathbf{h} \) are due to gravity:

\[
\mathbf{h} = \begin{bmatrix}
0 \\
m_1 g \\
0 \\
m_2 g
\end{bmatrix}
\]

## Initial Conditions and JIT Compilation

We initialize the state of the system and use JAX's JIT compilation to optimize the performance of our constraint computations and dynamics updates.

```python
# Time parameters
t_stop = 20
dt = 1/50
t = jnp.arange(0, t_stop, dt)
N = len(t)

BATCH_DIM = 1000  # Define the batch dimension

# Initial conditions
key = jax.random.PRNGKey(0)
initial_q = jax.random.uniform(key, (BATCH_DIM, 4), minval=0.9*lengths[0], maxval=1.1*lengths[0])
initial_q = initial_q.at[:, 2].set(initial_q[:, 0] + jax.random.uniform(key, (BATCH_DIM,), minval=0.9*lengths[1], maxval=1.1*lengths[1]))
initial_q = initial_q.at[:, 1].set(0)
initial_q = initial_q.at[:, 3].set(0)
initial_v = jnp.zeros((BATCH_DIM, 4))

# JIT compile the functions for better performance
constraints_jit = jit(constraints)

# Autodiff to obtain Jacobians
constraints_jacobian = jacfwd(constraints)

# Compute constraint velocity
constraint_velocity = jit(lambda q, v: constraints_jacobian(q) @ v)

# Differentiate constraint_velocity with respect to q
jacobian_derivative = jit(jacfwd(lambda q, v: constraint_velocity(q, v), argnums=0))
```

## Dynamics Step Function and KKT System

We define the dynamics step function, which updates the state of the system based on the current state, velocities, and constraints. This function forms and solves the Karush-Kuhn-Tucker (KKT) system to enforce the constraints.

### The KKT Matrix and Lagrange Multipliers

The KKT conditions are necessary for a solution in nonlinear programming to be optimal, given some regularity conditions. They are a generalization of the method of Lagrange multipliers. The KKT matrix is a block matrix that includes the mass matrix, the constraint Jacobian, and their transposes. It forms a linear system that couples the dynamics and the constraints.

Lagrange multipliers (\(\lambda\)) are introduced to convert a constrained problem into an unconstrained one by incorporating the constraints into the objective function. In the context of the double pendulum, they ensure that the motion adheres to the length constraints of the pendulum arms.

The KKT matrix \( \mathbf{A} \) is:

\[
\mathbf{A} = \begin{bmatrix}
\mathbf{M} & \mathbf{J}^T \\
\mathbf{J} & \mathbf{0}
\end{bmatrix}
\]

where \( \mathbf{J} \) is the Jacobian of the constraints.

The bias term \( \mathbf{b} \) includes contributions from the constraint forces and their derivatives.

```python
@jit
def dynamics_step(q, v, M, h, dt):
    """
    Perform a single dynamics step for the double pendulum system.

    Parameters:
    q (array): The current state [x1, y1, x2, y2]
    v (array): The current velocities
    M (array): The mass matrix
    h (array): The external forces
    dt (float): The time step

    Returns:
    q (array): The updated state
    v (array): The updated velocities
    """
    J = constraints_jacobian(q)
    m, n = J.shape

    # Form KKT system
    A = jnp.zeros((n + m, n + m))
    A = A.at[:n, :n].set(M)
    A = A.at[:n, n:].set(J.T)
    A = A.at[n:, :n].set(J)

    b = jnp.zeros(n + m)
    b = b.at[:n].set(-h)

    # Form bias term
    bias = -jacobian_derivative(q, v) @ v

    # Baumgarte constraint stabilization
    omega = 4
    dphi = constraint_velocity(q, v)
    phi = constraints(q)
    bias += -(omega**2) * phi - 2 * omega * dphi
    b = b.at[n:].set(bias)

    # Solve the linear system
    y = jnp.linalg.solve(A, b)
    dv, lambd = y[:n], -y[n:]

    v = v + dv * dt
    q = q + v * dt
    return q, v
```

## Simulation Loop and Visualization

Finally, we run the simulation loop, updating the state of the system at each time step, and visualize the motion of the double pendulum using the provided `visualize_double_pendulum` function.

```python
# Vectorize the dynamics_step function
batched_dynamics_step = jit(vmap(dynamics_step, in_axes=(0, 0, None, None, None)))

# Simulation loop
solution = np.zeros((N, BATCH_DIM, 4))
v = initial_v

print('First call for jitted function')
t1 = perf_counter()
batched_dynamics_step(initial_q, initial_v, M, h, dt)
t2 = perf_counter()
print(f'First call took {1000 * (t2 - t1):.2f} ms')
q, v = initial_q, initial_v
# Run the simulation
t1 = perf_counter()
for i in range(N):
    q, v = batched_dynamics_step(q, v, M, h, dt)
    solution[i] = np.array(q)  # Store result in NumPy array
t2 = perf_counter()
print(f'Simulation for batch of {BATCH_DIM} systems finished in {1000 * (t2 - t1):.2f} ms')
print(f'The average time for one step for batch of {BATCH_DIM}

 systems is: {1000 * ((t2 - t1)/N):.2f} ms')  # Execution time in milliseconds for dynamics_step

# Extract points for further processing or analysis
points_1 = np.array([solution[:, :, 0], solution[:, :, 1]])
points_2 = np.array([solution[:, :, 2], solution[:, :, 3]])

joint_points = [points_1, points_2]

# Visualize the pendulums
visualize_double_pendulum(joint_points,
                          save=False,  # save as html animation
                          axes=False,
                          show=True,
                          trace_len=0.2,
                          dt=dt,
                          batch_indices=range(20))
```

## Conclusion

Simulating the constrained dynamics of mechanical systems like the double pendulum offers deep insights into their behavior and complexity. By leveraging JAX's capabilities, we can perform these simulations efficiently, even for large batches of systems. The provided code serves as a foundation, and you can further customize and expand it to explore more complex scenarios or different types of constraints.

Feel free to reach out with any questions or suggestions, and happy simulating!
