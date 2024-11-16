import numpy as np
import matplotlib.pyplot as plt
import time

# Helper Function: Get Free DOF
def getFreeIndex(isFixed):
    nv = len(isFixed)
    all_DOFs = np.zeros(nv * 2)  # Two DOFs per node (x and y coordinates)
    for c in range(nv):  # loop over the nodes
        if isFixed[c] != 0:  # c-th node is fixed
            all_DOFs[2*c + 1] = 1  # Fix y-coordinate of c-th node

    free_index = np.where(all_DOFs == 0)[0]
    return free_index

# Objective Function: Newton-Raphson Iteration
def objfun(q_guess, q_old, u_old, dt, tol_dq, maximum_iter, m, mMat, W, free_index):
    q_new = q_guess.copy()
    iter_count = 0
    error = tol_dq * 10
    flag = 1  # Start with a good simulation

    while error > tol_dq:
        # Compute elastic forces (none in this case)
        f = m / dt * ((q_new - q_old) / dt - u_old) - W
        # Compute Jacobian matrix
        J = mMat / dt**2.0
        # We only take the free part of f and J
        f_free = f[free_index]
        J_free = J[np.ix_(free_index, free_index)]

        # Solve for dq (delta q) using linear system
        dq_free = np.linalg.solve(J_free, f_free)

        # Update q_new with the displacement
        q_new[free_index] -= dq_free

        # Calculate error using the change in position
        error = np.linalg.norm(dq_free)
        iter_count += 1

        if iter_count > maximum_iter:
            flag = -1  # Exceeded max iterations
            break

    reactionForce = f  # Reaction forces at fixed DOFs (non-zero)

    return q_new, flag, reactionForce

# Main Simulation Code
# Inputs
nv = 1  # Number of nodes (single ball)
ndof = 2 * nv  # Degrees of freedom (2 per node, x and y)
dt = 1e-4  # Time step
maximum_iter = 100  # Max iterations for solving the system
totalTime = 5  # Total simulation time
tol_dq = 1e-6  # Convergence tolerance for position update

# Initial conditions
nodes = np.zeros((nv, 2))  # Positions of nodes (ball)
nodes[0, 0] = 0  # Initial x position
nodes[0, 1] = 1.0  # Initial y position (1 meter above the ground)

# Mass of the ball
mVector = np.zeros(ndof)
mVector[0:2] = 0.1  # Mass of the ball (0.1 kg)

mMat = np.diag(mVector)  # Mass matrix

# Gravitational force (gravity in the negative y-direction)
W = np.zeros(ndof)
g = np.array([0, -9.8])  # Gravity (m/s^2)
for c in range(nv):
    W[2*c:2*c+2] = mVector[2*c:2*c+2] * g

# Initial velocity (some initial x-velocity and downward velocity)
q0 = np.zeros(ndof)  # Initial positions (zeros)
q0[0] = nodes[0, 0]  # Initial x position
q0[1] = nodes[0, 1]  # Initial y position

q = q0.copy()  # Current positions
u = np.zeros(ndof)  # Initial velocities
u[0:2] = [0.1, -0.05]  # Initial x velocity (0.1 m/s) and downward y velocity (-0.05 m/s)

# Fixed and free DOFs (y-position is fixed when ball hits the ground)
isFixed = np.zeros(nv)  # 0 means free, 1 means fixed
free_index = getFreeIndex(isFixed)  # Get indices of free DOFs

# Bouncing parameters
e = 0.9  # Coefficient of restitution (elasticity of the bounce)

# Number of steps
Nsteps = round(totalTime / dt)
ctime = 0  # Current time
all_pos = np.zeros((Nsteps, ndof))  # To store positions at each step
all_pos[0, :] = q0

# Main time-stepping loop
for timeStep in range(1, Nsteps):
    print(f't = {ctime:.4f} s')
    ground = 0 # Boolean to check if the ball is on ground

    # Predictor step: Guess next position
    q_guess = q0.copy()
    q, error, reactionForce = objfun(q_guess, q0, u, dt, tol_dq, maximum_iter, mVector, mMat, W, free_index)

    # Check for ground hit and apply bounce
    if q[1] <= 0:  # If ball hits the ground
        ground = 1
        q[1] = 0  # Set y-position to 0 (ground level)
        print(f"Hit the ground! Velocity before bounce: {u[1]:.4f} m/s")
        y_vel = -e * u[1] # Reverse and reduce the velocity in y-direction due to bounce
        print(f"Velocity after bounce: {y_vel:.4f} m/s")

    # Check if the ball hits the ground (y < 0) and apply bouncing
    for c in range(nv):
        # If the ball's y-coordinate is below 0 (hits the ground)
        if isFixed[c] == 0 and q[2*c+1] < 0:
            q[2*c+1] = 0  # Set the y-coordinate to the ground level
            y_vel = -u[2*c+1] * e  # Reverse the velocity and scale it by the coefficient of restitution (bounce)
            ground = 1
            print(f"Hit the ground! Velocity before bounce: {y_vel:.4f} m/s")
            break  # Exit the loop since the ball has hit the ground

    # Corrector step (if any additional adjustment needed)
    needCorrector = False
    for c in range(nv):
        if isFixed[c] == 1 and reactionForce[2*c+1] < 0.0:  # If node is fixed and has a negative reaction force
            isFixed[c] = 0  # Free the node (no longer fixed)
            q_guess[2*c+1] = 0.0
            needCorrector = True
            break

    # Corrector step if necessary
    if needCorrector:
        free_index = getFreeIndex(isFixed)
        q, error, reactionForce = objfun(q_guess, q, u, dt, tol_dq, maximum_iter, mVector, mMat, W, free_index)

    # Update velocity and position
    u = (q - q0) / dt
    if ground == 1:
        u[1] = y_vel
    q0 = q.copy()  # Update old position

    # Store position data
    all_pos[timeStep, :] = q0
    ctime += dt  # Increment time



# Plot the results: Position of the ball over time
plt.figure()
plt.plot(all_pos[:, 0], all_pos[:, 1], label="Ball position")
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Ball Bouncing')
# plt.xlim([-0.2, 0.2])
# plt.ylim([0, 1.1])
plt.legend()
plt.grid(True)
plt.show()
