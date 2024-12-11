import numpy as np
import matplotlib.pyplot as plt

"""# Helper Function"""

def getFreeIndex(isFixed):
  nv = len(isFixed)
  all_DOFs = np.zeros(ndof)
  for c in range(nv): # loop over the nodes
    if isFixed[c] != 0: # c-th node is fixed
      all_DOFs[2*c+1] = 1 # Fix y-coordinate of c-th node

  free_index = np.where(all_DOFs == 0)[0]
  return free_index

"""# Objective Function"""

def objfun(q_guess, q_old, u_old, dt, tol_dq, maximum_iter, m, mMat, W, free_index):
  q_new = q_guess.copy()
  # Newton-Raphson iterations
  iter_count = 0
  error = tol_dq * 10
  flag = 1 # Start with a good simulation

  while error > tol_dq:
    # Compute elastic forces
    # There is none
    # Equations of motion
    f = m / dt * ( (q_new - q_old) / dt - u_old ) - W
    # Jacobian
    J = mMat / dt**2.0
    # We only take the free part of f and J
    f_free = f[free_index]
    J_free = J[np.ix_(free_index, free_index)]

    # Solve for dq
    dq_free = np.linalg.solve(J_free, f_free)

    # Update q_new
    q_new[free_index] -= dq_free

    # Calculate error using the change in position (not force, but using force is OK too)
    error = np.linalg.norm(dq_free)
    iter_count += 1

    if iter_count > maximum_iter:
      flag = -1
      break

  reactionForce = f # f is zero (or close to it) for free DOFs. Non-zero at fixed DOFs.
  # The non-zero forces on fixed DOFs are my reaction forces.
  return q_new, flag, reactionForce

"""# Main"""

# Inputs
nv = 2
ndof = 2 * nv # I am working in 2D

# Time step
dt = 1e-4 # Play around with it to see it's aritificial effect on contact
maximum_iter = 100
totalTime = 5
tol_dq = 1e-6 # Small length value

# Nodes
nodes = np.zeros((nv,2))
nodes[0,0] = 0 # x0 [meter]
nodes[0,1] = 1.0 # y0 [meter]
nodes[1,0] = 0.1 # x0
nodes[1,1] = 1.0 # y0

# Mass
mVector = np.zeros(ndof)
mVector[0:2] = 0.001 # kg
mVector[2:4] = 0.002 # kg

mMat = np.diag(mVector) # Matrix corresponding to the mass vector that will be used to compute J

# Weight
W = np.zeros(ndof)
g = np.array([0, -9.8]) # m/s^2
for c in range(nv):
  W[2*c:2*c+2] = mVector[2*c:2*c+2] * g

# Initial conditions
q0 = np.zeros(ndof)
for c in range(nv):
  q0[2*c] = nodes[c,0]
  q0[2*c+1] = nodes[c,1]

q = q0.copy()
# Velocity
u = np.zeros(ndof)
u[0:2] = [-0.005, 0]
u[2:4] = [0.005, 0]

# Fixed and free DOFs
isFixed = np.zeros(nv) # If c-th element of isFixed is 1, then node c is fixed by the boundary (y direction in this example)
free_index = getFreeIndex(isFixed)

Nsteps = round(totalTime / dt) # number of time steps
ctime = 0 # current time
all_pos = np.zeros((Nsteps, ndof))
all_pos[0,:] = q0

for timeStep in range(1, Nsteps): # Loop over time steps
  print('t = %f\n' % ctime)
  # Predictor step
  q_guess = q0.copy()
  q, error, reactionForce = objfun(q_guess, q0, u, dt, tol_dq, maximum_iter, mVector, mMat, W, free_index)
  # error handling should be done

  # Check if corrector is necessary
  needCorrector = False
  for c in range(nv): # Loop over each node and check for two conditions
    # Condition 1: if the y coordinate is below 0
    if isFixed[c] == 0 and q[2*c+1] < 0:
      isFixed[c] = 1
      q_guess[2*c+1] = 0.0
      needCorrector = True
      break
    # Condition 2: if node is fixed and has negative reaction force
    elif isFixed[c] == 1 and reactionForce[2*c+1] < 0.0:
      isFixed[c] = 0 # Free that node
      q_guess[2*c+1] = 0.0
      needCorrector = True
      break

  # Corrector step
  if needCorrector == True:
    free_index = getFreeIndex(isFixed)
    q, error, reactionForce = objfun(q_guess, q, u, dt, tol_dq, maximum_iter, mVector, mMat, W, free_index)
    # error handling should be done

  u = (q - q0) / dt # update velocity
  q0 = q.copy() # update old position

  all_pos[timeStep,:] = q0 # Save the positions
  ctime += dt # Update the current time

# Plot
plt.figure(1)
plt.plot(all_pos[:,0], all_pos[:,1], 'ro', label='Node 1') # x,y plot for the first node
plt.plot(all_pos[:,2], all_pos[:,3], 'go', label='Node 2') # x,y plot for the first node
plt.xlabel('x [m]')
plt.ylabel('y [m]')
# plt.axis('equal')
plt.xlim([-0.25, 0.25])
plt.legend()
plt.show()
