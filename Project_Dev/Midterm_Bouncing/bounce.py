import numpy as np
import matplotlib.pyplot as plt
from helper_functions import getFreeIndex
import time
from matplotlib.animation import FuncAnimation

class Bouncing:

    ### INITIALIZE PARAMETERS
    def __init__(self, nv, dt, total_time, tol_dq, maximum_iter, elasticity):
        # Simulation parameters
        self.nv = nv  # Number of nodes
        self.ndof = 2 * nv  # Degrees of freedom (2 per node: x and y)
        self.dt = dt  # Time step
        self.total_time = total_time
        self.tol_dq = tol_dq  # Tolerance for convergence
        self.maximum_iter = maximum_iter  # Max iterations for solving
        self.elasticity = elasticity  # Coefficient of restitution (bounce factor)

        # Ball properties
        self.m = np.zeros(self.ndof)
        self.m = np.full(self.ndof, 0.1)  # Set all masses to 0.1 (or any other value)
        self.mMat = np.diag(self.m)  # Mass matrix

        # Gravitational force
        self.W = np.zeros(self.ndof)
        g = np.array([0, -9.8])  # Gravity
        for c in range(nv):
            self.W[2 * c:2 * c + 2] = self.m[2 * c:2 * c + 2] * g

        # Initial conditions
        self.q0 = np.array([0.0, 1], dtype=float)  # Initial positions
        self.u = np.array([0.1, 0], dtype=float)  # Initial velocities
        self.q = self.q0.copy()  # Current positions

        # Fixed and free DOFs
        self.isFixed = np.zeros(nv)  # All DOFs free initially
        self.free_index = getFreeIndex(self.isFixed)  # Get free DOF indices

        # Time variables
        self.Nsteps = round(total_time / dt)
        self.ctime = 0.0  # Current simulation time
        self.all_pos = np.zeros((self.Nsteps, self.ndof))  # Store positions
        self.all_pos[0, :] = self.q0

    ### OBJECTIVE FUNCTION
    def objfun(self, q_guess, q_old, u_old):
        q_new = q_guess.copy()
        iter_count = 0
        error = self.tol_dq * 10
        flag = 1  # Simulation success flag

        while error > self.tol_dq:
            # Forces and Jacobian
            f = self.m / self.dt * ((q_new - q_old) / self.dt - u_old) - self.W
            J = self.mMat / self.dt**2.0
            f_free = f[self.free_index]
            J_free = J[np.ix_(self.free_index, self.free_index)]

            # Solve for displacement
            dq_free = np.linalg.solve(J_free, f_free)
            q_new[self.free_index] -= dq_free
            error = np.linalg.norm(dq_free)
            iter_count += 1

            if iter_count > self.maximum_iter:
                flag = -1  # Max iterations exceeded
                break

        reactionForce = f  # Reaction forces
        return q_new, flag, reactionForce

    ### MAIN TIME STEPPING LOOP
    def main_loop(self):
        for timeStep in range(1, self.Nsteps):
            print(f't = {self.ctime:.4f} s')

            # Predictor step
            q_guess = self.q0.copy()
            self.q, error, reactionForce = self.objfun(q_guess, self.q0, self.u)
            
            # Check if corrector is necessary
            needCorrector = False
            bounce = False
            bounce_vels = np.zeros(self.ndof) # array to store bouncing velocities
            for c in range(self.nv): # Loop over each node and check for two conditions

                # Condition 1: if the y coordinate is below 0
                if self.isFixed[c] == 0 and self.q[2*c+1] < 0:
                    self.isFixed[c] = 1
                    q_guess[2*c+1] = 0.0
                    bounce_vels[2*c+1] = -self.elasticity * self.u[2*c+1] # reverse velocity
                    print(f"Node {c} hit the ground! Velocity after bounce: {bounce_vels[2*c+1]:.4f} m/s")
                    # time.sleep(1)
                    needCorrector = True
                    bounce = True

                # Condition 2: if the x coordinate exceeds 0.4
                if self.isFixed[c] == 0 and self.q[2 * c] > 0.4:
                    self.isFixed[c] = 1
                    q_guess[2 * c] = 0.4  # Reset x to 0.4
                    bounce_vels[2 * c] = -self.elasticity * self.u[2 * c]  # reverse velocity
                    bounce_vels[2 * c + 1] = self.u[2 * c + 1]  # Keep the current y-velocity
                    print(f"Node {c} hit the right boundary! x velocity after bounce: {bounce_vels[2 * c]:.4f} m/s")
                    # time.sleep(1)
                    needCorrector = True
                    bounce = True

                # Condition 3: if node is fixed and has negative reaction force
                if self.isFixed[c] == 1 and reactionForce[2*c+1] < 0.0:
                    self.isFixed[c] = 0 # Free that node
                    q_guess[2*c+1] = 0.0
                    print("Reaction force in y-direction!")
                    # time.sleep(1)
                    needCorrector = True

                # Condition 4: if node is fixed and has postive reaction force
                if self.isFixed[c] == 1 and reactionForce[2*c] > 0.0:
                    self.isFixed[c] = 0 # Free that node
                    q_guess[2*c] = 0.4
                    print("Reaction force in x-direction!")
                    # time.sleep(1)
                    needCorrector = True

            # Corrector step
            if needCorrector == True:
                self.free_index = getFreeIndex(self.isFixed)
                self.q, error, reactionForce = self.objfun(q_guess, self.q0, self.u)
                # error handling should be done

            self.u = (self.q - self.q0) / self.dt # update velocity

            # Add bounce back after calculating velocity!
            if bounce == True:
                    for c in range(self.ndof):
                        if bounce_vels[c] != 0:
                            self.u[c] = bounce_vels[c]

            self.q0 = self.q.copy()
            self.all_pos[timeStep, :] = self.q0
            self.ctime += self.dt

    ### PLOT RESULTS
    def plot_results(self):
        plt.figure()
        plt.plot(self.all_pos[:, 0], self.all_pos[:, 1], label="Ball position 1")
        #plt.plot(self.all_pos[:, 2], self.all_pos[:, 3], label="Ball position 2")
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('Ball Bouncing Simulation')
        plt.legend()
        plt.grid(True)
        plt.show()

    ### VISUALIZE MOTION
    def animate_results(self):
        fig, ax = plt.subplots()
        ax.set_xlim(-0.1, 0.5)  # Set x-axis limits
        ax.set_ylim(-0.1, 1.5)  # Set y-axis limits
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Ball Bouncing Simulation')
        ax.grid(True)

        # Plot the ground (y=0) and wall (x=0.4)
        ax.axhline(y=0, color='red', linestyle='-', linewidth=2)
        ax.axvline(x=0.4, color='red', linestyle='-', linewidth=2)

        # Initialize the ball as a scatter plot
        ball, = ax.plot([], [], 'bo', markersize=10)

        # Update function for each frame
        def update(frame):
            # Ensure that x and y values are passed as tuples or lists
            ball.set_data([self.all_pos[frame, 0]], [self.all_pos[frame, 1]])
            return ball,

        # Create animation
        ani = FuncAnimation(fig, update, frames=range(0, len(self.all_pos), 10), blit=True, interval=0.01)
        plt.show()


### RUN SIMULATION
if __name__ == "__main__":
    simulation = Bouncing(
        nv = 1, # Number of nodes
        dt = 1e-4, # Time step
        total_time = 5, # Total time
        tol_dq = 1e-6, # Tolerance for convergence
        maximum_iter = 100, # Maximum number of iterations
        elasticity = 0.9 # Coefficient of restitution
    )
    simulation.main_loop()
    simulation.plot_results()
    # simulation.animate_results()
