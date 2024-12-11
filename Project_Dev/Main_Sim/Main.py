import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation

# Helper Functions for MMM
import MMM_functions as MMMadj
from HelperFunctions.BendingFun import getFbP1
from HelperFunctions.StrechingFun import getFsP1

def simloop(q_old, u_old, dt, mass, EI, EA, deltaL, force, tol, mat, nv):
    """
    Simulation loop for simulating motion with contact constraints.
    """
    # Initialize variables
    Nsteps = round(totalTime / dt)  # Number of time steps
    ctime = 0  # Current time

    # Updated to store positions for all nodes
    all_x_pos = np.zeros((nv, Nsteps))  # x-positions for all nodes
    all_y_pos = np.zeros((nv, Nsteps))  # y-positions for all nodes
    all_rf = np.zeros(Nsteps)
    all_zvec = np.zeros(Nsteps)
    all_u = np.zeros((nv, Nsteps))  # Velocities for all nodes
    r_force = np.zeros(3 * nv)
    s_mat = np.eye(3 * nv)
    z_vec = np.zeros(3 * nv)

    # Initial conditions
    q0 = q_old
    q = q0
    u = u_old
    all_x_pos[:, 0] = q0[::3]  # x-coordinates of all nodes
    all_y_pos[:, 0] = q0[1::3]  # y-coordinates of all nodes

    # Simulation Parameters
    dt_def = dt  # Default time step
    close_d = 1e-5  # Proximity threshold for collision detection

    # Main Time Stepping Loop
    for timeStep in range(1, Nsteps):
        print(f"Time = {ctime:.6f}, Node 1 x-Position = {q0[0]:.6f}, Node 1 y-Position = {q0[1]:.6f}")

        # Main simulation step
        r_force, q, flag = MMMadj.MMM_cal(q0, q0, u, dt_def, mass, EI, EA, deltaL, force, tol, s_mat, z_vec)
        con_ind, free_ind, q_con, mat, flag_c, close_flag = MMMadj.test_col(q, r_force, close_d)
        s_mat, z_vec = MMMadj.MMM_Szcalc(mat, con_ind, free_ind, q_con, q0, u, dt_def, mass, force)

        # Update position and velocity
        u = (q - q0) / dt_def
        q0 = q.copy()

        # Save results for this time step
        all_rf[timeStep] = r_force[1]
        all_zvec[timeStep] = z_vec[1]
        all_x_pos[:, timeStep] = q0[::3]  # x-coordinates of all nodes
        all_y_pos[:, timeStep] = q0[1::3]  # y-coordinates of all nodes
        all_u[:, timeStep] = u[1::3]  # y-velocities of all nodes

        # Update current time
        ctime += dt

    return all_x_pos, all_y_pos, all_u, all_rf, all_zvec, u

def animate_simulation(all_x_pos, all_y_pos, totalTime, Nsteps, nv):
    """
    Animate the simulation results
    """
    # Define fruit obstacle
    radius = 0.05
    center_x = 0.1
    center_y = 0.1
    apple_y = np.linspace(center_y - radius, center_y + radius, 1000)
    circle = np.zeros(len(apple_y))
    for i in range(len(apple_y)):
        y = apple_y[i]
        circle[i] = MMMadj.left_circle(y, radius, center_x, center_y)

    # Plots
    fig, ax = plt.subplots()
    ax.set_xlim(-0.05, 0.225)
    ax.set_ylim(np.min(all_y_pos) - 0.1, np.max(all_y_pos) + 0.1)
    ax.set_xlabel('x Position [m]')
    ax.set_ylabel('y Position [m]')
    ax.set_title('MMM Bouncing Simulation')
    ax.grid(True)
    ax.plot(circle, apple_y, 'r-')
    ax.plot((2*center_x)-circle, apple_y, 'r-') # mirror to create right side of apple

    # Dynamic object positions connected by a line
    line, = ax.plot([], [], 'bo-', label='Nodes')  # Line connecting original nodes
    scatter_mirrored, = ax.plot([], [], 'bo-', label='Mirrored Nodes')  # Scatter plot for mirrored nodes

    def init():
        # Initialize the line and scatter plot as empty
        line.set_data([], [])
        scatter_mirrored.set_data([], [])
        return line, scatter_mirrored

    def update(frame):
        # Update the line connecting all nodes
        x_data = all_x_pos[:, frame]
        y_data = all_y_pos[:, frame]
        
        # Mirroring x data about the center_x
        mirrored_x_data = 2 * center_x - x_data
        
        # Plot the line for the original beam (connecting nodes)
        line.set_data(x_data, y_data)
        
        # Plot the mirrored positions (without connecting with a line)
        scatter_mirrored.set_data(mirrored_x_data, y_data)
        
        return line, scatter_mirrored

    # Skip every 500 frames for speed
    anim = FuncAnimation(fig, update, frames=range(0, Nsteps, 20), init_func=init, blit=True, interval=10)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Simulation Parameters
    nv = 5  # number of nodes
    dt = 1e-5
    totalTime = 0.1
    tol_dq = 1e-6

    # Problem setup
    RodLength = 0.1
    r0 = 1e-3 # cross-sectional radius
    deltaL = RodLength / (nv - 1) # discrete length
    ndof = 3 * nv
    Y = 200e9 # Young's modulus
    rho = 7000 # Density
    mass = 0.01 # mass of each nodes
    EI = Y * np.pi * r0**4 / 4
    EA = Y * np.pi * r0**2

    # Geometry of gripper
    q0 = np.zeros(ndof)
    q0[0::3] = 0  # Set all nodes to same x position
    for i in range(nv): # Arrange nodes y position
        q0[i*3 + 1] = 0.05 + (deltaL*i)
    q = q0.copy()

    # Initial conditions
    u = np.zeros(ndof)
    u[0::3] = 1  # Same x velocity
    mat = np.zeros((nv, 2, 3))
    q_con = np.zeros(ndof)

    # Forces
    F = np.zeros(ndof) # Using force of all 0 to ignore gravity (gripper shouldn't fall)
    W = np.zeros(ndof)
    g = np.array([0, -9.8, 0])
    for k in range(nv):
        W[3 * k:3 * k + 3] = mass * g

    # Run simulation
    all_x_pos, all_y_pos, all_u, all_rf, all_zvec, u= simloop(q0, u, dt, mass, EI, EA, deltaL, F, tol_dq, mat, nv)
    animate_simulation(all_x_pos, all_y_pos, totalTime, len(all_x_pos[0]), nv)
