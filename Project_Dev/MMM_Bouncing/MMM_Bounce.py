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
    all_x_pos = np.zeros(Nsteps)
    all_y_pos = np.zeros(Nsteps)
    ground_func = np.zeros(Nsteps)
    all_rf = np.zeros(Nsteps)
    all_zvec = np.zeros(Nsteps)
    all_u = np.zeros(Nsteps)
    r_force = np.zeros(3 * nv)
    s_mat = np.eye(3 * nv)
    z_vec = np.zeros(3 * nv)

    # Initial conditions
    q0 = q_old
    q = q0
    u = u_old
    all_x_pos[0] = q0[0]
    all_y_pos[0] = q0[1]

    # Simulation Parameters
    dt_def = dt  # Default time step
    close_d = 1e-5  # Proximity threshold for collision detection

    # Main Time Stepping Loop
    for timeStep in range(1, Nsteps):
        print(f"Time = {ctime:.6f}, x-Position = {q0[0]:.6f}, y-Position = {q0[1]:.6f}, Velocity = {u[1]:.6f}")

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
        ground_func[timeStep] = MMMadj.ground_surface(q[0])
        all_x_pos[timeStep] = q0[0]
        all_y_pos[timeStep] = q0[1]
        all_u[timeStep] = u[1]

        # Update current time
        ctime += dt

    return all_x_pos, all_y_pos, all_u, all_rf, all_zvec, u, ground_func

def plotting(all_x_pos, all_y_pos, all_u, all_rf, all_zvec, totalTime, Nsteps, ground_func):
    """
    Plot simulation results for position, velocity, and forces.
    """
    t = np.linspace(0, totalTime, len(all_y_pos))

    plt.figure(1)
    plt.plot(all_x_pos, all_y_pos, 'r-', label='Position')
    plt.plot(all_x_pos, ground_func, 'b-', label='Ground')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('Vertical Displacement of Node 1')
    plt.legend()
    plt.savefig('VerticalDisplacementNode1.png')

    plt.figure(2)
    plt.plot(t, all_u, 'b-', label='Velocity')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.title('Vertical Velocity of Node 1')
    plt.legend()
    plt.savefig('VerticalVelocityNode1.png')

    plt.show()


def animate_simulation(all_x_pos, all_y_pos, ground_func, totalTime, Nsteps):
    """
    Animate the simulation results with a static ground line and dynamic object position.
    """
    fig, ax = plt.subplots()
    ax.set_xlim(min(all_x_pos), max(all_x_pos))
    ax.set_ylim(min(all_y_pos) - 0.1, max(all_y_pos) + 0.1)
    ax.set_xlabel('x Position [m]')
    ax.set_ylabel('y Position [m]')
    ax.set_title('MMM Bouncing Simulation')
    ax.grid(True)
    ax.plot(all_x_pos, ground_func, 'b-', label='Ground')

    # Dynamic object position
    obj, = ax.plot([], [], 'ro', label='Ball Position')

    def init():
        # Initialize the object position as empty
        obj.set_data([], [])
        return obj,

    def update(frame):
        # Update only the current object position
        obj.set_data(all_x_pos[frame], all_y_pos[frame])
        return obj,

    # Skip every 400 frames for speed
    anim = FuncAnimation(fig, update, frames=range(0, Nsteps, 500), init_func=init, blit=True, interval=10)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Simulation Parameters
    dt = 1e-5
    totalTime = 5
    tol_dq = 1e-6

    # Problem setup
    RodLength = 0.10
    r0 = 1e-3
    nv = 1  # Number of nodes
    deltaL = 0.01
    ndof = 3 * nv
    Y = 1e9
    rho = 7000
    mass = 0.01
    EI = Y * np.pi * r0**4 / 4
    EA = Y * np.pi * r0**2

    # Forces
    W = np.zeros(ndof)
    g = np.array([0, -9.8, 0])
    for k in range(nv):
        W[3 * k:3 * k + 3] = mass * g

    # Initial Conditions
    q0 = np.zeros(ndof)
    q0[1] = 1
    q = q0.copy()
    u = np.zeros(ndof)
    u[0] = 1
    mat = np.zeros((nv, 2, 3))
    q_con = np.zeros(ndof)

    # Run simulation
    all_x_pos, all_y_pos, all_u, all_rf, all_zvec, u, ground_func = simloop(q0, u, dt, mass, EI, EA, deltaL, W, tol_dq, mat, nv)
    # plotting(all_x_pos, all_y_pos, all_u, all_rf, all_zvec, totalTime, int(totalTime / dt), ground_func)
    animate_simulation(all_x_pos, all_y_pos, ground_func, totalTime, len(all_x_pos))