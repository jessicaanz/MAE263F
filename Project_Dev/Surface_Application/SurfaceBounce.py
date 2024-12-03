import numpy as np
import matplotlib.pyplot as plt

# Helper Functions for MMM
import MMM_functions as MMMadj
from HelperFunctions.BendingFun import getFbP1
from HelperFunctions.StrechingFun import getFsP1
import time

def simloop(q_guess, q_old, u_old, dt, mass, EI, EA, deltaL, force, tol, mat, nv):
    """
    Simulation loop for simulating motion with contact constraints.
    """
    Nsteps = round(totalTime / dt)  # Number of time steps
    ctime = 0  # Current time
    all_pos = np.zeros(Nsteps)
    all_rf = np.zeros(Nsteps)
    all_zvec = np.zeros(Nsteps)
    all_u = np.zeros(Nsteps)

    q0 = q_old
    q = q0
    u = u_old
    all_pos[0] = q0[1]

    dt_def = dt  # Default time step
    end_flag = 0
    close_d = 1e-5  # Proximity threshold for collision detection

    r_force = np.zeros(3 * nv)
    s_mat = np.eye(3 * nv)
    z_vec = np.zeros(3 * nv)

    for timeStep in range(1, Nsteps):  # Loop over time steps
        print(f"Time = {ctime:.6f}, Position = {q0[1]:.6f}, Velocity = {u[1]:.6f}")

        # Main simulation step
        r_force, q, flag = MMMadj.MMM_cal(q0, q0, u, dt_def, mass, EI, EA, deltaL, force, tol, s_mat, z_vec)
        con_ind, free_ind, q_con, mat, flag_c, close_flag = MMMadj.test_col(q, u, r_force, close_d, 0, ctime)

        s_mat, z_vec = MMMadj.MMM_Szcalc(mat, con_ind, free_ind, q_con, q0, u, dt_def, mass, force)

        u = (q - q0) / dt_def  # Update velocity

        q0 = q.copy()  # Update position

        # Save results for this time step
        all_rf[timeStep] = r_force[1]
        all_zvec[timeStep] = z_vec[1]
        all_pos[timeStep] = q0[1]
        all_u[timeStep] = u[1]

        if end_flag == 1:  # Early termination for low oscillations
            print("Low oscillations, ending simulation")
            all_rf[timeStep:] = 0
            all_zvec[timeStep:] = 0
            all_pos[timeStep:] = 0
            all_u[timeStep:] = 0
            break

        ctime += dt  # Update current time

    return all_pos, all_u, all_rf, all_zvec, u

def plotting(all_pos, all_u, all_rf, all_zvec, totalTime, Nsteps):
    """
    Plot simulation results for position, velocity, and forces.
    """
    t = np.linspace(0, totalTime, len(all_pos))

    plt.figure(1)
    plt.plot(t, all_pos, 'r-', label='Position')
    plt.plot(t, MMMadj.ground_surface(t), 'b-', label='Ground')
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

if __name__ == '__main__':
    # Problem setup
    RodLength = 0.10
    r0 = 1e-3
    nv = 1  # Number of nodes
    deltaL = 0.01
    ndof = 3 * nv
    q0 = np.zeros(ndof)
    q0[1] = 1
    q = q0.copy()

    dt = 1e-5
    totalTime = 2
    tol_dq = 1e-6
    Y = 1e9
    rho = 7000
    mass = 0.01
    EI = Y * np.pi * r0**4 / 4
    EA = Y * np.pi * r0**2

    W = np.zeros(ndof)
    g = np.array([0, -9.8, 0])
    for k in range(nv):
        W[3 * k:3 * k + 3] = mass * g

    u = np.zeros(ndof)
    mat = np.zeros((nv, 2, 3))
    q_con = np.zeros(ndof)

    # Run simulation
    all_pos, all_u, all_rf, all_zvec, u = simloop(q0, q0, u, dt, mass, EI, EA, deltaL, W, tol_dq, mat, nv)
    plotting(all_pos, all_u, all_rf, all_zvec, totalTime, int(totalTime / dt))