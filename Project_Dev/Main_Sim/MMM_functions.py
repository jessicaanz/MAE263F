import numpy as np
import time
from HelperFunctions.BendingFun import getFbP2
from HelperFunctions.StrechingFun import getFsP2
from HelperFunctions.OpFun import crossMat

def get_matind(nodes):
    """
    This function takes in node numbers and outputs the corresponding indices for the S matrix.
    Assumes nodes are indexed starting from 1.
    """
    ind = np.zeros(3 * len(nodes))  # Initialize an array to store the indices
    st_ind = 3 * (nodes - np.ones(len(nodes)))  # Calculate starting indices for each node

    # Fill in the indices for the S matrix
    for ii in range(len(nodes)):
        ind[3*ii] = int(st_ind[ii])
        ind[3*ii + 1] = int(st_ind[ii] + 1)
        ind[3*ii + 2] = int(st_ind[ii] + 2)

    ind = ind.astype(int)  # Convert to integer type for index usage
    return ind


def getNnum(ind):
    """
    This function takes in node indices and returns the corresponding node numbers for 3D space.
    Assumes nodes are indexed starting from 1.
    """
    ind = np.array(ind)
    Nnum = np.zeros(int(len(ind) / 3))  # Initialize array for node numbers
    st_nodes = np.ones(len(ind)) + ((ind / 3).astype(int))  # Calculate the node numbers

    # Extract node numbers from the indices
    for jj in range(int(len(ind) / 3)):
        Nnum[jj] = st_nodes[3*jj]

    Nnum = Nnum.astype(int)
    return Nnum


def MMM_eq(q_new, q_old, u_old, dt, mass, force, S_mat, z_vec):
    """
    This function calculates the mass matrix equation for force adjustment due to collision presence.
    """
    f_n = (1 / dt) * ( ((q_new - q_old) / dt) - u_old ) - ((1 / mass) * S_mat @ force) - z_vec
    return f_n


def RF_eq(q_new, q_old, u_old, dt, mass, force):
    """
    This function calculates the reaction force based on the difference between new and old positions and velocities.
    """
    r_force = (mass / dt) * ( ((q_new - q_old) / dt) - u_old ) - force
    return r_force


def MMM_zcalc(q_con, q_old, u_old, dt, mass, force, S_mat):
    """
    This function calculates the z vector adjustment based on the constrained nodes.
    """
    z_vec = (1 / dt) * ( ((q_con - q_old) / dt) - u_old ) - ((1 / mass) * S_mat @ force)
    return z_vec


def MMM_Szcalc(mat, con_ind, free_ind, q_con, q_old, u_old, dt, mass, force):
    """
    This function calculates the updated S matrix and z vector for constrained and free nodes.
    It handles the constraints and adjustments to the S matrix and z vector based on node status.
    """
    ndof = len(q_old)  # Degree of freedom from all nodes
    s_mat = np.eye(ndof)  # Initialize the S matrix as an identity matrix
    z_vec = np.zeros(ndof)  # Initialize the z vector

    # Update S matrix and z vector for constrained nodes
    if len(con_ind) > 1:
        Nnum_con = getNnum(con_ind)
        for ii in range(int(len(Nnum_con))):
            cur_node = Nnum_con[ii] - 1
            S_n = np.eye(3) - np.outer(mat[cur_node][0], mat[cur_node][0]) - np.outer(mat[cur_node][1], mat[cur_node][1])
            z_n = MMM_zcalc(q_con[(con_ind[3*ii]):(con_ind[3*ii] + 3)], q_old[(con_ind[3*ii]):(con_ind[3*ii] + 3)],
                             u_old[(con_ind[3*ii]):(con_ind[3*ii] + 3)], dt, mass, force[(con_ind[3*ii]):(con_ind[3*ii] + 3)], S_n)
            z_n = np.dot(mat[cur_node][0], z_n) * mat[cur_node][0] + np.dot(mat[cur_node][1], z_n) * mat[cur_node][1]


            s_mat[(con_ind[3*ii]):(con_ind[3*ii] + 3), (con_ind[3*ii]):(con_ind[3*ii] + 3)] = S_n
            z_vec[(con_ind[3*ii]):(con_ind[3*ii] + 3)] = z_n

    # Update S matrix and z vector for free nodes
    if len(free_ind) > 1:
        for kk in range(int(len(free_ind) / 3)):
            S_n = np.eye(3)
            z_n = np.zeros(3)

            s_mat[(free_ind[3*kk]):(free_ind[3*kk] + 3), (free_ind[3*kk]):(free_ind[3*kk] + 3)] = S_n
            z_vec[(free_ind[3*kk]):(free_ind[3*kk] + 3)] = z_n

    return s_mat, z_vec


def left_circle(y, radius, center_x, center_y):
    """
    Solves for the x values on the left side of the circle for a given y value.
    """
    distance_from_center = np.abs(y - center_y)
    if distance_from_center <= radius:
        x = center_x - np.sqrt(radius**2 - (y - center_y)**2)  # Left side x values
        return x
    else:
        return None

def left_circle_normal(y, radius, center_x, center_y):
    """
    Computes the normal vector to the left side of the circle at a given y value.
    """
    x_value = left_circle(y, radius, center_x, center_y)
    if x_value is not None:
        normal = np.array([x_value - center_x, y - center_y])
        magnitude = np.linalg.norm(normal)
        if magnitude != 0:
            return normal / magnitude
    else:
        return None


def test_col(q_test, r_force, close_d):
    """
    This function tests for collision by checking the reaction force and adjusting the node positions
    based on proximity to a surface or other constraints.
    """
    q_con = q_test  # The current test positions for the nodes.
    con_ind = np.zeros(len(q_test), dtype=int)  # Array to store indices of constrained nodes.
    free_ind = np.zeros(len(q_test), dtype=int)  # Array to store indices of free nodes.
    mat = np.zeros((int(len(q_test) / 3), 2, 3))  # Matrix to store the constraint conditions for each node.
    flag = 0  # Flag to indicate if a node is constrained.
    close_flag = 0  # Flag to indicate if any node is within a threshold for collision.
    radius = 0.05
    center_x = 0.1
    center_y = 0.1

    # Loop through each node to check for collision.
    for ii in range(int(len(q_test) / 3)):
        # Define ground and normal vector
        node_x = q_test[ii*3]
        node_y = q_test[ii*3 + 1]

        # CIRCLE COLLISION
        circle_x = left_circle(node_y, radius, center_x, center_y)
        circle_x_norm = left_circle_normal(node_y, radius, center_x, center_y)
        if circle_x != None:
            if node_x > circle_x: # Detects collision once node passes circle!
                q_con[3 * ii] = circle_x  # Set position to circle
                mat[ii] = np.array([[circle_x_norm[0], circle_x_norm[1], 0], [0, 0, 0]])
                con_ind[ii] = ii + 1
                flag = 1
                print("Collision with circle!")

        else:  # No collision or constraint
            free_ind[ii] = ii + 1
            mat[ii] = np.array([[0, 0, 0], [0, 0, 0]])

    # Update indices for constrained and free nodes
    con_ind = get_matind(con_ind[con_ind != 0]) if len(con_ind[con_ind != 0]) >= 1 else np.array([-1])
    free_ind = get_matind(free_ind[free_ind != 0]) if len(free_ind[free_ind != 0]) >= 1 else np.array([-1])

    return con_ind, free_ind, q_con, mat, flag, close_flag



def MMM_cal(q_guess, q_old, u_old, dt, mass, EI, EA, deltaL, force, tol, S_mat, z_vec):
    """
    This function calculates the placement of the node assuming no collision
    """
    q_new = q_guess.copy()
    iter_count = 0
    max_itt = 500
    error = tol * 10
    flag = 1 # Start with a good simulation

    while error > tol:
        Fb, Jb = getFbP2(q_new, EI, deltaL)
        Fs, Js = getFsP2(q_new, EA, deltaL)

        # Calculation of correction step from mass matrix
        f_n = MMM_eq(q_new, q_old, u_old, dt, mass, (force+Fs+Fb), S_mat, z_vec)
        J_n = np.eye(len(q_old)) / dt**2.0 + S_mat @ (Js + Jb)

        # Solve for dq
        dq = np.linalg.solve(J_n, f_n)
        
        # Update q_new
        q_new = q_new - dq

        # Calculate error using the change in position (not force, but using force is OK too)
        error = np.linalg.norm(dq)

        iter_count += 1
        if iter_count > max_itt:
            flag = -1
            break

    # Calculation of reaction force (done within iteration)
    r_force = RF_eq(q_new, q_old, u_old, dt, mass, force)

    return r_force, q_new, flag