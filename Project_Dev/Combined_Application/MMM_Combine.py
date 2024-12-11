import numpy as np

from HelperFunctions.BendingFun import getFbP2
from HelperFunctions.StrechingFun import getFsP2
from HelperFunctions.OpFun import crossMat

def get_matind(nodes):
    # takes in node numbers and outputs corresponding indices for S matrix
    # ASSUMING nodes start at 1
    ind = np.zeros(3*len(nodes))
    # print(ind)
    st_ind = 3 * (nodes - np.ones(len(nodes)))
    # print(st_ind)

    for ii in range(len(nodes)):
        ind[3*ii] = int(st_ind[ii])
        ind[3*ii + 1] = int(st_ind[ii] + 1)
        ind[3*ii + 2] = int(st_ind[ii] + 2)

    ind = ind.astype(int)

    return ind

def getNnum(ind):
    # takes in node indices and gives the corresponging number for 3D
    # ASSUMING nodes start at 1
    ind = np.array(ind)
    Nnum = np.zeros(int(len(ind)/3))
    st_nodes = np.ones(len(ind)) + ((ind/3).astype(int))

    for jj in range(int(len(ind)/3)):
        Nnum[jj] = st_nodes[3*jj]

    Nnum = Nnum.astype(int)

    return Nnum


def MMM_eq(q_new, q_old, u_old, dt, mass, force, S_mat, z_vec):
    # Iterates over nodes and adjusts according to collision presence

    f_n = (1 / dt) * ( ((q_new - q_old) / dt) - u_old ) - ((1/mass)* S_mat @ force) - z_vec

    return f_n


def RF_eq(q_new, q_old, u_old, dt, mass, force, mat):
    # Calculates the reaction force on each node in the constrainted direction, p
    r_force = np.zeros(int(len(q_old)))

    for ii in range(int(len(q_old)/3)):

        r_fc = (mass / dt) * ( ((q_new[3*ii:3*ii+3] - q_old[3*ii:3*ii+3]) / dt) - u_old[3*ii:3*ii+3]) - (force[3*ii:3*ii+3])
        
        # Project the vector onto the contrained direction
        r_force[3*ii:3*ii+3] = (np.dot(mat[ii][0], r_fc)) * np.array(mat[ii][0])
    
    return r_force


def MMM_zcalc(q_con, q_old, u_old, dt, mass, force, S_mat):
    # Iterates over nodes and adjusts according to collision presence
    e_factor = 1
    z_vec = e_factor*((1 / dt) * ( ((q_con - q_old) / dt) - u_old ) - ((1/mass)* S_mat @ force))

    return z_vec


def MMM_Szcalc(mat, con_ind, free_ind, q_con, q_old, u_old, dt, mass, force):
    # mat is a 3D array containing vectors p and q for a 2D case
    # con_node -> indicates nodes that were free to be constricted by this step
    # free_node -> indicates nodes that were constrained to be freed in this step

    ndof = len(q_old)                               # Degree of freedom from all nodes
    s_mat = np.eye(ndof)                            # Initialize the S matrix
    z_vec = np.zeros(ndof)                          # Initialize the z vector

    # Calculates the new S and z terms to correct in the objective function
    # If any nodes need to be constrained
    if len(con_ind) > 1:
        # Specify the node corresponding to the indices
        Nnum_con = getNnum(con_ind)
        for ii in range(int(len(Nnum_con))):
            cur_node = Nnum_con[ii] - 1
            S_n = np.eye(3) - np.outer(mat[cur_node][0], mat[cur_node][0]) - np.outer(mat[cur_node][1], mat[cur_node][1])
            # print(cur_node)
            # print(mat[cur_node][0])
            # print(S_n)
            z_n = MMM_zcalc(q_con[(con_ind[3*ii]):(con_ind[3*ii] + 3)], q_old[(con_ind[3*ii]):(con_ind[3*ii] + 3)], \
                            u_old[(con_ind[3*ii]):(con_ind[3*ii] + 3)], dt, mass, force[(con_ind[3*ii]):(con_ind[3*ii] + 3)], S_n)
            #print(z_n)
            z_n = ( np.dot(mat[cur_node][0], z_n) ) * np.array(mat[cur_node][0])

            s_mat[(con_ind[3*ii]):(con_ind[3*ii] + 3), (con_ind[3*ii]):(con_ind[3*ii] + 3)] = S_n
            z_vec[(con_ind[3*ii]):(con_ind[3*ii] + 3)] = z_n

    # If any nodes need to be freed
    if len(free_ind) > 1:
        for kk in range(int(len(free_ind)/3)):
            S_n = np.eye(3)
            z_n = np.zeros(3)

            s_mat[(free_ind[3*kk]):(free_ind[3*kk] + 3), (free_ind[3*kk]):(free_ind[3*kk] + 3)] = S_n
            z_vec[(free_ind[3*kk]):(free_ind[3*kk] + 3)] = z_n

    return s_mat, z_vec

import numpy as np

def right_circle(y, radius, center_x, center_y):
    """
    Solves for the x values on the right side of the circle for a given y value.
    """
    distance_from_center = np.abs(y - center_y)
    if distance_from_center <= radius:
        x = center_x + np.sqrt(radius**2 - (y - center_y)**2)  # Right side x values
        return x
    else:
        return None

def right_circle_normal(y, radius, center_x, center_y):
    """
    Computes the normal vector to the right side of the circle at a given y value.
    """
    x_value = right_circle(y, radius, center_x, center_y)
    if x_value is not None:
        normal = np.array([x_value - center_x, y - center_y])
        magnitude = np.linalg.norm(normal)
        if magnitude != 0:
            return normal / magnitude
    else:
        return None



def test_col(q_test, r_force, close_d, close_off):
    # free_ind indicates nodes currently free that are being tested
    # con_ind indicates nodes currently constrained that are being tested
    q_con = q_test
    con_ind = np.zeros(len(q_test))
    con_ind = con_ind.astype(int)
    free_ind = np.zeros(len(q_test))
    free_ind = free_ind.astype(int)
    mat = np.zeros((int(len(q_test)/3),2,3))
    flag = 0
    close_flag = 0
    p = np.array([1,0,0])
    radius = 0.05
    center_x = -0.05
    center_y = -0.05

    for ii in range(int(len(q_test)/3)):
        # Define ground and normal vector
        node_x = q_test[ii*3]
        node_y = q_test[ii*3 + 1]

        # CIRCLE COLLISION
        circle_x = right_circle(node_y, radius, center_x, center_y)
        circle_x_norm = right_circle_normal(node_y, radius, center_x, center_y)
        if circle_x != None:
            if node_x < circle_x: # Detects collision once node passes circle!
                q_con[3 * ii] = circle_x  # Set position to circle
                mat[ii] = np.array([[circle_x_norm[0], circle_x_norm[1], 0], [0, 0, 0]])
                con_ind[ii] = ii + 1
                flag = 1
                print("Collision with circle!")

        else:  # No collision or constraint
            free_ind[ii] = ii + 1
            mat[ii] = np.array([[0, 0, 0], [0, 0, 0]])
        # # If the reaction force vector is zero
        # if np.round(np.sum(r_force)) == 0:
        #     proj_vec = np.zeros(3)
        # else:
        #     unit_r = (1/np.linalg.norm(r_force[(3*ii):(3*ii + 3)])) * r_force[(3*ii):(3*ii + 3)]
        #     proj_vec = ( np.dot(p, unit_r) ) * p

        # #print(proj_vec[1])
        # #print(q_test[3*ii])
        # if q_test[3*ii] < 0 and q_test[3*ii + 1] < -0.02:
            
        #     q_con[3*ii] = 0
        #     mat[ii] = np.array([p,[0,0,0]])
        #     #print(mat)
        #     con_ind[ii] = ii + 1
        #     #print("Below surface")
        #     flag = 1
        # elif q_test[3*ii] >= 0 and proj_vec[0] <= 0 and q_test[3*ii + 1] < -0.02:
        #     free_ind[ii] = ii + 1
        #     mat[ii] = np.array([[0,0,0],[0,0,0]])
        #     #print("Negative reaction")
        #     if q_test[3*ii] < (close_d + close_off):
        #         free_ind[ii] = ii + 1
        #         mat[ii] = np.array([[0,0,0],[0,0,0]])
        #         close_flag = 1
        #         print("Close")
        # else:
        #     free_ind[ii] = ii + 1
        #     mat[ii] = np.array([[0,0,0],[0,0,0]])
        #     #print("Other")
            

    #print(free_ind)
    if len(con_ind[con_ind != 0]) >= 1:
        con_ind = get_matind(con_ind[con_ind != 0])
        #print(con_ind)
    else:
        con_ind = np.array([-1])
    if len(free_ind[free_ind != 0]) >= 1:
        # print(free_ind)
        free_ind = get_matind(free_ind[free_ind != 0])
        #print(free_ind)
    else:
        free_ind = np.array([-1])

    #print(mat)
    return con_ind, free_ind, q_con, mat, flag, close_flag


def MMM_cal(q_guess, q_old, u_old, dt, mass, EI, EA, deltaL, force, tol, S_mat, z_vec, mat, fix_ix):
    # Calculates the placement of the node assuming no collision
    # q_con gives enforced displacement values at nodes from collision

    # ndof = 3*nv
    # Simulation parameters
    q_new = q_guess.copy()
    iter_count = 0
    max_itt = 500
    error = tol * 10
    flag = 1 # Start with a good simulation

    while error > tol:
        # print(np.linalg.norm(f_n))
        Fb, Jb = getFbP2(q_new, EI, deltaL)
        Fs, Js = getFsP2(q_new, EA, deltaL)

        Fb[fix_ix] = 0
        Fs[fix_ix] = 0
        Jb[np.ix_(fix_ix, fix_ix)] = 0
        Js[np.ix_(fix_ix, fix_ix)] = 0

        # Calculation of correction step from mass matrix
        # For use with one node
        # f_n = MMM_eq(q_new, q_old, u_old, dt, mass, (force), S_mat, z_vec)
        # J_n = np.eye(len(q_old)) / dt**2.0
        # For use with 2 nodes
        # f_n = MMM_eq(q_new, q_old, u_old, dt, mass, (force+Fs), S_mat, z_vec)
        # J_n = np.eye(len(q_old)) / dt**2.0 + S_mat @ (Js)
        # For use with 3 nodes or more
        f_n = MMM_eq(q_new, q_old, u_old, dt, mass, (force+Fs+Fb), S_mat, z_vec)
        J_n = np.eye(len(q_old)) / dt**2.0 + S_mat @ (Js + Jb)

        # Solve for dq
        dq = np.linalg.solve(J_n, f_n)
        # print(dq)
        # print(q_new)
        
        q_new = q_new - dq                            # Update q_new
        #print(q_new)
        # Calculate error using the change in position (not force, but using force is OK too)
        error = np.linalg.norm(dq)

        # print(iter_count)
        iter_count += 1
        if iter_count > max_itt:
            flag = -1
            break
    #print(iter_count)

    # Calculation of reaction force (done within iteration)
    r_force = RF_eq(q_new, q_old, u_old, dt, mass, force, mat)
    f_in = Fb + Fs

    return r_force, f_in, q_new, flag


if __name__ == '__main__':

    # a = [[1,2,3], [4, 5, 6], [7, 8, 9]]
    # print(a[np.ix_(np.array([1,2]), np.array([1,2]))])
    # print(getNnum(np.array([0,1,2,9,10,11,12,13,14])))

    nv = 1
    # nv = 2
    q_old = np.array([0, 0.00958374, 0])
    #q_old = np.zeros(6)
    q_guess = q_old.copy()
    u_old = np.array([0, -4.40446291, 0])
    #u_old = np.zeros(6)
    dt = 1e-6
    mass = 0.01
    force = mass * np.array([0, -9.81, 0])
    #force = mass * np.array([0, -9.81, 0, 0, -9.81, 0])
    tol = 1e-6
    ob_flag = -1
    mat = [[[0, 1, 0], [0, 0, 0]]]
    #mat = [[[0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]
    con_nodes = [1]
    q_con = np.array([0, 0, 0])
    #q_con = np.array([0, 0.001, 0, 0, 0, 0])
    con_ind = [0,1,2]
    free_ind = [3,4,5]
     

    test_ind = get_matind([1])
    # print(type(test_ind))
    # print(test_ind[0])

    # s_mat = MMM_Scalc(mat, 3, test_ind)
    # # print(mat[0][0])
    # print(s_mat)

    index = np.array(test_ind)
    # print(index)
    # print(q_con[index])

    # print(mat[0][0])
    # print(np.array(mat[0][0]) * np.dot(mat[0][0], [0.5, 0.5, 0.3]))
    
    # S, z = MMM_Szcalc(mat, con_ind, free_ind, q_con, q_old, u_old, dt, mass, force)
    # print(S)
    # print(z)

    EA = 1
    EI = 1
    deltaL = 0.01
    s_mat = np.eye(3)
    z_vec = np.zeros(3)

    r_force, q, flag = MMM_cal(q_old, q_old, u_old, dt, mass, EI, EA, deltaL, force, tol, s_mat, z_vec)
    print(q)
