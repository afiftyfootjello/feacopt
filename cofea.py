import numpy as np
import scipy.integrate as integrate
from . import numerics

A_GRAV = 9.834 # m/s

'''
||||||||||||||||||||||||||||||||
    2D truss processing
||||||||||||||||||||||||||||||||
'''

def truss_rotation(na, nb):
    coordy = nb[1] - na[1]
    coordx = nb[0] - na[0]
    return np.arctan2(coordy, coordx)


def truss_len(na, nb):
    return np.linalg.norm(np.subtract(nb,na))


def axial_element_stiffness(nodes, element, young_mod: float, area: float):

    na = nodes[element[0]]
    nb = nodes[element[1]]

    truss_length = truss_len(na,nb)

    rotation = truss_rotation(na,nb)

    rotation_mat = np.asarray([
        [np.cos(rotation), 0],
        [np.sin(rotation), 0],
        [0, np.cos(rotation)],
        [0, np.sin(rotation)]
    ])

    base_mat = np.asarray(
    [
    [1, -1],
    [-1, 1]
    ]
    )

    rotation_trans = np.transpose(rotation_mat)

    local_stiffness = np.matmul(rotation_mat, base_mat)
    local_stiffness = np.matmul(local_stiffness, rotation_trans)

    try:
        local_stiffness *= young_mod*area/truss_length
    except:
        breakpoint()

    return local_stiffness


def truss_stiffness(nodes, elements, young_mod: float, areas):

    problem_size = len(nodes) * 2

    global_stiffness = np.zeros((problem_size, problem_size))

    for element,area in zip(elements,areas):
        local_stiffness_mat = axial_element_stiffness(nodes, element, young_mod, area)

        na_idx = 2*element[0]
        nb_idx = 2*element[1]

        global_stiffness[na_idx:na_idx+2, na_idx:na_idx+2] += local_stiffness_mat[0:2,0:2]
        global_stiffness[na_idx:na_idx+2, nb_idx:nb_idx+2] += local_stiffness_mat[0:2,2:4]
        global_stiffness[nb_idx:nb_idx+2, na_idx:na_idx+2] += local_stiffness_mat[2:4,0:2]
        global_stiffness[nb_idx:nb_idx+2, nb_idx:nb_idx+2] += local_stiffness_mat[2:4,2:4]

    return global_stiffness


def axial_element_load_vec(nodes, element, mass_dens: float, area: float, grav_angle=-np.pi/2):

    na = nodes[element[0]]
    nb = nodes[element[1]]

    truss_length = truss_len(na,nb)

    grav_vec = A_GRAV*np.asarray([np.cos(grav_angle), np.sin(grav_angle)])

    mass = mass_dens * truss_length * area

    load_vec = np.concatenate([0.5*mass*grav_vec, 0.5*mass*grav_vec])

    return load_vec


def truss_load_vec(nodes, elements, mass_dens: float, areas: float, grav_angle=-np.pi/2):

    problem_size = len(nodes) * 2

    global_load = np.zeros(problem_size)

    for element,area in zip(elements,areas):
        local_load = axial_element_load_vec(nodes, element,mass_dens, area, grav_angle=grav_angle)

        na_idx = 2*element[0]
        nb_idx = 2*element[1]

        global_load[na_idx:na_idx+2] = local_load[0:2]
        global_load[nb_idx:nb_idx+2] = local_load[2:4]

    return global_load

def solve2dtruss(nodes, elements, extern_loads, bounds, mass_dens, cs_area, young_mod, grav_angle=-np.pi/2):
    '''
        bounds is an array with 1 indicating that the node/direction is allowed to move, 0 otherwise

        NOTE: this formulation assumes that if there is a reaction force in a given direction,
        then the structure is not allowed to move in that direction
    '''
    for el in bounds:
        assert(el == 0 or el == 1)

    stiff = truss_stiffness(nodes, elements, young_mod, cs_area)
    loads = truss_load_vec(nodes, elements, mass_dens, cs_area)

    loads += extern_loads

    keep_indices, = np.nonzero(bounds)

    stiff = stiff[keep_indices[:, None], keep_indices]
    loads = loads[keep_indices]

    try:
        redux_sol = np.linalg.solve(stiff, loads)
    except:
        breakpoint()

    # Put the full solution together
    sol = np.zeros(2*len(nodes))
    count=0
    for i,el in enumerate(bounds):
        if el:
            sol[i] = redux_sol[count]
            count += 1

    return sol

#######################
# Truss post-processing
#######################
def axial_strain(nodes, element, nodal_disp):
    # note nodal disp is the displacement vector reshapen into the same shape as nodes
    na = nodes[element[0]]
    nb = nodes[element[1]]

    na_post = np.add(na, nodal_disp[element[0]])
    nb_post = np.add(nb, nodal_disp[element[1]])


    l_pre = truss_len(na, nb)
    l_post = truss_len(na_post, nb_post)

    return (l_post - l_pre)/l_pre

def axial_stress(nodes, element, nodal_disp, ymod):
    return ymod*axial_strain(nodes, element, nodal_disp)

def axial_stresses(nodes, elements, nodal_disp, ymod):
    return [axial_stress(nodes, element, nodal_disp, ymod) for element in elements]

def axial_strains(nodes, elements, nodal_disp):
    return [axial_strain(nodes, element, nodal_disp) for element in elements]

from matplotlib import pyplot as plt
def plot_displacement(nodes, elements, displacement, scale_disp=1, label=True):

    displacement *= scale_disp

    # original nodes
    nodes_x, nodes_y = zip(*nodes)
    plt.plot(nodes_x, nodes_y, 'go', linewidth=0, markersize=12)

    # displaced nodes
    nodes_disp = nodes + np.reshape(displacement, (len(nodes), 2))
    nodes_x_disp, nodes_y_disp = zip(*nodes_disp)
    plt.plot(nodes_x_disp, nodes_y_disp, 'rx', linewidth=0, markersize=12)

    # elements between nodes
    nodes_x = np.asarray(nodes_x)
    nodes_y = np.asarray(nodes_y)
    nodes_x_disp = np.asarray(nodes_x_disp)
    nodes_y_disp = np.asarray(nodes_y_disp)

    counter = 1
    for elem in elements:
        xs = []
        plt.plot(nodes_x[elem], nodes_y[elem], 'g-', linewidth=2, markersize=0)
        plt.plot(nodes_x_disp[elem], nodes_y_disp[elem], 'r-', linewidth=2, markersize=0)

        if label:
            # marker location
            labelx = np.average(nodes_x_disp[elem])
            labely = np.average(nodes_y_disp[elem])
            plt.text(labelx, labely, str(counter))
            counter +=1

    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()



'''
|||||||||||||||||||||||||||||||||||||||||||||||
    2D planar structure.

    - bilinear shape function
    - plane stress (strain exists out of plane)




|||||||||||||||||||||||||||||||||||||||||||||||
'''
def quad_shape_funs(xi, eta):

    N_upleft = (1-xi)*(1+eta)/4
    N_upright = (1+xi)*(1+eta)/4
    N_lowleft = (1-xi)*(1-eta)/4
    N_lowright = (1+xi)*(1-eta)/4

    return [N_lowleft,N_upleft,N_lowright,N_upright]

def shape_fun_matrix(xi, eta):
    # Pack the N values into a 2x8 matrix
    Ns = quad_shape_funs(xi, eta)

    NN = np.zeros((2,8))

    for i in [0,1,2,3]:
        NN[0,2*i] = Ns[i]
        NN[1,2*i+1] = Ns[i]
    return NN

def shape_grad_xi(xi, eta):
    # NOTE: ordering is a bit switcherood from class notes
    # I number nodes like: ll, ul, lr, ur
    N1 = -(1+eta)/4
    N2 = (1+eta)/4
    N3 = -(1-eta)/4
    N4 = (1-eta)/4
    return [N3,N1,N4,N2]


def shape_grad_eta(xi, eta):
    # NOTE: ordering is a bit switcherood from class notes
    # I number nodes like: ll, ul, lr, ur
    N1 = (1-xi)/4
    N2 = (1+xi)/4
    N3 = -(1-xi)/4
    N4 = -(1+xi)/4
    return [N3,N1,N4,N2]

def global_to_local_jacobian(nodes, global_node_indices, xi, eta):
    '''
        Change in global coordinates wrt local coordinates

    '''
    # NOTE: again, only for planar quad
    global_nodes = nodes[global_node_indices]

    Nxi = shape_grad_xi(xi,eta)
    Neta = shape_grad_eta(xi,eta)

    xs, ys = zip(*global_nodes)
    #breakpoint()
    # element-wise multiply (i.e. dot product)
    xeta = np.sum(np.multiply(xs, Neta))
    xxi = np.sum(np.multiply(xs, Nxi))
    yeta = np.sum(np.multiply(ys, Neta))
    yxi = np.sum(np.multiply(ys, Nxi))

    jaco = [
        [xxi, yxi],
        [xeta, yeta]
        ]
    return jaco

def shape_grad_xy(nodes, global_node_indices, xi, eta):

    jacobian = global_to_local_jacobian(nodes, global_node_indices, xi, eta)

    jaco_inv = np.linalg.inv(jacobian)

    Nxi = shape_grad_xi(xi, eta)
    Neta = shape_grad_eta(xi, eta)

    Nxy = []

    for i in [0,1,2,3]:
        Nxy.append(np.matmul(jaco_inv, [Nxi[i], Neta[i]] ))

    return Nxy

def strain_disp_matrix(nodes, global_node_indices, xi, eta):
    # Strain displacement matrix.
    # Basically, mapping matrix from local to global displacements
    Nxy = shape_grad_xy(nodes, global_node_indices, xi, eta)

    B = np.zeros((3,8))

    for i,N in enumerate(Nxy):
        B[0,i*2] = N[0]
        B[1,i*2+1] = N[1]
        B[2,i*2] = N[1]
        B[2,i*2+1] = N[0]

    return B

def plane_stress_constitutive(ymod, poisson):
    # Constitutive matrix for the plane stress case.
    # out-of-plane strain is allowed in this case
    E = np.asarray([
        [1, poisson, 0],
        [poisson, 1, 0],
        [0, 0, (1-poisson)/2]
    ])

    E *= ymod/(1-poisson**2)

    return E

def planar_element_stiffness(nodes, global_node_indices, ymod, poisson, thickness):
    # K Stiffness matrix for an element
    E = plane_stress_constitutive(ymod, poisson)

    def integrand(xi, eta):
        B = strain_disp_matrix(nodes, global_node_indices, xi, eta)
        J = global_to_local_jacobian(nodes, global_node_indices, xi, eta)

        detJ = np.linalg.det(J)

        BtE = np.matmul(np.transpose(B), E)

        BtEB = np.matmul(BtE, B)*detJ
        return BtEB

    K = numerics.gquad332d(integrand, -1, 1, -1, 1)

    K *= thickness

    return K


def local_to_global(nodes, global_node_indices, xi, eta):
    # convert local xi, eta to global x,y
    Ns = quad_shape_funs(xi, eta)

    global_nodes = nodes[global_node_indices]

    xs, ys = zip(*global_nodes)

    xs = np.multiply(xs, Ns)
    ys = np.multiply(ys, Ns)

    return [np.sum(xs), np.sum(ys)]



from typing import Callable
def local_plane_load_vec(nodes, element,  extern_loads_fun, extern_loads_face, mass_dens, thick, weightless=False):

    ###
    # TODO: this only allows one face traction per element right now
    ###

    ###
    # TODO: add body forces
    ###

    # Local load vector for an element.
    force = np.zeros(2*len(element))

    ###
    # Applied loads
    ###

    # Edges for a quad
    edges = [
        [element[0],element[1]],
        [element[1],element[3]],
        [element[3],element[2]],
        [element[2],element[0]]
    ]

    # Standard +- 1 local nodes
    local_nodes = [
        [-1,-1],
        [-1,1],
        [1,-1],
        [1,1]
    ]

    # Local node indexes for a quad
    local_edges = [
        [0,1], #left
        [1,3], #up
        [2,3], #right
        [0,2]  #down
    ]

    # Taking extern loads face to be an enum for now
    el_face_options = ['l','u','r','d']

    # Change the integral based on which face the traction is applied
    if extern_loads_face == 'l':
        local_edge = local_edges[0]
        setval = -1
        edge_vertical = 1
        node_mask = [0,1]

    elif extern_loads_face == 'r':
        local_edge = local_edges[2]
        setval = 1
        edge_vertical = 1
        node_mask = [0,0,0,0,1,1,1,1]

    elif extern_loads_face == 'u':
        local_edge = local_edges[1]
        setval = 1
        edge_vertical = 0
        node_mask = [0,0,1,1,0,0,1,1]

    elif extern_loads_face == 'd':
        local_edge = local_edges[3]
        setval = -1
        edge_vertical = 0
        node_mask = [1,1,0,0,1,1,0,0]

    else:
        raise "We got a bad value for element loads face selection"

    def integrand(var):
        # decide which is xi and which is eta
        if edge_vertical:
            args = (setval, var)
        else:
            args = (var, setval)

        # integral is either over xi or eta. doesn't matter. limits are same
        # one each for x and y
        Ns = np.repeat(quad_shape_funs(*args), 2)

        x,y = local_to_global(nodes, element, *args)
        q = np.asarray(extern_loads_fun(x, y))

        # make one for each node, the only select the relevant ones
        q = np.tile(q,4)
        q = q*Ns

        jaco = global_to_local_jacobian(nodes, element, *args)
        ds = np.sqrt(jaco[edge_vertical][0]**2 + jaco[edge_vertical][1]**2)

        return q*ds

    # add the contribution from this edge to the rest
    force = force + numerics.gquad331d(integrand, -1, 1)

    return force


def global_plane_stiffness(nodes, elements, young_mod, poisson, thickness):

    if not isinstance(thickness, list):
        thickness = [thickness] * len(elements)
    problem_size = len(nodes) * 2

    global_stiffness = np.zeros((problem_size, problem_size))

    for element, thick in zip(elements, thickness):
        local_stiffness_mat = planar_element_stiffness(nodes, element, young_mod, poisson, thick)

        for i, nodei in enumerate(element):
            for j, nodej in enumerate(element):
                # map each 2x2 block of the local 2Mx2M stiffness matrix to the respective block in the global matrix
                global_stiffness[2*nodei:2*nodei+2, 2*nodej:2*nodej+2] += local_stiffness_mat[2*i:2*i+2, 2*j:2*j+2]

    return global_stiffness


def global_plane_load_vec(nodes, elements, extern_loads_fun, extern_loads_face, mass_dens: float, thick ):

    problem_size = len(nodes) * 2

    global_load = np.zeros(problem_size)

    for element in elements:
        local_load = local_plane_load_vec(nodes, element, extern_loads_fun,extern_loads_face, mass_dens, thick)


        for i, nodei in enumerate(element):
            # map each 2x2 block of the local 2Mx2M stiffness matrix to the respective block in the global matrix
            global_load[2*nodei:2*nodei+2] += local_load[2*i:2*i+2]

    return global_load

from pytest import approx
def solve2dplane(nodes, elements, extern_loads_fun, bounds, mass_dens, thickness, young_mod, poisson):
    '''
        bounds is an array with 1 indicating that the node/direction is allowed to move, 0 otherwise

        NOTE: this formulation assumes that if there is a reaction force in a given direction,
        then the structure is not allowed to move in that direction

        TODO: this only supports one traction direction at a time (although
        extending it probably won't be too bad).

        TODO: This also doesnt support body forces, like gravity, but that
        shouldn't be too bad either.
    '''
    for el in bounds:
        assert(el == 0 or el == 1)

    nodes = np.asarray(nodes)

    stiff_full = global_plane_stiffness(nodes, elements, young_mod, poisson, thickness)
    loads_full = global_plane_load_vec(nodes, elements, extern_loads_fun, 'd', mass_dens, thickness)

    tot_load = sum(loads_full)
    # detemine which fixed supports to remove from the equation
    keep_indices, = np.nonzero(bounds)

    # only keep the parts of the problem that are not pinned down
    stiff = stiff_full[keep_indices[:, None], keep_indices]
    loads = loads_full[keep_indices]

    assert(tot_load == approx(sum(loads)))
    print('total applied force: ')
    print(tot_load)
    redux_sol = np.linalg.solve(stiff, loads)

    # Put the full solution together
    sol = np.zeros(2*len(nodes))
    count=0
    for i,el in enumerate(bounds):
        if el:
            sol[i] = redux_sol[count]
            count += 1

    def element_area():
        el = nodes[elements[0]]
        n1 = np.asarray(el[0])
        n2 = np.asarray(el[3])

        nn = n2-n1
        return nn[0]*nn[1]

    area = element_area()

    # Calculate strain energy density I guess
    subdKd = np.zeros(len(elements))
    for i,element in enumerate(elements):
        element = np.asarray(element)
        subK = stiff_full[element[:, None], element]
        subd = sol[element]

        sKd = np.matmul(subK, subd)
        sdKd = np.matmul(subd, sKd)/2
        subdKd[i] = sdKd/area

    #total strain enery:
    Kd = np.matmul(stiff_full, sol)
    dKd = np.matmul(sol, Kd)/2

    return sol, dKd, subdKd


def plot_displacement_plane(nodes, elements, displ, scale_disp=1, label=True):

    displacement = scale_disp*displ

    # original nodes
    nodes_x, nodes_y = zip(*nodes)
    plt.plot(nodes_x, nodes_y, 'go', linewidth=0, markersize=12)

    # displaced nodes
    nodes_disp = nodes + np.reshape(displacement, (len(nodes), 2))
    nodes_x_disp, nodes_y_disp = zip(*nodes_disp)
    plt.plot(nodes_x_disp, nodes_y_disp, 'rx', linewidth=0, markersize=12)

    # elements between nodes
    nodes_x = np.asarray(nodes_x)
    nodes_y = np.asarray(nodes_y)
    nodes_x_disp = np.asarray(nodes_x_disp)
    nodes_y_disp = np.asarray(nodes_y_disp)

    counter = 1
    for elem in elements:
        xs = []
        sides = [
        [elem[0], elem[1]],
        [elem[0], elem[2]],
        [elem[1], elem[3]],
        [elem[2], elem[3]]
        ]

        for side in sides:
            plt.plot(nodes_x[side], nodes_y[side], 'g-', linewidth=2, markersize=0)
            plt.plot(nodes_x_disp[side], nodes_y_disp[side], 'r-', linewidth=2, markersize=0)

        if label:
            # marker location
            labelx = np.average(nodes_x_disp[elem])
            labely = np.average(nodes_y_disp[elem])
            plt.text(labelx, labely, str(counter))
            counter +=1

    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()

def plot_energy_dens(nodes, elements, energy):

    # coords for centers of elements
    xc = [0.05, 0.15, 0.25, 0.35, 0.45]
    yc = [0.05, 0.15]

    energy_elem = np.zeros((len(yc),len(xc)))
    count = 0

    for j in range(len(xc)):
        for i in range(len(yc)):
            energy_elem[-i, j] = energy[count]
            count = count+1

    fig, ax = plt.subplots()
    im = ax.imshow(energy_elem)

    # original nodes
    nodes_x, nodes_y = zip(*nodes)
    #plt.plot(nodes_x, nodes_y, 'go', linewidth=0, markersize=12)

    # elements between nodes
    nodes_x = np.asarray(nodes_x)
    nodes_y = np.asarray(nodes_y)

    counter = 1
    for elem in elements:
        xs = []
        sides = [
        [elem[0], elem[1]],
        [elem[0], elem[2]],
        [elem[1], elem[3]],
        [elem[2], elem[3]]
        ]

        #for side in sides:
            #plt.plot(nodes_x[side], nodes_y[side], 'g-', linewidth=2, markersize=0)

    plt.gca().set_aspect('equal', adjustable='box')
    for i in range(2):
        for j in range(5):
            text = ax.text(j, i, '{:.2e}'.format(energy_elem[i, j]),
                           ha="center", va="center", color="w")

    #plt.ylim((0,0.2))
    #plt.xlim((0,0.5))
    plt.yticks(np.arange(2), '')
    plt.xticks(np.arange(5), '')
    plt.show()

def truss1(young_mod, cs_area, mass_dens):
    # 3 element problem from notes with L=1
    nodes = [
    [0,0],
    [2,0],
    [1,-1]
    ]

    elements = [
    [0,1],
    [1,2],
    [0,2]
    ]

    extern_loads = np.zeros(6)
    bounds = [0,0,1,0,1,1]


    sol = solve2dtruss(nodes, elements, extern_loads, bounds, mass_dens, cs_area, young_mod)
    plot_displacement(nodes,elements,sol)
    return




import numpy as np
from IPython.display import HTML, display
import tabulate
def solplane():
    P_0 = 2e9 #N/m^2
    mass_dens = 800 # kg/m^3
    ymod = 150e9 #Pa
    poisson = 0.3
    depth = 0.05 # meters
    height = 0.2 # meters
    width = 0.5 # meters

    def mesh(nx, ny):
        ''' Build nodes'''
        x_dist = width/nx
        y_dist = height/ny

        xs = x_dist*np.arange(0, nx + 1)
        ys = y_dist*np.arange(0, ny + 1)

        xs,ys = np.meshgrid(xs, ys)

        nodes = list(zip(xs.ravel(), ys.ravel()))

        nodes = sorted(nodes, key=lambda node: (node[0], node[1]) )

        ''' Build elements'''
        elements = []
        for i in range(nx):
            for j in range(ny):
                elements.append([
                    i*(ny+1)+j,
                    i*(ny+1)+j + 1,
                    (i+1)*(ny+1)+j,
                    (i+1)*(ny+1)+j + 1
                ])

        return nodes,elements


    # Create a grid
    nx = 5
    ny = 2
    nodes,elements = mesh(nx,ny)

    # fix the left side
    bounds = np.ones(2*len(nodes))
    bounds[:(nx+1)*2] = 0

    def extern_loads_fun(x,y, tol=1e-3):
        loady = 0
        loadxmin = 0.4
        loadxmax = 0.5
        n_elems = (loadxmax-loadxmin)/(width/nx)
        elem_force = P_0*(loadxmax-loadxmin)/n_elems

        if abs(y-loady) < tol and x >= loadxmin and x <= loadxmax:
            return np.asarray([0,-elem_force])
        else:
            return np.zeros(2)

    # solve
    disp,energy,Ks = solve2dplane(nodes, elements, extern_loads_fun, bounds,
                 mass_dens, depth, ymod, poisson)


    plot_displacement_plane(nodes, elements,disp, scale_disp=1)
