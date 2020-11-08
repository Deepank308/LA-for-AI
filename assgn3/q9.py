import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt


def plot_unit_circle(ax):
    
    x_coord = np.arange(-1, 1, 0.001)
    y_coord = np.array(list(map(lambda x : math.sqrt(1 - x**2), x_coord)))

    y_coord = np.append(y_coord, np.array(list(map(lambda x : -math.sqrt(1 - x**2), x_coord))))
    x_coord = np.append(x_coord, x_coord)

    ax.scatter(x_coord, y_coord, marker='.', c='r')

    return ax


def plot_unit_sphere(ax):
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    # alpha controls opacity
    ax.plot_surface(x, y, z, color='r', alpha=0.3)

    return ax


def get_transformation_matrix(option, eps=-1):
    A = None

    if option == 0:
        A = np.array([[-1 / math.sqrt(2), 0.0], [0, -1 / math.sqrt(2)], [-1.0, 1.0]])
    elif option == 1:
        A = np.array([[-2, 1, 2], [0, 2, 0]])
    elif option == 2:
        A = np.array([[1.0, 0.9], [0.9, 0.8]])
    elif option == 3:
        A = np.array([[1.0, 0.0], [0.0, -10.0]])
    elif option == 4 and eps != -1:
        A = np.array([[1.0, 1.0], [1.0, eps]])
    else:
        print('Enter a valid option.')

    return A


def tranform(A, point):
    return np.matmul(A, point)


def calc_condition_number_determinant(option=0, eps=0):
    A = get_transformation_matrix(option, eps)
    print(A)
    cond = np.linalg.cond(A)
    print(f'Condition number: {cond}')

    if A.shape[0] == A.shape[1]:
        if cond < (1 / sys.float_info.epsilon):
            det = np.linalg.det(A)
            print('Invertible')
            print(f'Determinant: {det}')
        else:
            print('Not Invertible')

    print('')


def transform_and_plot(option, eps=None, name='sample.png', save_plt=True):
    if option == 4:
        A = get_transformation_matrix(option, eps)
    else:
        A = get_transformation_matrix(option)

    fig = plt.figure()

    if A.shape[0] > 2 or A.shape[1] > 2:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111, aspect='equal')

    x_coord = np.arange(-1, 1, 0.001)

    y_coord = np.array(list(map(lambda x : math.sqrt(1 - x**2), x_coord)))
    y_coord = np.append(y_coord, np.array(list(map(lambda x : -math.sqrt(1 - x**2), x_coord))))
    x_coord = np.append(x_coord, x_coord)

    z_coord = np.zeros(x_coord.shape)

    if A.shape[1] == 2:
        ax = plot_unit_circle(ax) 
    else:
        ax = plot_unit_sphere(ax)

    if A.shape[1] == 3:
        x_coord = np.arange(-1, 1, 0.001)
        y_coord = np.arange(-1, 1, 0.001)
        z_coord = []
        z_coord_n = []
        for x in x_coord:
            for y in y_coord:
                val = x**2 + y**2
                if val <= 1:
                    z = math.sqrt(1 - val)
                    z_coord.append(z)
                    z_coord_n.append(-z)

        z_coord += z_coord_n
        x_coord = np.append(x_coord, x_coord)
        y_coord = np.append(y_coord, y_coord)


    transformed_x = [0]
    transformed_y = [0]
    transformed_z = [0]
    for x, y, z in zip(x_coord, y_coord, z_coord):
        if A.shape[1] > 2:
            point = np.array([x, y, z])
        else:
            point = np.array([x, y])
        transformed_p = tranform(A, point)

        transformed_x.append(transformed_p[0])
        transformed_y.append(transformed_p[1])
        if len(transformed_p) > 2:
            transformed_z.append(transformed_p[2])

    # plot the points
    if A.shape[0] > 2:
        # print(transformed_z)
        ax.scatter(transformed_x, transformed_y, transformed_z, c='b')
    else:
        ax.scatter(transformed_x, transformed_y, marker='.', c='b', linewidth=0.5)
    # ax.plot_surface(x, y, z, color='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    if type(ax).__name__ == 'Axes3DSubplot':
        ax.set_zlabel('Z')
        minbound = -1.5
        maxbound = 1.5
        ax.auto_scale_xyz([minbound, maxbound], [minbound, maxbound], [minbound, maxbound])
    else:
        ax.set_aspect(1.0)
    plt.grid()
    
    if save_plt:
        plt.savefig(name)
    plt.close(fig)


def main():
    os.makedirs('./plots', exist_ok=True)
    ifsave = False
    # option = int(input('Which option?'))

    for option in range(5):
        if option == 4:
            eps_list = [10, 5, 1, 0.1, 0.01, 0.0001, 0]

            for i, eps in enumerate(eps_list):
                transform_and_plot(option, eps=eps, name=f'./plots/q9-{option}-{i}.png', save_plt=ifsave)
                calc_condition_number_determinant(option, eps)
        else:
            transform_and_plot(option, name=f'./plots/q9-{option}.png', save_plt=ifsave)
            calc_condition_number_determinant(option)


if __name__ == "__main__":
    main()
