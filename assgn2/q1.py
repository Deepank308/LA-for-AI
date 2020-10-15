"""
Deepank Agrawal
17CS30011

LA for AI 
Assignment 2
"""
import numpy as np
from pprint import PrettyPrinter


def gram_schmidt(A):
    """
    Orthogonalize a set of vectors stored as the columns of matrix A.
    """
    # Get the number of vectors.
    n = A.shape[1]

    for j in range(n):
        # To orthogonalize the vector in column j with respect to the
        # previous vectors, subtract from it its projection onto
        # each of the previous vectors.
        temp = np.copy(A[:, j])
        for k in range(j):
            A[:, j] -= np.dot(A[:, k], temp) * A[:, k]
        
        # normalize
        A[:, j] = A[:, j] / np.linalg.norm(A[:, j])
    
    return np.round(A, decimals=3)


def construct_matrix_get_gm(vec_list):
    """
    Create matrix using given vector list
    Then calculate the G-S othogonalization
    """
    A = vec_list[0]
    for v in vec_list[1:]:
        A = np.hstack((A, v))

    return gram_schmidt(A)


if __name__ == '__main__':
    a1 = np.array([[-1], [0], [0], [0], [0]], dtype=np.float64)
    a2 = np.array([[-1], [-1], [0], [0], [0]], dtype=np.float64)
    a3 = np.array([[-1], [-1], [-1], [0], [0]], dtype=np.float64)
    a4 = np.array([[-1], [-1], [-1], [-1], [0]], dtype=np.float64)
    a5 = np.array([[-1], [-1], [-1], [-1], [-1]], dtype=np.float64)

    pp = PrettyPrinter()

    print("part a:")
    vec_list = [a1, a2, a3, a4, a5]
    pp.pprint(construct_matrix_get_gm(vec_list))

    print("part b:")
    vec_list = [a1, a3, a5, a2, a4]
    pp.pprint(construct_matrix_get_gm(vec_list))

    print("part c:")
    vec_list = [a5, a4, a3, a2, a1]
    pp.pprint(construct_matrix_get_gm(vec_list))
