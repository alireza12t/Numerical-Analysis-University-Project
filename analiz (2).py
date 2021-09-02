import numpy as np
import scipy
import math


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def Cholesky_Decomposition(matrix, n):
    lower = [[0 for x in range(n + 1)]
             for y in range(n + 1)]


    for i in range(n):
        for j in range(i + 1):
            sum1 = 0

            if j == i:
                for k in range(j):
                    sum1 += pow(lower[j][k], 2)
                lower[j][j] = int(math.sqrt(matrix[j][j] - sum1))
            else:

                for k in range(j):
                    sum1 += (lower[i][k] * lower[j][k])
                if lower[j][j] > 0:
                    lower[i][j] = int((matrix[i][j] - sum1) /
                                      lower[j][j])

    print("Lower Triangular         Transpose")
    for i in range(n):

        # Lower Triangular
        for j in range(n):
            print(lower[i][j], end="\t")
        print("", end="\t")

        for j in range(n):
            print(lower[j][i], end="\t")
        print("")


def get_matrix():
    R = int(input("Enter the number of rows:"))

    matrix = []
    print("Enter the entries:")

    # For user input
    for i in range(R):  # A for loop for row entries
        a = []
        for j in range(R):  # A for loop for column entries
            a.append(int(input()))
        matrix.append(a)

        # For printing the matrix
    for i in range(R):
        for j in range(R):
            print(matrix[i][j], end=" ")
        print()
    return matrix


def get_vector():
    R = int(input("Enter the number of Matrix rows:"))
    b = []
    for i in range(R):
        b.append("Enter the entries:")
    return b


def forward_sub(L, b):
    y = []
    for i in range(len(b)):
        y.append(b[i])
        for j in range(i):
            y[i] = y[i] - (L[i, j] * y[j])
        y[i] = y[i] / L[i, i]

    return y


def backward_sub(U, y):
    x = np.zeros_like(y)

    for i in range(len(x), 0, -1):
        x[i - 1] = (y[i - 1] - np.dot(U[i - 1, i:], x[i:])) / U[i - 1, i - 1]

    return x


def lu_factor(A):
    L = np.zeros_like(A)
    U = np.zeros_like(A)

    N = np.size(A, 0)

    for k in range(N):
        L[k, k] = 1
        U[k, k] = (A[k, k] - np.dot(L[k, :k], U[:k, k])) / L[k, k]
        for j in range(k + 1, N):
            U[k, j] = (A[k, j] - np.dot(L[k, :k], U[:k, j])) / L[k, k]
        for i in range(k + 1, N):
            L[i, k] = (A[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]
    return L, U


def lu_solve(L, U, b):
    y = forward_sub(L, b)
    x = backward_sub(U, y)

    return x


def linear_solve(A, b):
    L, U = lu_factor(A)
    x = lu_solve(L, U, b)
    return x


if __name__ == '__main__':
    matrix = get_matrix()
    vector = get_vector()
    if is_pos_def(matrix):
        Cholesky_Decomposition(matrix, len(matrix))
    else:
        A = scipy.array(matrix)
        P, L, U = scipy.linalg.lu(A)
    print()
    print()
    print()
    print(linear_solve(matrix, vector))

