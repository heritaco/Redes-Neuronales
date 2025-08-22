def pascal_matrix(n):
    """
    Construye el triángulo de Pascal en una matriz n×n
    (rellenando con ceros las posiciones no usadas).
    """
    # matriz inicial llena de ceros
    M = [[0]*n for _ in range(n)]

    for i in range(n):
        M[i][0] = 1
        M[i][i] = 1
        for j in range(1, i):
            M[i][j] = M[i-1][j-1] + M[i-1][j]

    return M


# Ejemplo
n = 6
M = pascal_matrix(n)

for fila in M:
    print(fila)
