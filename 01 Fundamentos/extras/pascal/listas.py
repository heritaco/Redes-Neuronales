def pascal(n):
    """
    Construye el triángulo de Pascal con n filas
    usando listas anidadas.
    """
    triangulo = []

    for i in range(n):
        # cada fila inicia con 1
        fila = [1] * (i + 1)

        # relleno de elementos internos con la recurrencia
        for j in range(1, i):
            fila[j] = triangulo[i-1][j-1] + triangulo[i-1][j]

        triangulo.append(fila)

    return triangulo


# Ejemplo
n = 6
tri = pascal(n)

for fila in tri:
    print(fila)

