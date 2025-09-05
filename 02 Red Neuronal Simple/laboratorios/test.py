# %%
from perceptron import Perceptron
import numpy as np

# Datos de entrada 
S = np.array([[1, 3],
            [3, 3],
            [1, 1],
            [3, 1],
            [-2, -1],
            [-1, -2]])

T = np.array([-1, -1, -1, -1, 1, 1])  # Salidas esperadas

# Crear y entrenar el perceptrón
p = Perceptron(S, T, learning_rate=0.5, n_iter=10, umbral=0.5)
p.fit()

# Mostrar pesos y bias finales
print("Pesos finales:", p.weights)
print("Bias final:", p.bias)

# Probar el perceptrón con los mismos datos de entrada
for si in S:
    jane = np.dot(si, p.weights) + p.bias
    y = p._activation_function(jane)
    print(f"Entrada: {si}, Salida: {y}")
# %%

p.plot_decision_boundary()
# %%

# %% Ejemplo de uso
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    S = data.data[:, :2]  # Usar solo las dos primeras características para visualización
    T = np.where(data.target == 0, -1, 1)  # Convertir etiquetas a -1 y 1

    p = Perceptron(S, T, learning_rate=0.01, n_iter=1000, umbral=0.5)
    Wi, Bi, MSEi, Malos = p.fit()

    print("Pesos finales:", p.weights)
    print("Bias final:", p.bias)

    p.plot_decision_boundary()
