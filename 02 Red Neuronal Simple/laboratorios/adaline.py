# %%
import numpy as np

class ADALINE:
    def __init__(self, S, T, weights=None, bias=None, learning_rate=0.01,
                 n_iter=1000, umbral=0.5, tolerance=1e-9):
        self.S = S
        self.T = T
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.umbral = umbral
        self.tolerance = tolerance

    def _activation_function(self, jane):
        if jane > self.umbral:
            return 1
        elif abs(jane) <= self.umbral:
            return 0
        else:
            return -1

    def fit(self):

        S = self.S
        T = self.T
        
        n_samples, n_features = S.shape

        # Inicializar pesos y bias
        if self.weights is None:
            self.weights = np.zeros(n_features)
        if self.bias is None:
            self.bias = 0

        # Listas para almacenar el historial de pesos, bias, MSE y patrones mal clasificados
        Wi = []
        Bi = []
        MSEi = []
        Malos = []

        # Step 1. While stopping condition false
        while self.n_iter:
            # Step 2. For each training pair (s : t)
            for si, ti in zip(S, T):
                # Step 3. Set activations of input units
                # Step 4. Compute response of output unit
                jane = np.dot(si, self.weights) + self.bias
                y = self._activation_function(jane)

                weights_old = self.weights.copy()
                bias_old = self.bias

                #Step 5. Update weights and bias if an error occurred for this pattern. 
                # If y ̸= t:
                if y != ti:
                    self.weights = self.weights + self.learning_rate * (ti - y) * si
                    self.bias = self.bias + self.learning_rate * (ti - y)

                # Almacenar el historial
                Wi.append(self.weights.copy())
                Bi.append(self.bias)
                MSEi.append((ti - y)**2)
                if y != ti:
                    Malos.append((si, ti, jane, y))

            # Step 6. Test stopping condition:
            self.n_iter -= 1
            if np.all(np.abs(self.weights - weights_old) < self.tolerance) and abs(self.bias - bias_old) < self.tolerance:
                break

        return Wi, Bi, MSEi, Malos
    
# %% Test the ADALINE implementation
if __name__ == "__main__":
    # Datos de entrada 
    S = np.array([[1, 3],
                [3, 3],
                [1, 1],
                [3, 1],
                [-2, -1],
                [-1, -2]])

    T = np.array([-1, -1, -1, -1, 1, 1])  # Salidas esperadas

    # Crear y entrenar el ADALINE
    adaline = ADALINE(S, T, learning_rate=0.01, n_iter=10, umbral=0.5, tolerance=1e-16)
    adaline.fit()

    # Mostrar pesos y bias finales
    print("Pesos finales:", adaline.weights)
    print("Bias final:", adaline.bias)

    # Probar el ADALINE con los mismos datos de entrada
    for i, (si, ti) in enumerate(zip(S, T)):
        jane = np.dot(si, adaline.weights) + adaline.bias
        y = adaline._activation_function(jane)
        print(f"Entrada: {si}, Target: {ti}, Jane: {jane:.2f}, Salida: {y}")
# %%
adaline
# %%
