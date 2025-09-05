# %%
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, S, T, weights=None, bias=None, learning_rate=0.01,
                 n_iter=1000, umbral=0.5):
        
        self.S = S
        self.T = T
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.umbral = umbral
        
        self.Wi = []
        self.Bi = []
        self.MSEi = []
        self.Malos = []

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
                    self.weights = self.weights + self.learning_rate * ti * si
                    self.bias = self.bias + self.learning_rate * ti

                # Almacenar el historial
                self.Wi.append(self.weights.copy())
                self.Bi.append(self.bias)
                self.MSEi.append((ti - y)**2)
                if y != ti:
                    self.Malos.append((si, ti, jane, y))

            # Step 6. Test stopping condition:
            self.n_iter -= 1
            if not np.any(self.weights != weights_old) and self.bias == bias_old:
                break

        return self

    # Función para graficar los datos y el hiperplano
    def plot_decision_boundary(self):
        
        S = self.S
        T = self.T
        Wi = self.Wi
        Bi = self.Bi
        MSEi = self.MSEi
        Malos = self.Malos

        plt.figure(figsize=(10, 6))
        plt.scatter(S[:, 0], S[:, 1], c=T, cmap='bwr', edgecolors='k')

        x_min, x_max = S[:, 0].min() - 1, S[:, 0].max() + 1
        y_min, y_max = S[:, 1].min() - 1, S[:, 1].max() + 1

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        Z = np.zeros(xx.shape)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                jane = np.dot([xx[i, j], yy[i, j]], self.weights) + self.bias
                Z[i, j] = self._activation_function(jane)

        # plot the line of the decision xi wi + b = 0
        plt.plot([x_min, x_max], [(-self.bias - self.weights[0] * x_min) / self.weights[1],
                                   (-self.bias - self.weights[0] * x_max) / self.weights[1]],
                 color='k', linestyle='--')

        plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Perceptron Decision Boundary')
        plt.show()


# %%
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

p.plot_decision_boundary()

# %%

# %%
