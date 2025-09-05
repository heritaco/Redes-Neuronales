# %%
import numpy as np
from matplotlib.widgets import Slider
import matplotlib
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, S, T, weights=None, bias=None, learning_rate=0.01,
                 n_iter=1000, umbral=0.5):
        """Inicializar el perceptrón con los parámetros dados.
        Args:
            S (np.ndarray): Matriz de características de entrada (n_samples, n_features).
            T (np.ndarray): Vector de etiquetas objetivo (n_samples,).
            weights (np.ndarray, optional): Pesos iniciales del perceptrón. Si es None, se inicializan a cero.
            bias (float, optional): Sesgo inicial del perceptrón. Si es None, se inicializa a cero.
            learning_rate (float, optional): Tasa de aprendizaje para la actualización de pesos. Default es 0.01.
            n_iter (int, optional): Número máximo de iteraciones para el entrenamiento. Default es 1000.
            umbral (float, optional): Umbral para la función de activación. Default es 0.5.
        Returns:
            None
        """

        self.S = np.array(S)
        self.T = np.array(T)
        self.weights = np.array(weights) if weights is not None else None
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
        """Entrenar el perceptrón usando el algoritmo de aprendizaje del perceptrón.
        Returns:
            self: objeto entrenado
            """

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
    
    # Función para graficar los datos y el hiperplano usando TkAgg
    def plot(self):
        """Graficar los datos y el hiperplano de decisión del perceptrón.
        Utiliza la biblioteca matplotlib con el backend TkAgg para crear una gráfica interactiva"""
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        S = self.S
        T = self.T
        Wi = self.Wi
        Bi = self.Bi

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.25)

        ax.scatter(S[:, 0], S[:, 1], c=T, cmap='bwr', edgecolors='k')

        x_min, x_max = S[:, 0].min() - 1, S[:, 0].max() + 1
        y_min, y_max = S[:, 1].min() - 1, S[:, 1].max() + 1

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                np.linspace(y_min, y_max, 100))

        # Texto para mostrar Wi y Bi
        wi_text = fig.text(0.15, 0.01, '', fontsize=10)
        bi_text = fig.text(0.55, 0.01, '', fontsize=10)

        def plot_boundary(epoch):
            ax.clear()
            ax.scatter(S[:, 0], S[:, 1], c=T, cmap='bwr', edgecolors='k')
            if epoch < len(Wi):
                weights = Wi[epoch]
                bias = Bi[epoch]
            else:
                weights = self.weights
                bias = self.bias

            Z = np.zeros(xx.shape)
            for i in range(xx.shape[0]):
                for j in range(xx.shape[1]):
                    jane = np.dot([xx[i, j], yy[i, j]], weights) + bias
                    Z[i, j] = self._activation_function(jane)

            # plot the line of the decision xi wi + b = 0
            if weights[1] != 0:
                ax.plot([x_min, x_max],
                        [(-bias - weights[0] * x_min) / weights[1],
                            (-bias - weights[0] * x_max) / weights[1]],
                        color='k', linestyle='--')
            ax.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title(f'Perceptron Decision Boundary - Epoch {epoch+1}/{len(Wi)}')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            # Mostrar Wi y Bi
            wi_text.set_text(f'Wi: {np.round(weights, 3)}')
            bi_text.set_text(f'Bi: {np.round(bias, 3)}')

            fig.canvas.draw_idle()

        # Slider
        ax_epoch = plt.axes([0.2, 0.08, 0.6, 0.03])
        slider = Slider(ax_epoch, 'Epoch', 1, max(1, len(Wi)), valinit=1, valstep=1)

        def update(_):
            plot_boundary(int(slider.val) - 1)

        slider.on_changed(update)
        plot_boundary(0)
        plt.ion()
        plt.show(block=True)