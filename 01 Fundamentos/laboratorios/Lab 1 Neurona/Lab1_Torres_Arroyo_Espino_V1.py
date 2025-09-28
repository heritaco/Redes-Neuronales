# %%
"""
Lab1_Torres_Arroyo_Espino_V1.py  
"""
import numpy as np

class Neurona:
    """
    Una clase para representar una neurona artificial simple.

    Attributes
    ----------
    n : int
        Número de entradas de la neurona.
    pesos : numpy.ndarray
        Vector de pesos sinápticos.
    umbral : float
        Umbral de activación.

    """
    def __init__(self, n, pesos=None, umbral=None):
        """
        Inicializa una nueva neurona.

        Parameters
        ----------
        n : int
            El número de entradas que la neurona recibirá.
        pesos : array_like, optional
            Los pesos sinápticos de la neurona. Si es None, se inicializan
            aleatoriamente. (default is None)
        umbral : float, optional
            El umbral de activación de la neurona. Si es None, se establece
            en 0. (default is None)
        """
        self.n = n
        if pesos is None:
            self.pesos = np.random.rand(n)  # Inicializa pesos aleatorios si no se proporcionan
        else:
            self.pesos = np.array(pesos)

        if umbral is None:
            self.umbral = 0
        else:
            self.umbral = umbral

    def ProdVec(self, x):
        """
        Calcula la suma ponderada de las entradas.

        Calcula el producto punto entre el vector de entrada `x` y el vector
        interno de pesos de la neurona, y luego resta el umbral.

        Parameters
        ----------
        x : array_like
            Un vector de entrada de tamaño `n`.

        Returns
        -------
        float
            El resultado de la suma ponderada menos el umbral.
        """
        return np.dot(self.pesos, x) - self.umbral
    
    def Activacion(self, r):
        """
        Aplica la función de activación escalón.

        Parameters
        ----------
        r : float
            El resultado de la suma ponderada (salida de `ProdVec`).

        Returns
        -------
        int
            1 si `r` es mayor o igual a 0, y 0 en caso contrario.
        """
        return 1 if r >= 0 else 0

    # Getters
    def get_pesos(self):
        """
        Obtiene los pesos de la neurona.

        Returns
        -------
        numpy.ndarray
            El vector de pesos de la neurona.
        """
        return self.pesos

    def get_umbral(self):
        """
        Obtiene el umbral de la neurona.

        Returns
        -------
        float
            El valor del umbral de la neurona.
        """
        return self.umbral

    def get_n(self):
        """
        Obtiene el número de entradas de la neurona.

        Returns
        -------
        int
            El número de entradas.
        """
        return self.n

    # Setters
    def set_pesos(self, pesos):
        """
        Establece los pesos de la neurona.

        Parameters
        ----------
        pesos : array_like
            El nuevo vector de pesos para la neurona.
        """
        self.pesos = np.array(pesos)

    def set_umbral(self, umbral):
        """
        Establece el umbral de la neurona.

        Parameters
        ----------
        umbral : float
            El nuevo valor del umbral para la neurona.
        """
        self.umbral = umbral

    
# %%

class AND(Neurona):
    """
    Una neurona que implementa la compuerta lógica AND.
    """
    def __init__(self):
        """
        Inicializa la neurona AND con pesos y umbral predefinidos.

        Para la puerta AND, los pesos son [1, 1] y el umbral es 1.5
        """
        super().__init__(n=2, pesos=[1, 1], umbral=1.5)

class OR(Neurona):
    """
    Una neurona que implementa la compuerta lógica OR.
    """
    def __init__(self):
        """
        Inicializa la neurona OR con pesos y umbral predefinidos.
        
        Para la puerta OR, los pesos son [1, 1] y el umbral es 0.5
        """
        super().__init__(n=2, pesos=[1, 1], umbral=0.5)

class NOT(Neurona):
    """
    Una neurona que implementa la compuerta lógica NOT.
    """
    def __init__(self):
        """
        Inicializa la neurona NOT con peso y umbral predefinidos.

        Para la puerta NOT, los pesos son [-1] y el umbral es -0.5
        """
        super().__init__(n=1, pesos=[-1], umbral=-0.5)

class IDENTITY(Neurona):
    """
    Una neurona que implementa la función de identidad.
    """
    def __init__(self):
        """
        Inicializa la neurona IDENTITY con peso y umbral predefinidos.

        Para la puerta IDENTITY, los pesos son [1] y el umbral es 0.5
        """
        super().__init__(n=1, pesos=[1], umbral=0.5)

# %% 

# Ejemplos
# %%
neuronaand = AND()
print("AND:")
print("0 AND 0 =", neuronaand.Activacion(neuronaand.ProdVec([0, 0])))
print("0 AND 1 =", neuronaand.Activacion(neuronaand.ProdVec([0, 1])))
print("1 AND 0 =", neuronaand.Activacion(neuronaand.ProdVec([1, 0])))
print("1 AND 1 =", neuronaand.Activacion(neuronaand.ProdVec([1, 1])))

# %% 
neuronaor = OR()
print("\nOR:")
print("0 OR 0 =", neuronaor.Activacion(neuronaor.ProdVec([0, 0])))
print("0 OR 1 =", neuronaor.Activacion(neuronaor.ProdVec([0, 1])))
print("1 OR 0 =", neuronaor.Activacion(neuronaor.ProdVec([1, 0])))
print("1 OR 1 =", neuronaor.Activacion(neuronaor.ProdVec([1, 1])))

# %%
neuronanot = NOT()
print("\nNOT:")
print("NOT 0 =", neuronanot.Activacion(neuronanot.ProdVec([0])))
print("NOT 1 =", neuronanot.Activacion(neuronanot.ProdVec([1])))

# %%
neuronaid = IDENTITY()
print("\nIDENTITY:")
print("IDENTITY 0 =", neuronaid.Activacion(neuronaid.ProdVec([0])))
print("IDENTITY 1 =", neuronaid.Activacion(neuronaid.ProdVec([1])))

# %%
not0 = neuronanot.Activacion(neuronanot.ProdVec([0]))
not1 = neuronanot.Activacion(neuronanot.ProdVec([1]))

print("\nNOT0 and NOT0:")
print(neuronaand.Activacion(neuronaand.ProdVec([not0, not0])))
# %%
print("\nNOT0 and NOT1:")
print(neuronaand.Activacion(neuronaand.ProdVec([not0, not1])))

# %% 
print("\nNOT1 or NOT1:")
print(neuronaor.Activacion(neuronaor.ProdVec([not1, not1])))

# %%
print("\nNOT1 or NOT0:")
print(neuronaor.Activacion(neuronaor.ProdVec([not1, not0])))

# %%
print("\nNOT1 or NOT1 or NOT1 or NOT1 or 1:")
primeror = neuronaor.Activacion(neuronaor.ProdVec([not1, not1]))
segundor = neuronaor.Activacion(neuronaor.ProdVec([primeror, not1]))
terceror = neuronaor.Activacion(neuronaor.ProdVec([segundor, not1]))
cuartoror = neuronaor.Activacion(neuronaor.ProdVec([terceror, 1]))
print(cuartoror)
# %%
