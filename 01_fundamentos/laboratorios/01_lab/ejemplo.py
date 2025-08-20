
#%%
from Lab1_Torres_Arroyo_Espino_V1 import *

#%%
neurona = AND()
print("AND:")
print("0 AND 0 =", neurona.Activacion(neurona.ProdVec([0, 0])))
print("0 AND 1 =", neurona.Activacion(neurona.ProdVec([0, 1])))
print("1 AND 0 =", neurona.Activacion(neurona.ProdVec([1, 0])))
print("1 AND 1 =", neurona.Activacion(neurona.ProdVec([1, 1])))

neurona = OR()
print("\nOR:")
print("0 OR 0 =", neurona.Activacion(neurona.ProdVec([0, 0])))
print("0 OR 1 =", neurona.Activacion(neurona.ProdVec([0, 1])))
print("1 OR 0 =", neurona.Activacion(neurona.ProdVec([1, 0])))
print("1 OR 1 =", neurona.Activacion(neurona.ProdVec([1, 1])))

neurona = NOT()
print("\nNOT:")
print("NOT 0 =", neurona.Activacion(neurona.ProdVec([0])))
print("NOT 1 =", neurona.Activacion(neurona.ProdVec([1])))

neurona = IDENTITY()
print("\nIDENTITY:")
print("IDENTITY 0 =", neurona.Activacion(neurona.ProdVec([0])))
print("IDENTITY 1 =", neurona.Activacion(neurona.ProdVec([1])))

# %%

