# %%
from neurona import *


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
