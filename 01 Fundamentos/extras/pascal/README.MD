Rendimiento depende de la estructura:

---

### 1. **Listas de listas (triangulares)**

* Memoria: $O(n^2/2) \approx O(n^2)$ pero con **mitad** de entradas que la matriz cuadrada.
* Operaciones: solo calculas lo necesario.
* Velocidad: más rápida porque recorres menos celdas y no hay ceros sobrantes.

Ejemplo: para $n=1000$, el triángulo tiene

$$
\frac{1000\cdot 1001}{2} \approx 5\times 10^5 \quad \text{entradas}
$$

---

### 2. **Matriz n×n completa**

* Memoria: $O(n^2)$ pero ahora con **todos** los ceros extra.
* Operaciones: rellenas también los bordes de ceros.
* Velocidad: más lenta, porque se reservan más posiciones en memoria y Python itera sobre más elementos.

Ejemplo: para $n=1000$, la matriz tiene

$$
1000^2 = 10^6 \quad \text{entradas}
$$

el doble de lo necesario.

---

### Conclusión

* **Más rápido y eficiente** → listas triangulares.
* **Más simple para indexar con coordenadas fijas** → matriz cuadrada.

---

¿Quieres que te haga una **comparación real con `timeit`** en Python para medir los tiempos de ambas implementaciones?
