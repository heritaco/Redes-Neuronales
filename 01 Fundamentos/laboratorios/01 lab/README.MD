# Laboratorio 1: Neuronas McCulloch-Pitts y Compuertas Lógicas

## Objetivo final

Crear una clase en Python que modele neuronas del tipo McCulloch-Pitts y, a partir de ella, implementar compuertas lógicas básicas utilizando herencia de clase.

---

## Instrucciones

### 1. Clase base: `Neurona`

Crea una clase llamada `Neurona` con las siguientes características:

#### Parámetros del constructor `__init__`

* **n** (obligatorio): número entero que indique cuántas entradas tendrá la neurona.
* **pesos** (opcional): lista o arreglo de números reales de longitud `n`, que representa los pesos de conexión de cada entrada.

  * Si no se proporciona, se debe generar automáticamente una lista de `n` valores aleatorios entre 0 y 1.
* **umbral** (opcional): número real que representa el umbral de activación.

  * Si no se proporciona, deberá establecerse por defecto en `0`.

#### Atributos

* `n`: número de entradas de la neurona.
* `pesos`: lista de longitud `n` con los pesos de la neurona.
* `umbral`: número real correspondiente al umbral.

#### Métodos

* **`ProdVec(x)`**

  * Recibe un vector `x` de tamaño `n`.
  * Calcula el producto punto entre `x` y el vector interno de pesos.
  * Devuelve el valor de esa suma ponderada menos el umbral.

* **Función de activación**

  * Recibe un número real `r`.
  * Devuelve:

    * `1` si `r ≥ 0`
    * `0` en caso contrario.

* **Getters / Setters**

  * Consultar: `get_pesos()`, `get_umbral()`, `get_n()`.
  * Modificar: `set_pesos()`, `set_umbral()`.

---

### 2. Clases hijas: Compuertas lógicas

Crea las siguientes clases hijas de `Neurona`:

* `AND`
* `OR`
* `NOT`
* `IDENTITY`

Cada compuerta debe:

* Usar herencia de la clase `Neurona`.
* Definir en el `__init__` los pesos y el umbral necesarios para que se comporte como la compuerta lógica correspondiente.

---

## Documentación y estilo

* Documenta tu código con comentarios claros para cada clase y método.
* Incluye ejemplos de uso al final del archivo, como la impresión de tablas de verdad de las compuertas.

---

## Entregable

Guarda tu archivo con el siguiente formato:

```
Lab1_Apellido1_Apellido2_Apellido3_V#
```

Donde `#` es el número de versión de tu entrega.

Ejemplo:

```
Lab1_Maravillo_Gómez_V1.py
```

