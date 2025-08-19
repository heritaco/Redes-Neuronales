Objetivo final: Crear una clase en Python que modele neuronas del tipo McCulloch-Pitts y, a partir de ella, implementar compuertas lógicas básicas utilizando herencia de clase.





Instrucciones:

1.- Clase base: Neurona.

Crea una clase llamada Neurona con las siguientes características:

Parámetros del constructor (__init__):
n (obligatorio): número entero que indique cuantas entradas tendrá la neurona.
pesos (Opcional) Una lista o arreglo de números reales de longitud n, que representa los pesos de conexión de cada entrada:
Si no se proporciona, se debe generar automáticamente una lista de n valores aleatorios entre 0 y 1.
umbral: (Opcional) Un número real, que representa el umbral de activación.
Si no se proporciona, deberá establecerse por defecto en 0.
Establece el tipo de dato aceptado para cada parámetro de entrada.
Atributos:
n: número de entradas de la neurona.
pesos: lista de longitud n con los pesos de la neurona.
umbral: número real correspondiente al umbra.
Métodos:
ProdVec(x):
Recibe un vector x de tamaño n
Calcula el producto punto entre x y el vector interno de pesos interno.
Devuelve el valor de esa suma ponderada menos el umbral.
Función de activación:
Recibe un número real r.
Devuelve: 1 si ﻿r mayor o igual que 0﻿;  0 en caso contrario.
Getter / Setter:
Métodos para consultar (get_pesos, get_umbral, get_n) y modificar (set_pesos, set_umbral) los atributos de la neurona.


2.- Clases hijas: Compuertas lógicas.

Crea las siguientes clases hijas de Neurona: AND, OR, NOT, IDENTITY.

Usa las propiedades de herencia para que cada compuerta sea una clase hija de Neurona.
En el constructor (__init__) de cada clase, define explícitamente los pesos y el umbral necesarios para que la neurona se comporte como esa compuerta lógica.


Documentación y estilo:

Documenta tu código con comentarios claros para cada clase y método.
Incluye ejemplos de uso en la parte final del archivo (por ejemplo, imprimir las tablas de verdad de las compuertas).




Entregable:

Guarda tu archivo con el siguiente formato:
Lab1_Apellido1_Apellido2_Apellido3_V#
Donde # es el número de versión de tu entrega.
Ejemplo: Lab1_Maravillo_Gómez_V1.py