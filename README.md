# Proyecto de Paralelización de Funciones en Python

Este proyecto tiene como objetivo paralelizar diversas funciones en Python utilizando técnicas como la vectorización, el pooling de procesos, el balanceo de carga y la gestión de dependencias de datos. El README proporciona una explicación detallada de cada función paralelizada y las técnicas utilizadas, así como los tiempos de ejecución finales y las mejoras conseguidas.

## Índice

1. [Función: Generar Datos](#función-generar-datos)
2. [Función: Procesar Datos 1](#función-procesar-datos-1)
3. [Función: Procesar Datos 2](#función-procesar-datos-2)
4. [Función: Procesar Datos 3](#función-procesar-datos-3)
5. [Función: Procesar Datos 4](#función-procesar-datos-4)
6. [Función: Procesar Datos 5](#función-procesar-datos-5)
7. [Tiempos de Ejecución Finales](#tiempos-de-ejecución-finales)
8. [Optimización de Funciones](#optimización-de-funciones)

---

## Función: Generar Datos

En esta función se generan bloques de datos de forma paralela utilizando un pooling de procesos. Se implementa un balanceo de carga y se gestionan las dependencias de datos para garantizar una generación ordenada de los bloques. Los bloques generados se concatenan en un array de NumPy para permitir operaciones SIMD eficientes en funciones posteriores.

## Función: Procesar Datos 1

Esta función utiliza instrucciones SIMD (Single Instruction, Multiple Data) mediante la librería NumPy para operaciones vectoriales. Aunque no se implementa paralelismo a nivel de procesos, la vectorización permite una ejecución eficiente de las operaciones en datos divididos en varios bloques.

## Función: Procesar Datos 2

Se emplea el pooling de procesos para paralelizar el procesamiento de bloques. Se aplican técnicas de balanceo de carga, dependencias de datos y vectorización para mejorar la eficiencia del procesamiento paralelo. Cada bloque se procesa secuencialmente, asegurando la correcta gestión de dependencias y una ejecución eficiente.

## Función: Procesar Datos 3

Esta función utiliza un pooling de procesos con dependencias de datos (semillas). Se crea un array de multiprocessing para compartir resultados entre procesos, evitando la necesidad de reordenar los resultados. Cada proceso almacena sus resultados en la posición adecuada del array compartido, basándose en los índices de los bloques.

## Función: Procesar Datos 4

Se emplea nuevamente el pooling de procesos para procesar bloques en paralelo. La división de datos en bloques se realiza utilizando generadores, permitiendo que cada proceso procese su bloque de forma independiente. Se garantiza una distribución equitativa de la carga de trabajo entre los procesos para una ejecución eficiente.

## Función: Procesar Datos 5

Se divide el conjunto de datos en bloques más pequeños para permitir una mayor paralelización. Se utilizan pooling de procesos, balanceo de carga y vectorización para mejorar el rendimiento del procesamiento paralelo. Se aplican operaciones vectorizadas a cada bloque simultáneamente, lo que resulta en una mejora significativa del rendimiento.

## Tiempos de Ejecución Finales

Se proporcionan los tiempos de ejecución finales de cada función en la versión paralelizada, demostrando las mejoras de rendimiento obtenidas mediante las técnicas de paralelización y optimización implementadas.

## Optimización de Funciones

Se ofrece una explicación detallada de las optimizaciones aplicadas a cada función, como la vectorización, el balanceo de carga y la división de datos en bloques. Se describen las mejoras en los tiempos de ejecución conseguidas mediante estas optimizaciones.

---

Este README proporciona una visión general del proyecto, explicando las técnicas utilizadas y las mejoras conseguidas en cada función paralelizada. Para más detalles sobre cada función y sus implementaciones, se recomienda consultar el código fuente y la memoria del proyecto.
