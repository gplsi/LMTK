---
number: 15
title: "Tutorials"
state: closed
labels:
---

Se han probado las tres tareas que tenemos hasta ahora (tokenización, clm_training y publish) con GPT-2, dando resultados satisfactorios en local. 

Hemos añadido unos parámetros nuevos en los schemas de training para poder manejar mejor el guardado de los checkpoints.

Hemos corregido algunos errores en las llamadas de las funciones del orchestrator para tokenización.

También he corregido el makefile que hace referencia a un volumen de la carpeta de Ernesto, aunque creo que esto ya estará arreglado en master.

Hemos añadido la carpeta de tutoriales, que contiene un readme y un jupyter notebook por tarea, tengo que correrlos aún, pero es una carpeta paralela a todo el framework, por lo que no debería de haber problema.