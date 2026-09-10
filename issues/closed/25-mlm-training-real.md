---
number: 25
title: "Mlm training real"
state: closed
labels:
---

· Se ha implementado y probado exhaustivamente la tokenización para instrucción, siendo implementada de forma agnóstica para diferentes subcarpetas del dataset de instrucción seleccionado y correctamente formateado con los tokens específicos de cada modelo de lenguaje instruido.

· Se ha añadido y probado la tokenización para mlm.

· Se ha añadido la tarea de entrenamiento para mlm.

· Hemos añadido un pequeño script para poder evaluar rápidamente los datasets tokenizados dentro de /scripts

· Se ha comprobado que el padding dinámico espeífico de instrucción funciona correctamente

· Se ha eliminado código legacy que quedaba en desuso y que no se volverá a utilizar de las implementaciones de instrucción

· Se han establecido subclases dentro de la tokenización en función de la tarea a realizar, la definición de los archivos de configuración se mantiene igual que antes.

· Se han establecido subclases de entrenamiento para que, cuando desarrollemos el código en estas áreas, sea más fácil de mantener; los archivos de configuración siguen como antes.

· Se ha reestructurado los schemas definidos, haciendo que se dividan por compoenentes que sean más fáciles de mantener y extender, la funcionalidad de los archivos de configuración se mantiene como antes, por lo que el usuario final no tendrá ningún problema en seguir trabajando con estos cambios

· Para todos los casos de entrenamiento y tokenización hemos añadido los archivos de configuración de ejemplo dentro de config/examples.