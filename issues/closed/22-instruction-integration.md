---
number: 22
title: "Instruction integration"
state: closed
labels:
---

Hemos integrado la instrucción como una subtarea de la tokenización, se han realizado pruebas con GPT-2 sobre los datasets que ya se estaban utilizando para hacer las instrucciones (en formato JSON) y se ha tokenizado correctamente.

Hemos añadido las dos variantes de tokenización que hay para las instrucciones, poniendo como default la más usada (es la opción que puede ver los tokens de la instrucción pero no los tiene en cuenta para el cálculo de la pérdida), mientras que la otra opción también está disponible (sin ver los tokens de la instrucción, de acuerdo a los valores de la máscara de atención).

Por otro lado, hemos añadido también el padding dinámico, permitiendo seleccionar secuencia máxima de padding y adaptarla al mayor tamaño de secuencia encontrado de entre todos los textos de entrada, en lugar de hacerlo en base a un valor fijado por el usuario.

De paso, hemos eliminado los archivos antiguos de la tarea tokenization_instruction, lo cual incluye la carpeta entera de la tarea así como el schema específico y sus ejemplos en yaml.