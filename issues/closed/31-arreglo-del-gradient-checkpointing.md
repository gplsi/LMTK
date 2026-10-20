---
number: 31
title: "Arreglo del gradient checkpointing"
state: closed
labels:
---

Hemos arreglado un problema encotrado en el gradient checkpointing donde se recalculan las activaciones en una precisión mayor lo que provoca un error al comparar los elementos en diferentes precisiones. Para arreglarlo de forma consistente hemos hecho un wrapper alrededor del modelo que asegura que en caso de recomputar estos parámetros siempre se realizan en la precisión indicada en el archivo de configuración.

También hemos arreglado algunos métodos que se llamaban de forma errónea así como el nombre de algunos parámetros que han cambiado.