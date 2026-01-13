---
number: 23
title: "MLM training, Instruction Fixing, training schemas re-estructure and training task reestructure"
state: closed
labels:
---

Se han realizado diferentes cambios en la rama:

· Integración de la tarea de tokenización para masked language models, con la idea de poder realizar continual pretraining de modelos discriminativos como BERT con enmascaramiento. 

· Añadida la funcionalidad de entrenar estos modelos, al igual que entrenamos los modelos de lenguaje causales.

· Reestructuración de los schemas: hacer los esquemas de yaml puramente jerárquicos provoca que al añadir nuevas tareas de entrenamiento más allá de causal, se generen una gran cantidad de archivos de esquemas, haciendo más engorroso el mantenimiento de los mismos. En lugar de eso, optamos por emplear tanto jerarquía como composición de los diferentes esquemas para hacerlo más fácil de mantener. Así, por ejemplo, los diferentes tipos de tokenización heredan tanto del esquema base como del esquema base de tokenización. No obstante, los esquemas de entrenamiento tienen un archivo específico según el entrenamiento pero cuentan con diferentes componentes para poder formarlos en su totalidad (data, model, optimizer, scheduler, training_args), logging y estrategy. De esta forma, los archivos de configuración se siguen formando igual que antes pero resulta mucho más sencillo ampliar los esquemas bases de los que parten. Solo se ha modificado la organización de los esquemas de entrenamiento.

· Reestructuración de la tarea de "entrenamiento" como tarea general que engloba cualquier tipo de entrenamiento. En un inicio, solo teníamos la tarea de clm_training, pero ahora contamos con la posibilidad de hacer instrucción y también entrenamiento enmascarado. Para lidiar con este problema, he optado por definir una única tarea de entrenamiento y separar entre los "Modelos" y los "Trainers". "Modelos" define los modelos como clases de LightningModule de acuerdo a los requisitos de fabric y lightning, llamando a su respectiva clase de HuggingFace y estarán encapsulados dentro de los "Trainers". De esta forma, si necesitamos definir un modelo nuevo que requiera una clase nueva de huggingface, solo tenemos que crear una pequeña clase en "Modelos" y actualizarla. El trainer contiene el que utilizabamos hasta ahora para Fabric, solo que carga el modelo de un diccionario (MODEL_CLASS_MAP) en lugar de estar fijo para seleccionar clm_training.

· He generado un script dentro de /scripts que está pensado para poder evaluar la calidad de la tokenización realizada, independientemente del subtipo de tokenización que hemos llevado a cabo. Este script toma como valores de entrada la carpeta de un dataset tokenizado, el número de ejemplos que queremos visualizar y de forma opcional el tokenizador del modelo. Como salida genera un JSON con los input_ids, attention_mask y labels de una muestra del número de ejemplos seleccionados. Si el tokenizador se ha proporcionado además de esto veremos cada uno de los token_ids decodificados, lo que resulta especialmente útil para identificar si los tokens corresponden con lo esperado.

Por último decir que todos los cambios realizados se han probado en local a través de las configuraciones que podemos encontrar en config/examples, llegando a validar la tokenización de instrucción, clm y mlm, así como el entrenamiento causal y enmascarado.

No se han realizado pruebas con slurm todavía.