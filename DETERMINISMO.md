# LMTK: copia con determinismo estricto para CPT

Base: `gplsi/LMTK`, rama `hpc`, commit
`65e8ddc45c746609c55310ce27096ed9b6568767`.

**Estado: controles implementados; repetibilidad en GPU pendiente de verificar.**
No se ha ejecutado un entrenamiento real en esta entrega: el entorno de edición
no tiene PyTorch ni GPU y la instalación de dependencias fue bloqueada por la
aprobación de red. Las pruebas locales cubren lógica y controles con dobles de
prueba; no certifican los kernels de tu clúster.

## Qué cambia

Se conserva el modelo, BF16, FSDP, cuatro workers, la estructura del proyecto y
el flujo de SLURM. `clm_training` activa por defecto `strict_determinism: true`.

1. Se exige una seed válida y hashing de Python fijo desde el arranque.
2. Se configura cuBLAS antes de inicializar CUDA; PyTorch exige algoritmos
   deterministas con `warn_only=False`. Se desactiva TF32 y el backend SDPA
   memory-efficient. La atención se fija a SDPA nativo de PyTorch; Flash nativo
   puede seguir utilizándose si soporta la ejecución determinista. No se cambia
   silenciosamente a FP32 ni a otra arquitectura.
3. El sampler recibe la seed explícitamente, incluso con FSDP; los workers
   tienen un generador independiente por split/rank. Las épocas tienen un orden
   reproducible. Esto puede cambiar el orden respecto al sampler antiguo.
4. Cada rank guarda versiones, dispositivo, fingerprint del dataset, revisión
   resuelta del modelo y configuración relevante en un JSON de reproducibilidad.
5. Se rechaza `checkpoint`/`initial_weights_checkpoint` en modo estricto. Esta
   copia soporta nuevas ejecuciones desde `model_name`, que es tu caso, no una
   promesa de reanudación exacta del entrenador original.

Las correcciones de optimización están separadas en `patches/02-optimization.patch`:
se normalizan todos los microbatches, se procesa el grupo incompleto de final
de época, se ajusta el scheduler al número real de actualizaciones/datos y se
elimina el clipping doble. **Estas correcciones cambian el algoritmo respecto
a tus experimentos anteriores.** No atribuyas un cambio de F1 solo al determinismo.
`patches/01-strict-determinism.patch` contiene los controles de reproducibilidad.
Ambos cambios ya están aplicados en esta carpeta: no vuelvas a aplicar los patches.

`state_dict_type` continúa siendo `full` en el constructor FSDP original. No se
ha reestructurado la configuración general de estrategias. El comparador acepta
esos checkpoints completos `.pth`, no checkpoints distribuidos fragmentados.

## Uso con tu configuración

Usa el mismo contenedor y las mismas versiones que en el clúster. No actualices
las dependencias durante una comparación. Se incluye
`config/examples/random8_deterministic.yaml`, con tus parámetros y una carpeta
de salida nueva. Comprueba la ruta local del dataset antes de ejecutarlo.

Desde la raíz de esta copia, en una asignación GPU ya obtenida:

```bash
bash scripts/run_deterministic.sh --config config/examples/random8_deterministic.yaml
```

Para SLURM usa el `slurm/submit_job.sh` existente con tus opciones habituales y
`-c config/examples/random8_deterministic.yaml`. `slurm/p.slurm` ya exporta las
variables necesarias. Si usas otro lanzador/contenedor, asegúrate de que dentro
del contenedor, **antes de arrancar Python**, están:

```bash
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

La seed de entrenamiento sigue siendo 42. El 0 del hashing de Python es otro
control independiente. `PYTHONHASHSEED` no debe establecerse por primera vez
desde dentro de un intérprete Python ya arrancado.

## Prueba pequeña antes del CPT de 3B

Dentro del contenedor, desde la raíz de esta copia:

```bash
python scripts/prepare_determinism_smoke.py determinism-smoke
```

Se crea una única inicialización local de un Llama pequeño y un único dataset
tokenizado, sin descargar modelos. Se crean `run-a.yaml` y `run-b.yaml`, iguales
salvo la carpeta de salida, con BF16, FSDP, seed 42 y acumulación 16.

Ejecuta las dos configuraciones con el mismo número/tipo de GPU y el mismo
lanzador que vayas a usar para el experimento real. En una asignación GPU local:

```bash
bash scripts/run_deterministic.sh --config determinism-smoke/run-a.yaml
bash scripts/run_deterministic.sh --config determinism-smoke/run-b.yaml
python scripts/compare_cpt_runs.py determinism-smoke/run-a determinism-smoke/run-b
```

En SLURM entrega cada YAML al lanzador existente; no se envía ningún trabajo
automáticamente. Para certificar FSDP necesitas al menos dos GPU: una sola GPU
usa la estrategia automática original. No cambies el número de GPU entre runs.

El comparador exige los mismos manifests y el mismo conjunto de checkpoints.
Compara bit a bit pesos y estados de optimizador/scheduler, además de contadores;
rechaza valores no finitos y sale con error si hay diferencias. No compara los
bytes de los archivos `.pth` completos ni los tiempos/logs de WandB.

Si funciona, repite una prueba corta con **tu Llama-3.2-3B, tus secuencias de 8192,
tu dataset y tu topología real**, por ejemplo usando el mismo `train_data_ratio`
pequeño en dos copias del YAML, dos épocas y distintas salidas. El Llama pequeño
no certifica todos los kernels usados por un modelo mayor. Una prueba corta
tampoco certifica matemáticamente una ejecución de duración arbitraria; verifica
los checkpoints completos cuando repitas los CPT definitivos.

Si PyTorch informa de una operación no determinista, conserva el error y la
versión exacta para resolver ese kernel. No pongas `warn_only=True` para saltarlo.
Si el comparador detecta diferencias, la prueba **no ha pasado**, aunque los F1
coincidan. Si falla el cargador seguro del checkpoint, no habilites pickle
arbitrario: revisa el formato guardado en tu versión.

## Condiciones de la comparación

Mantén iguales: datos tokenizados y su orden, pesos iniciales, configuración,
modelo de GPU y topología, número de procesos, contenedor, PyTorch/Lightning/
Transformers/CUDA/cuDNN y parámetros de ejecución. Puedes fijar
`model_revision` al SHA de Hugging Face que usaste (se registra `model_commit`),
o usar una copia local inmutable. El fingerprint del dataset es metadata útil,
no sustituye un checksum criptográfico de todos sus archivos.

No existe garantía universal entre hardware o versiones diferentes. Los
controles estrictos detectan operaciones conocidas por PyTorch; no son una
prueba formal de ausencia de errores en todas las bibliotecas externas.

## Verificación local y entrega

```bash
python tests/unit/training/test_deterministic_copy.py
```

Estas pruebas no requieren instalar PyTorch y no son un entrenamiento GPU.
El resultado y los límites están documentados en `issues/deterministic-cpt.md`.

El ZIP contiene el árbol completo y el historial Git disponible del snapshot
original. Los cambios nuevos quedan sin commit, para que los revises con
`git diff`. El remoto original se conserva como `upstream`, con push desactivado.
No se ha creado ni modificado un repositorio remoto de GitHub.

Para publicar la copia, crea un repositorio vacío en tu cuenta y, tras revisar
los cambios, añade su URL como `origin`, crea tu commit y sube la rama
`deterministic`. No se han incluido datos, pesos ni credenciales nuevos.
