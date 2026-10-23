# Validación en Codespaces — 22 de octubre de 2026

Separación de responsabilidades en la rama refactor/cookiecutter. Checkpoint de implementación: 7bbaf64.

| Carpeta | Responsabilidad verificada |
|---|---|
| pipelines/ | Scripts 01–06, run_pipeline.py y run_training.py; reutilizan el núcleo |
| api/ | FastAPI en app.py y contratos HTTP en schemas.py |
| mlruns/ | Historial SQLite y artefactos privados de MLflow |
| src/loan_renewal/ | Datos, features, modelos, evaluación, inferencia y lógica compartida |

| Comprobación | Resultado |
|---|---|
| Pruebas | 34 aprobadas, sin omisiones; incluyen API y MLflow reales, scripts desde otro directorio y migración del historial |
| Notebook | 18 celdas ejecutadas en Python 3.12 sin errores; usa api.app y registra en mlruns |
| Wheel | Paquete construido e instalado en un destino aislado; imports de api y loan_renewal verificados desde fuera del repositorio |
| Historial MLflow | Copia de respaldo antes de moverlo; conteos de filas conservados y rutas absolutas actualizadas |
| Registry existente | LoanRenewal-Propensity, versión 1 y alias candidato recargados desde la nueva ubicación con predicciones idénticas |
| Docker/API | Nueva imagen con api.app:app saludable; readiness 200, paridad offline/HTTP y campo ausente 422 |
| DVC | Reproducción desde pipelines/run_training.py completada; segundo repro omitido sin cambios; estado actualizado |
| Fuentes originales | Los cuatro hashes coinciden con docs/fuentes.json |

## Metadatos de instalación

Se verificaron las copias generadas src/sdg_loan_renewal.egg-info y sdg_loan_renewal.egg-info. Los metadatos instalados están disponibles en .venv/lib/python3.12/site-packages/sdg_loan_renewal-0.1.0.dist-info. Se retiraron las copias redundantes del árbol de trabajo y se verificaron en un proceso nuevo los imports, la versión y el comando loan-renewal check-env --include-dev desde otro directorio.

Setuptools puede regenerar egg-info al construir o reinstalar. Esos archivos se excluyen de Git y del explorador. Los metadatos activos de la distribución instalada se conservan.

## Persistencia

La migración mantiene SQLite y el historial existente; la estructura interna de experimentos y Registry la gestiona MLflow. No se crean directorios de versiones ficticios ni se cambia a otro backend. scripts/migrate_tracking.py rechaza un destino con estado existente y conserva una copia antes de modificar rutas.

El notebook original, diccionario y presentación permanecen en references/original; el CSV en data/raw. Los modelos, logs y resultados completos permanecen fuera de Git. DVC conserva caché local y metadatos; no se configuró un remoto ni se publicaron datos. La validación técnica no acredita calidad comercial ni aprobación productiva.
