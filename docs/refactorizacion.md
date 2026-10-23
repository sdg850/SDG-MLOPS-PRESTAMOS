# Refactorización del caso Renovación de Préstamo

## Objetivo y alcance

El caso describe una campaña de marketing para identificar clientes con mayor propensión a aceptar la renovación de un préstamo. La etiqueta es `FLAG_VENTA` (`1`: renovación/venta observada; `0`: ausencia de venta). No es un modelo de impago, de solvencia ni de elegibilidad crediticia. Sus scores no constituyen decisiones de aprobación o denegación de préstamos.

Las fuentes revisadas son el notebook original de 186 celdas, el CSV separado por punto y coma, el diccionario Excel y la única diapositiva de la presentación. Los números de celda siguientes son índices **desde cero** del JSON del notebook original. La numeración visible en Jupyter puede ser diferente.

## Datos observados y discrepancias

- CSV: **87.556 filas, 22 columnas**, 87.556 clientes distintos y ningún registro duplicado completo.
- Etiqueta: 84.064 valores `0` y 3.492 valores `1` (**3,9883 %** de renovaciones).
- Campañas: nueve meses, enero–septiembre de 2015. La descripción del notebook y la diapositiva hablan de dos meses; se conserva el CSV completo y se documenta la discrepancia.
- El diccionario denomina `USO_TRIM_LINEA_Bco` a la variable de uso de tarjeta del banco; el CSV y el notebook utilizan `USO_TRIM_LINEA_BBVA`. El contrato de ejecución conserva el nombre real del CSV.
- Diez columnas tienen nulos. Los negativos se concentran en diferencia de entidades (25.367, potencialmente válidos), ahorro (4), préstamo vigente (1) y deuda media semestral (2). No se elimina globalmente toda observación negativa.
- La columna `Historia` del diccionario no especifica completamente el instante de disponibilidad de cada variable. Debe confirmarse que todas las variables de oferta y saldos estén disponibles **antes** de la llamada/venta. El nombre `RENOVADO` por sí solo no demuestra fuga de la etiqueta ni garantiza ausencia de ella.

Hash SHA-256 del CSV inspeccionado: `150479bd2718a023760cc1e39a41f68cee324d2c04ebe8c9bd8d0d69f13c1dab`.

## Problemas del notebook y tratamiento

| Celdas originales | Hallazgo verificable | Tratamiento |
|---|---|---|
| 53, 56, 59 | Imputación aleatoria a partir de media y desviación de todo el dataset, antes del split | Sustituir por imputación determinista aprendida solo en entrenamiento; registrar el cambio metodológico |
| 62, 64, 66, 68, 81, 87, 93 | Medianas y modas calculadas usando también las futuras particiones de evaluación | Ajustar imputadores dentro del pipeline de entrenamiento |
| 99 | `get_dummies` define las categorías a partir de todos los meses | Ajustar `OneHotEncoder` en train y soportar categorías nuevas en inferencia |
| 105–108 | `SelectKBest` usa todas las etiquetas y guía las tres variables elegidas para clustering; además incluye candidatos `MES` y `CLIENTE` | No repetir selección supervisada sobre test. Conservar las tres variables como configuración heredada documentada, sin afirmar nueva selección independiente |
| 110, 112, 115 | Búsqueda de K y KMeans entrenados sobre todos los meses | Ajustar clustering solo en train. K=3 es una configuración heredada, no un óptimo nuevamente demostrado |
| 112 | Silhouette completo repetido sobre unas 87.000 filas puede ser costoso por distancias entre pares | Mantenerlo fuera del entrenamiento obligatorio y limitar muestra si se usa para EDA |
| 128–129 | Escribe CSV con extensión `.xlsx` y lo presenta como Excel | Usar extensiones correspondientes al formato real de salida |
| 131 | Split aleatorio estratificado después de aprender el preprocesamiento y con campañas futuras mezcladas | Separar primero por `MES`: train enero–junio, validación julio–agosto, test septiembre |
| 133 | Desempaqueta `value_counts()` suponiendo orden y presencia de las dos clases | Validar la etiqueta y contar por valor explícito |
| 141 | El supuesto oversampling toma una muestra de la **clase 0 mayoritaria** del tamaño de la minoritaria, con reemplazo | Implementar oversampling real de la minoritaria hasta el tamaño de la mayoritaria; usar muestreo solo en train |
| 145 | Instalación de dependencias dentro de una celda de negocio | Declarar dependencias y extras fuera del notebook |
| 148 | SMOTE opera después de haber aprendido features con todo el dataset; one-hot interpolado puede producir categorías fraccionarias | Mantenerlo opcional y documentar limitación; no presentarlo como SMOTENC ni como requisito para obtener un baseline |
| 153–160, 167–174 | Se consulta repetidamente el mismo test para decidir familias/modelos | Comparar candidatos en validación y evaluar solo el ganador en test final |
| 158 | Imprime `accuracy * 10` como porcentaje | Calcular métricas dinámicamente y formatear porcentaje correctamente |
| 162, 174 | Tablas de comparación con constantes escritas manualmente | Generar comparación a partir de las predicciones de cada ejecución |
| 177 | `make_scorer(roc_auc_score)` usa la respuesta predeterminada `predict`, no scores/probabilidades | Calcular ROC-AUC con probabilidades de la clase positiva; añadir average precision para el desbalance |
| 177 | Repite gran parte de la preparación y realiza CV sobre datos ya preparados y submuestreados | Mantener preparación y muestreo encapsulados; el flag `--tune` usa GridSearchCV con folds temporales y ajusta ambos dentro de cada fold |
| 182 | Construye features one-hot, logs y cluster a mano; valores como `Ahorro_LOG=2093` no representan el log declarado | Predecir con registros crudos y transformar con el mismo objeto ajustado en entrenamiento |
| 184 | Predice con `model4`, aunque después se entrenó `best_rf_model` | Cargar el artefacto seleccionado y persistido, sin depender de variables de celdas previas |
| 185 | Interpreta score sin evidencia de calibración como probabilidad individual exacta | Identificarlo como score de propensión del modelo; no prometer calibración ni certeza |

## Mapa de responsabilidades

| Celdas / sección original | Destino | Responsabilidad |
|---|---|---|
| 0–2 | Notebook y `docs/diccionario.md` | Caso de negocio y significado de campos |
| 4–24 | `data.py`, `schema.py` y notebook | Ingestión, contrato y EDA agregado |
| 28–102 | `features.py` y notebook | Transformaciones persistibles; gráficos exploratorios fuera del flujo obligatorio |
| 104–127 | `features.py` y documentación | Variables heredadas del clustering y ajuste exclusivo con train |
| 128–148 | `data.py`, `modeling.py` | Split temporal y estrategias de balanceo dentro de fit |
| 152–178 | `modeling.py`, `evaluation.py`, `pipeline.py` | Entrenamiento, comparación, umbral y evaluación calculada |
| 180–185 | `predict.py` | Inferencia con datos crudos y artefacto completo |
| Estado implícito entre celdas | CLI y notebook delgado | Un punto de entrada reproducible, sin ejecutar celdas en un orden oculto |

## Cambios metodológicos deliberados

Esta entrega **no promete equivalencia numérica con las métricas originales**. Además de extraer módulos, cambia el protocolo para que la evaluación sea defendible:

1. Divide cronológicamente antes de aprender transformaciones: 60.511 filas de entrenamiento (enero–junio), 17.830 de validación (julio–agosto) y 9.215 de test (septiembre). `MES` dirige el split y `CLIENTE` permite verificar identidad, pero ninguno entra como predictor.
2. Aprende medianas, modas, categorías, escalas y centroides exclusivamente en entrenamiento. Una llamada `transform` o `predict` nunca reajusta esos estados.
3. Sustituye la imputación aleatoria por una política determinista y persistida. Los diez campos logarítmicos mantienen su intención, con tratamiento explícito de negativos previo a `log1p`.
4. Compara candidatos y elige su umbral con validación. La métrica principal de implementación es F1 positiva como compromiso reproducible; los costes de contacto y pérdida de oportunidad no fueron suministrados y pueden justificar otro criterio antes de una siguiente selección.
5. Reserva test para la evaluación final del candidato ya fijado. El reporte conserva la prevalencia y las métricas que no dependen del umbral. PR-AUC se reporta como **average precision**, sin confundirla con integración trapezoidal de la curva PR.
6. Persiste features y modelo juntos, con umbral, esquema y manifiesto. Las métricas y predicciones provienen de la ejecución, nunca de las tablas ilustrativas del notebook fuente.

Las tres variables usadas para KMeans y varias decisiones de transformación ya fueron exploradas sobre todo el CSV en el notebook histórico. El nuevo split evita nuevas fugas computacionales, pero no deshace ese conocimiento previo. Para una estimación prospectiva completamente independiente se necesitan campañas posteriores no utilizadas durante el desarrollo.

## Evidencia y límites

Las pruebas automatizadas deben cubrir aislamiento temporal, contrato de entrada, manejo de categorías no vistas, invariancia del estado del transformador al evaluar y paridad tras serialización. Los reportes de entrenamiento describen la ejecución concreta, no el despliegue productivo. La adaptación posterior de las unidades 01–06 añade DVC local, MLflow SQLite y serving local mediante comandos explícitos; ver `unidades_01_06.md`. No implementa serving público, calibración ni monitoreo de drift.

Los CSV originales, los registros de clientes y los modelos derivados no deben incorporarse a un repositorio público por defecto. Las pruebas usan fixtures sintéticos declarados como tales y no publican identificadores reales. El notebook de trabajo conserva EDA agregado y evita salidas con filas de clientes.
