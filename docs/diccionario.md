# Contrato de datos de Renovación de Préstamo

Fuente semántica: `Diccionario Renovacion prestamo.xlsx`, hoja 1. Fuente de nombres y tipos observados: `Dataset Renovacion_prestamo.csv`, 87.556 registros separados por `;`. Se conservan los nombres crudos para ingestión e inferencia; las columnas derivadas no forman parte de la entrada del usuario.

| Columna real del CSV | Significado del diccionario | Tipo observado | Uso |
|---|---|---|---|
| `MES` | Mes de campaña (`YYYYMM`) | entero | Orden temporal; excluido de predictores |
| `CLIENTE` | Código de cliente | entero | Identidad/validación; excluido de predictores |
| `FLAG_VENTA` | Variable de venta | entero binario 0/1 | Etiqueta; excluida de predictores |
| `LINEA_RENOVADO` | Monto de línea ofrecida | entero | Numérico con log1p |
| `PLAZO_RENOVADO` | Número de meses ofertados | entero | Numérico |
| `USO_LINEA_TOTAL_TC_T2` | Porcentaje de uso de línea de tarjeta del sistema financiero | decimal nullable | Numérico con log1p |
| `USO_TRIM_LINEA_BBVA` | Porcentaje de uso de línea de tarjeta del banco | decimal nullable | Numérico con log1p; alias documental `USO_TRIM_LINEA_Bco` |
| `NR_ENTIDADES_TOTAL_T2` | Número de entidades | entero | Numérico |
| `DIFF_NRO_ENTIDA_TOTALES_T2_T12` | Diferencia del número de entidades en el último año | entero con signo | Numérico; conservar negativos |
| `SDO_CONSUMO_T2` | Monto de saldo de consumo | decimal nullable | Numérico con log1p |
| `RESENCIA_OFERTA_PLD_RENOVADO` | Meses transcurridos desde la primera oferta | decimal nullable | Numérico |
| `Ahorro_Sldo_Bco_T1` | Monto de ahorro en el banco | entero con signo | Numérico con log1p tras política de negativos |
| `PConsumo_Sldo_Bco_T1` | Monto de préstamo vigente en el banco | entero con signo | Numérico con log1p tras política de negativos |
| `SDO_BCO_tot_sm_pasivo_Bco_6M` | Monto del promedio semestral de deuda | decimal con signo | Numérico con log1p tras política de negativos |
| `EDAD` | Edad del cliente | decimal nullable | Numérico |
| `SEXO` | Sexo del cliente | texto nullable | Categórico |
| `EST_CIVIL` | Estado civil | texto nullable | Categórico |
| `ANTIGUEDAD_MES` | Meses de antigüedad como cliente | decimal nullable | Numérico con log1p |
| `REGION` | Región de residencia | texto nullable | Categórico |
| `FLAG_LIMA_PROVINCIA` | Flag Lima/provincias | entero | Numérico; el diccionario no especifica el sentido 0/1 |
| `SUELDO_ESTIMADO` | Monto de sueldo | decimal nullable | Numérico con log1p |
| `CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD` | Porcentaje de deuda cubierta con la renovación | decimal | Numérico con log1p |

## Observaciones del contrato

- El diccionario no define la moneda de los montos ni garantiza si los porcentajes se expresan como proporciones o valores 0–100. No convertir unidades silenciosamente.
- `MES` es un código de mes, no una fecha diaria. El dataset contiene enero a septiembre de 2015, aunque la presentación describe dos meses.
- La columna `Historia` usa `Mes_t2` para muchos campos, `Anual` para diferencia de entidades y `Semestral` para deuda media. Varias filas no tienen historia documentada. Confirmar el corte de observación y la disponibilidad precontacto antes de uso operativo.
- Los códigos de `EST_CIVIL` no tienen traducción completa documentada; conservarlos como categorías. No asignar significados a `X` o `Y` por inferencia.
- Los nulos numéricos y categóricos se imputan con estado aprendido en entrenamiento. Una columna requerida ausente es un error de esquema, no un valor cero implícito.
- La inferencia recibe las 19 variables predictoras crudas. No requiere `FLAG_VENTA`, `MES`, `CLIENTE`, columnas `_LOG`, dummies ni `Cluster`; el pipeline deriva sus propias features.
- Las categorías nuevas se codifican sin cambiar el ancho de la matriz. Ello evita un fallo técnico, pero no acredita calidad sobre categorías desconocidas.

## Conteos de nulos del CSV inspeccionado

| Campo | Nulos |
|---|---:|
| `USO_LINEA_TOTAL_TC_T2` | 19.192 |
| `USO_TRIM_LINEA_BBVA` | 19.192 |
| `RESENCIA_OFERTA_PLD_RENOVADO` | 13.856 |
| `SDO_CONSUMO_T2` | 5.186 |
| `SUELDO_ESTIMADO` | 37 |
| `REGION` | 18 |
| `EDAD` | 1 |
| `SEXO` | 1 |
| `EST_CIVIL` | 1 |
| `ANTIGUEDAD_MES` | 1 |

Estos conteos describen el archivo fuente inspeccionado. La validación de cada ejecución recalcula sus propios resultados.

## Contrato HTTP documentado

`api/schemas.py` publica en `/docs` las descripciones de los 19 campos raw y un
JSON completo inventado. Todas las claves son obligatorias, aunque se admite
`null`; un campo ausente devuelve 422. Se rechazan campos adicionales, números
no finitos y texto usado como número. Los nulos se imputan en el pipeline.

El contrato conserva categorías nuevas y saldos con signo. No impone límites
arbitrarios de edad, moneda, escala porcentual ni un catálogo categórico que el
diccionario no documenta. Los ejemplos solo verifican el contrato técnico.

`PrediccionOutput` restringe score y umbral a [0, 1] y la clase a 0/1. El score
no está calibrado. `HealthResponse` documenta `/health` con estado, nombre de
modelo y versión del servicio API (distinta de la versión de MLflow Registry).
El endpoint mantiene HTTP 503 si no se pudo cargar el modelo.
