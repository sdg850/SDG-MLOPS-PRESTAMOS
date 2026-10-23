"""Documented raw-input and response contracts for commercial renewal scoring."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class ClienteInput(BaseModel):
    """Las 19 claves raw son obligatorias, aunque sus valores pueden ser null.

    El pipeline imputa nulos con estado aprendido en train. No enviar IDs,
    etiqueta, mes de campaña, columnas log, dummies ni clusters. Se conservan
    saldos con signo, unidades originales y categorías nuevas. El ejemplo es
    inventado para probar el contrato; no acredita calidad predictiva.
    """

    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, strict=True,
        json_schema_extra={"examples": [{'LINEA_RENOVADO': 12000.0, 'PLAZO_RENOVADO': 24.0, 'USO_LINEA_TOTAL_TC_T2': 0.65, 'USO_TRIM_LINEA_BBVA': 0.4, 'NR_ENTIDADES_TOTAL_T2': 3.0, 'DIFF_NRO_ENTIDA_TOTALES_T2_T12': -1.0, 'SDO_CONSUMO_T2': 4500.0, 'RESENCIA_OFERTA_PLD_RENOVADO': 6.0, 'Ahorro_Sldo_Bco_T1': -25.0, 'PConsumo_Sldo_Bco_T1': 3000.0, 'SDO_BCO_tot_sm_pasivo_Bco_6M': 2500.0, 'EDAD': 35.0, 'ANTIGUEDAD_MES': 48.0, 'FLAG_LIMA_PROVINCIA': 1.0, 'SUELDO_ESTIMADO': 3500.0, 'CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD': 0.75, 'SEXO': 'F', 'EST_CIVIL': 'CATEGORIA_SINTETICA', 'REGION': 'REGION_SINTETICA'}]},
    )
    LINEA_RENOVADO: FiniteFloat | None = Field(
        ..., description='Monto de la línea ofrecida; conservar la unidad monetaria de la fuente.', examples=[12000.0],
    )
    PLAZO_RENOVADO: FiniteFloat | None = Field(
        ..., description='Plazo ofertado, expresado en meses.', examples=[24.0],
    )
    USO_LINEA_TOTAL_TC_T2: FiniteFloat | None = Field(
        ..., description='Uso de línea de tarjeta en el sistema financiero, corte T2. Conservar la escala original; no convertir a 0–1 ni 0–100.', examples=[0.65],
    )
    USO_TRIM_LINEA_BBVA: FiniteFloat | None = Field(
        ..., description='Uso trimestral de línea de tarjeta del banco. Conservar la escala de la fuente.', examples=[0.4],
    )
    NR_ENTIDADES_TOTAL_T2: FiniteFloat | None = Field(
        ..., description='Número de entidades financieras en el corte T2.', examples=[3.0],
    )
    DIFF_NRO_ENTIDA_TOTALES_T2_T12: FiniteFloat | None = Field(
        ..., description='Diferencia anual del número de entidades; puede ser negativa.', examples=[-1.0],
    )
    SDO_CONSUMO_T2: FiniteFloat | None = Field(
        ..., description='Saldo de consumo en el corte T2; conservar la unidad de la fuente.', examples=[4500.0],
    )
    RESENCIA_OFERTA_PLD_RENOVADO: FiniteFloat | None = Field(
        ..., description='Meses transcurridos desde la primera oferta. Nombre original conservado.', examples=[6.0],
    )
    Ahorro_Sldo_Bco_T1: FiniteFloat | None = Field(
        ..., description='Saldo de ahorro en el banco, corte T1. Se admiten saldos con signo como en la fuente.', examples=[-25.0],
    )
    PConsumo_Sldo_Bco_T1: FiniteFloat | None = Field(
        ..., description='Saldo de préstamo vigente en el banco, corte T1. Se admiten valores con signo.', examples=[3000.0],
    )
    SDO_BCO_tot_sm_pasivo_Bco_6M: FiniteFloat | None = Field(
        ..., description='Promedio semestral de deuda en el banco; admite valores con signo.', examples=[2500.0],
    )
    EDAD: FiniteFloat | None = Field(
        ..., description='Edad del cliente en la representación numérica de la fuente; no se impone un intervalo no documentado.', examples=[35.0],
    )
    ANTIGUEDAD_MES: FiniteFloat | None = Field(
        ..., description='Antigüedad del cliente expresada en meses.', examples=[48.0],
    )
    FLAG_LIMA_PROVINCIA: FiniteFloat | None = Field(
        ..., description='Indicador Lima/provincias en la codificación original. El diccionario no especifica el sentido de sus códigos.', examples=[1.0],
    )
    SUELDO_ESTIMADO: FiniteFloat | None = Field(
        ..., description='Sueldo estimado; moneda no especificada por el diccionario.', examples=[3500.0],
    )
    CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD: FiniteFloat | None = Field(
        ..., description='Porcentaje de deuda cubierta por la renovación; conservar la escala original.', examples=[0.75],
    )
    SEXO: str | None = Field(
        ..., description='Código de sexo de la fuente, sin recodificación manual.', examples=['F'],
    )
    EST_CIVIL: str | None = Field(
        ..., description='Código de estado civil de la fuente; catálogo completo no documentado.', examples=['CATEGORIA_SINTETICA'],
    )
    REGION: str | None = Field(
        ..., description='Categoría de región de residencia. Se aceptan categorías nuevas para el encoder entrenado.', examples=['REGION_SINTETICA'],
    )


class PrediccionOutput(BaseModel):
    """Score comercial y clasificación con el umbral persistido del modelo.

    No representa riesgo de impago ni una decisión de elegibilidad crediticia.
    """

    score_propension: Annotated[FiniteFloat, Field(
        ge=0, le=1, description="Score de propensión a renovar; no es una probabilidad calibrada.",
        examples=[0.42],
    )]
    flag_venta: Literal[0, 1] = Field(
        ..., description="Clase predicha de FLAG_VENTA usando el umbral guardado: 0 o 1.",
        examples=[1],
    )
    umbral: Annotated[FiniteFloat, Field(
        ge=0, le=1, description="Umbral seleccionado en validación y guardado con el modelo.",
        examples=[0.35],
    )]
    modelo: str = Field(..., description="Nombre del modelo cargado para inferencia.", examples=["modelo_sintetico"])


class HealthResponse(BaseModel):
    """Readiness HTTP 200 cuando el bundle está cargado; su ausencia devuelve 503."""

    status: Literal["ok"] = Field(..., description="Modelo disponible para atender inferencias.")
    modelo: str = Field(..., description="Nombre del modelo cargado.")
    version: str = Field(..., description="Versión del servicio API; no es la versión de MLflow Registry.", examples=["0.1.0"])
