"""Local commercial-propensity API using the same raw-input inference path.

Only load artifacts created by a trusted training process: joblib uses pickle.
The service predicts renewal propensity and never lending eligibility.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from loan_renewal.predict import load_artifact, predict_frame
from loan_renewal.config import MODELS_DIR
from .schemas import ClienteInput, HealthResponse, PrediccionOutput

LOGGER = logging.getLogger(__name__)


def _model_name(bundle: dict) -> str:
    manifest = bundle.get("manifest", {})
    if isinstance(manifest, dict):
        for key in ("winner", "selected_model", "model_name"):
            value = manifest.get(key)
            if isinstance(value, str) and value:
                return value
    return type(bundle["pipeline"]).__name__


def create_app(model_path: str | Path | None = None) -> FastAPI:
    """Load one trusted bundle during lifespan; keep failed readiness diagnosable.

    An explicit path takes precedence over MODEL_PATH. Relative paths are
    relative to the invocation's working directory, documented as repo root.
    There is no hot reload, training, resampling or model upload endpoint.
    """
    artifact_path = Path(model_path or os.environ.get("MODEL_PATH", str(MODELS_DIR / "model.joblib")))

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.bundle = None
        try:
            application.state.bundle = load_artifact(artifact_path)
        except Exception as error:
            # Do not log artifact internals, data, filesystem paths or exception
            # messages. Readiness remains 503 until an operator fixes/restarts.
            LOGGER.warning("Model unavailable at startup (%s).", type(error).__name__)
        yield
        application.state.bundle = None

    application = FastAPI(
        title="Loan Renewal Propensity API",
        description="Educational commercial propensity scoring. No loan approval or rejection decisions.",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.bundle = None

    @application.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, error: RequestValidationError):
        # FastAPI's default validation body includes the offending input. Return
        # only error types and static schema fields, never submitted values.
        known_fields = set(ClienteInput.model_fields)
        issues = []
        for issue in error.errors():
            location = ["body"]
            for component in issue.get("loc", ())[1:]:
                if component in known_fields:
                    location.append(component)
                elif isinstance(component, int):
                    location.append(component)
                else:
                    location.append("unknown_field")
            issues.append({"loc": location, "type": issue.get("type", "invalid_input")})
        return JSONResponse(status_code=422, content={"detail": issues})

    @application.get("/", tags=["Info"])
    def root():
        return {
            "api": "Loan renewal commercial propensity",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/health",
            "prediction": "POST /predecir",
        }

    @application.get("/health", response_model=HealthResponse, tags=["Readiness"])
    def health(request: Request):
        bundle = request.app.state.bundle
        if bundle is None:
            raise HTTPException(status_code=503, detail="Model unavailable.")
        return {"status": "ok", "modelo": _model_name(bundle), "version": "0.1.0"}

    @application.post("/predecir", response_model=PrediccionOutput, tags=["Propensity"])
    def predecir(cliente: ClienteInput, request: Request):
        bundle = request.app.state.bundle
        if bundle is None:
            raise HTTPException(status_code=503, detail="Model unavailable.")
        try:
            result = predict_frame(bundle, pd.DataFrame([cliente.model_dump()])).iloc[0]
            return PrediccionOutput(
                score_propension=float(result["probabilidad_renovacion"]),
                flag_venta=int(result["prediccion_flag_venta"]),
                umbral=float(result["umbral"]),
                modelo=_model_name(bundle),
            )
        except ValueError:
            # Contract failures become 422 without raw inputs or model internals.
            raise HTTPException(status_code=422, detail="Input is incompatible with the saved model contract.") from None
        except Exception as error:
            LOGGER.error("Inference failed (%s).", type(error).__name__)
            raise HTTPException(status_code=500, detail="Inference unavailable.") from None

    return application


app = create_app()
