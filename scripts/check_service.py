"""Check the live local API against the offline artifact using synthetic inputs."""
import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from loan_renewal.predict import load_artifact, predict_frame
from loan_renewal.tracking import synthetic_probe

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--model", type=Path, default=Path("models/model.joblib"))
args = parser.parse_args()

with urlopen(args.url + "/health", timeout=10) as response:
    assert response.status == 200
probe = synthetic_probe()
offline = predict_frame(load_artifact(args.model), probe)
for i, row in enumerate(probe.to_dict(orient="records")):
    request = Request(args.url + "/predecir", data=json.dumps(row).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=10) as response:
        result = json.load(response)
    np.testing.assert_allclose(result["score_propension"], offline.iloc[i]["probabilidad_renovacion"], rtol=1e-12, atol=1e-12)
    assert result["flag_venta"] == int(offline.iloc[i]["prediccion_flag_venta"])
    assert result["umbral"] == float(offline.iloc[i]["umbral"])
bad = probe.iloc[0].to_dict()
bad.pop("EDAD")
try:
    urlopen(Request(args.url + "/predecir", data=json.dumps(bad).encode(),
                    headers={"Content-Type": "application/json"}, method="POST"), timeout=10)
    raise AssertionError("Missing required field did not return HTTP 422")
except HTTPError as error:
    assert error.code == 422
print("API_OK: readiness 200, synthetic offline/HTTP parity, missing-field 422")
