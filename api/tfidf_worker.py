"""Persistent TF-IDF inference worker.

Runs in a separate Python environment with scikit-learn 1.6.1.
Communicates with the FastAPI process through JSON lines.
"""

import json
import os
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import sklearn
from sklearn.exceptions import InconsistentVersionWarning


MODEL_ROOT = Path(
    os.environ.get(
        "EVIDENCITE_TFIDF_ROOT",
        "/opt/evidencite-models",
    )
)

MODEL_IDS = {
    "tfidf-a": "tfidf_A",
    "tfidf-b": "tfidf_B",
}


def load_models():
    if sklearn.__version__ != "1.6.1":
        raise RuntimeError(
            "TF-IDF requires scikit-learn 1.6.1; "
            f"found {sklearn.__version__}"
        )

    # Keep stdout exclusively for JSON responses.
    with redirect_stdout(sys.stderr):
        from src.config.schema import InputRepresentationConfig
        from src.modeling.predict import TfidfPredictor

        predictors = {}

        for model_id, folder_name in MODEL_IDS.items():
            folder = MODEL_ROOT / folder_name

            config = json.loads(
                (folder / "input_representation.json").read_text(
                    encoding="utf-8"
                )
            )

            representation = InputRepresentationConfig(**config)

            assert representation.configuration.lower() == model_id[-1]
            assert representation.strip_cite_tags is False

            with warnings.catch_warnings():
                warnings.simplefilter(
                    "error",
                    InconsistentVersionWarning,
                )

                predictor = TfidfPredictor.from_path(
                    model_path=folder / "tfidf_pipeline.joblib",
                    rep_config=representation,
                    model_name=model_id,
                )

            predictors[model_id] = predictor

        return predictors


def main():
    predictors = load_models()

    from src.modeling.predict import InferenceRequest

    for line in sys.stdin:
        if not line.strip():
            continue

        try:
            payload = json.loads(line)
            model_id = payload.pop("model_id")

            if model_id not in predictors:
                raise ValueError(f"Unknown TF-IDF model: {model_id}")

            request = InferenceRequest(**payload)

            with redirect_stdout(sys.stderr):
                prediction = predictors[model_id].predict(request)

            response = {
                "ok": True,
                "result": {
                    **prediction.model_dump(),
                    "is_demo": False,
                },
            }

        except Exception as exc:
            response = {
                "ok": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }

        sys.stdout.write(
            json.dumps(response, ensure_ascii=False) + "\n"
        )
        sys.stdout.flush()


if __name__ == "__main__":
    main()
