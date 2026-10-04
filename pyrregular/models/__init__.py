"""Model pipelines. Each module needs its own extra (see the README).

Importing this package does not import any model library.
"""

from importlib import import_module

# the 12 models of the pyrregular paper (the columns of its result tables), as
# "module:pipeline" so that listing them imports nothing
PAPER_MODELS = {
    "BORF": "aeon_borf:aeon_borf_pipeline",
    "BRITS": "brits:brits_pipeline",
    "GRU-D": "grud:grud_pipeline",
    "KNN": "knn:knn_dtw",
    "LGBM": "lgbm:lgbm_pipeline",
    "NCDE": "ncde:ncde_pipeline",
    "RAINDROP": "raindrop:raindrop_pipeline",
    "RIFC": "rifc:rifc_pipeline",
    "ROCKET": "rocket:rocket_pipeline",
    "SAITS": "saits:saits_pipeline",
    "SVM": "svm:svm_pipeline",
    "TIMESNET": "timesnet:timesnet_pipeline",
}


def list_paper_models():
    return list(PAPER_MODELS)


def get_paper_model(name):
    """Return an unfitted copy of a paper model's pipeline (imports its library)."""
    from sklearn.base import clone

    module, pipeline = PAPER_MODELS[name].split(":")
    return clone(getattr(import_module(f"pyrregular.models.{module}"), pipeline))
