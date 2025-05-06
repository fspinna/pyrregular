from tslearn.neighbors import KNeighborsTimeSeriesClassifier

from pyrregular.wrappers.tslearn_wrapper import TslearnWrapper

knn_dtw = TslearnWrapper(
    KNeighborsTimeSeriesClassifier(
        n_neighbors=5,
        metric="dtw",
        metric_params={
            "global_constraint": "sakoe_chiba",
            "sakoe_chiba_radius": 10,
        },
    )
)
