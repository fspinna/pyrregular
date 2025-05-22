"""KNN.
K-Nearest Neighbors classifier with Dynamic Time Warping (DTW) and Sakoe-Chiba band.
"""

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
"""This pipeline applies KNeighborsTimeSeriesClassifier with DTW and a sakoe-chiba band."""
