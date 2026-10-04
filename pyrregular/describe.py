"""Checks of how regular the time axis of a dataset is.

``check_regularity(fn, df, ts_level)`` applies one of the checks below:

- ``ts_level=False``: each series is checked on the timestamps of its signals;
  one answer per series.
- ``ts_level=True``: each series is reduced to its merged timeline (the union of
  the timestamps of its signals) and all series are checked together, as the
  signals of one group; one answer for the whole dataset.

Correspondence with the taxonomy in the paper's appendix (a check returning
False means the irregularity is present):

- uneven sampling: ``are_all_signals_sampled_at_constant_intervals``
- ragged length: ``are_all_signals_equal_length``
- shift: ``are_all_signals_not_strongly_offset``
- ragged sampling: ``do_all_signals_have_equal_sampling`` (the k-th intervals of
  two signals are compared, for the intervals both signals have)
- partial observation (NaN where a value was expected) is not checked here.

Each check receives ``T``: a list (one item per series) of lists (one item per
signal) of sorted timestamp arrays. Timestamps are the original values, compared
exactly (datetimes as integer nanoseconds), so no rounding makes equal intervals
look different.
"""

import numpy as np


def _timestamps(df):
    """T[i][j]: sorted timestamps of signal j in series i (empty if absent)."""
    dims = df.irr.dims
    arr = df.data
    time = df["time_id"].values
    if time.dtype.kind in "mM":
        time = time.view(np.int64)
    ts = arr.coords[dims["ts_id"]]
    signal = arr.coords[dims["signal_id"]]
    t = time[arr.coords[dims["time_id"]]]
    order = np.lexsort((t, signal, ts))
    ts, signal, t = ts[order], signal[order], t[order]
    n_ts, n_signals = arr.shape[dims["ts_id"]], arr.shape[dims["signal_id"]]
    bounds = np.searchsorted(ts * n_signals + signal, np.arange(n_ts * n_signals + 1))
    return [
        [
            t[bounds[i * n_signals + j] : bounds[i * n_signals + j + 1]]
            for j in range(n_signals)
        ]
        for i in range(n_ts)
    ]


def check_regularity(fn, df, ts_level=True, return_percentage=True):
    T = _timestamps(df)
    if ts_level:
        T = [[np.unique(np.concatenate(signals)) for signals in T]]
    out = np.array([fn(signals) for signals in T], dtype=bool)
    if return_percentage:
        return out.mean()
    return out


def get_time_delta(signals):
    return [np.diff(t) for t in signals]


def get_lengths(signals):
    return np.array([len(t) for t in signals])


def are_all_signals_sampled_at_constant_intervals(signals):
    # signals with fewer than 2 timestamps have no interval and are skipped
    return all(d.min() == d.max() for d in get_time_delta(signals) if len(d))


def are_all_signals_equal_length(signals):
    lengths = get_lengths(signals)
    return lengths.min() == lengths.max()


def _starts_and_ends(signals):
    # signals without timestamps are skipped
    present = [t for t in signals if len(t)]
    return np.array([t[0] for t in present]), np.array([t[-1] for t in present])


def are_all_signals_not_offset(signals):
    # signals t_i,t_j such that their start and end times are different do not exist
    # i.e. all signals start and end at the same time
    starts, ends = _starts_and_ends(signals)
    if not len(starts):
        return True
    return starts.min() == starts.max() and ends.min() == ends.max()


def are_all_signals_not_strongly_offset(signals):
    # signals t_i,t_j such that t_i[0] < t_j[0] and t_i[-1] < t_j[-1] do not exist
    # strong offset means that not only they do not start or end at the same time,
    # but also that one starts and end before the other
    starts, ends = _starts_and_ends(signals)
    starts_before_another = starts[:, None] < starts[None, :]
    ends_before_another = ends[:, None] < ends[None, :]
    return not np.any(starts_before_another & ends_before_another)


def do_all_signals_have_equal_sampling(signals):
    # the k-th interval is the same in every signal that has a k-th interval
    deltas = get_time_delta(signals)
    for k in range(max((len(d) for d in deltas), default=0)):
        values = [d[k] for d in deltas if len(d) > k]
        if min(values) != max(values):
            return False
    return True
