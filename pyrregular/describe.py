import awkward as ak


def check_regularity(fn, df, ts_level=True, return_percentage=True):
    X, T = df.irr.to_awkward(ts_level=ts_level)
    if ts_level:
        T = ak.flatten(T)[None]
    out = fn(T)
    if return_percentage:
        return ak.mean(out)
    return out


def get_time_delta(T):
    return T[:, :, 1:] - T[:, :, :-1]


def get_lengths(T):
    return ak.count(T, axis=2)


def are_all_signals_sampled_at_constant_intervals(T):
    delta_t = get_time_delta(T)
    delta_t_diff = ak.min(delta_t, axis=2) == ak.max(delta_t, axis=2)
    return ak.all(delta_t_diff, axis=1)


def are_all_signals_equal_length(T):
    lengths = get_lengths(T)
    return ak.max(lengths, axis=1) == ak.min(lengths, axis=1)


def are_all_signals_not_offset(T):
    # signals t_i,t_j such that their start and end times are different do not exist
    # i.e. all signals start and end at the same time

    start_times = ak.min(T, axis=2)
    end_times = ak.max(T, axis=2)
    start_times_equal = ak.min(start_times, axis=1) == ak.max(start_times, axis=1)
    end_times_equal = ak.min(end_times, axis=1) == ak.max(end_times, axis=1)
    return start_times_equal & end_times_equal


def are_all_signals_not_strongly_offset(T):
    # signals t_i,t_j such that t_i[0] < t_j[0] and t_i[-1] < t_j[-1] do not exist
    # strong offset means that not only they do not start or end at the same time,
    # but also that one starts and end before the other

    start_times = ak.min(T, axis=2)
    end_times = ak.max(T, axis=2)

    # at least one signal starts before another
    starts_before_another = start_times[..., :, None] < start_times[..., None, :]

    # at least one signal ends before another
    ends_before_another = end_times[..., :, None] < end_times[..., None, :]

    starts_and_ends_before_another = starts_before_another & ends_before_another

    return ~ak.any(ak.any(starts_and_ends_before_another, axis=-1), axis=-1)


def do_all_signals_have_equal_sampling(T):
    delta_t = get_time_delta(T)
    delta_t_aligned = ak.min(delta_t, axis=1) == ak.max(delta_t, axis=1)
    return ak.all(delta_t_aligned, axis=1)
