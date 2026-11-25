import pandas as pd

TOLERANCE = 0.10

def rolling_mean(values, window):
    return pd.Series(values).rolling(window, min_periods=1).mean()


def convergence_band(returns, window=50):
    """(episode, final_level, band_half_width) of the "episodes until stable" measure.

    The return is smoothed with a rolling mean. The final level is the mean of the last
    10% of the smoothed curve, and the band is TOLERANCE x (max - min) of that curve.
    The run has converged at the first episode after which it never leaves the band.
    """
    smooth = rolling_mean(returns, window).to_numpy()
    if len(smooth) < 2:
        return len(smooth), float(smooth.mean()) if len(smooth) else 0.0, 0.0
    final_level = smooth[-max(1, len(smooth) // 10):].mean()
    band = TOLERANCE * (smooth.max() - smooth.min())
    outside = abs(smooth - final_level) > band
    if not outside.any():
        return 1, final_level, band
    last_outside = outside.nonzero()[0][-1]                  # last episode outside the band
    return int(last_outside + 2), final_level, band          # episodes are numbered from 1


def convergence_episode(returns, window=50):
    return convergence_band(returns, window)[0]
