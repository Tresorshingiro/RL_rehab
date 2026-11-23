import pandas as pd

TOLERANCE = 0.10

def rolling_mean(values, window):
    return pd.Series(values).rolling(window, min_periods=1).mean()


def convergence_episode(returns, window=50):
    smooth = rolling_mean(returns, window).to_numpy()
    if len(smooth) < 2:
        return len(smooth)
    final_level = smooth[-max(1, len(smooth) // 10):].mean()
    band = TOLERANCE * (smooth.max() - smooth.min())
    outside = abs(smooth - final_level) > band
    if not outside.any():
        return 1
    last_outside = outside.nonzero()[0][-1]                  # last episode the band
    return int(last_outside + 2)                             # episodes are numbered from 1
