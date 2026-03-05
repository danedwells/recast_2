
import eq
import numpy as np
import torch
torch.set_num_threads(1)
"""
Set up some classes and functions to generate synthetic data

Geometry Functions
---------------------------------
geometry_cross_fault(half_length=5, width=0.2)
geometry_diffuse(scale=3.0)
goemetry_edge_cluster(center=(4.,4.,), spread=0.5, half_length=5.0,width=0.2)
geometry_single_fault(center=(0.,0.),strike=0.0,half_length=5.0,width=0.2)

Temporal Functions:
----------------------------------
omori_rate(K=10, c= 0.1, p =1.1)
constant_rate(lam=5.0)
swarm_rate(peaks=None,K=8,sigma=2)

"""

# ─── Temporal behavior functions ──────────────────────────────────────────────

def omori_rate(t, K=10.0, c=0.1, p=1.1, t_start=0.):
    """Omori-Utsu rate: lambda(t) = K / (t + c)^p
    K = productivity (higher - more events)
    c = time offset (small value for numerical stability)
    p = decay componet (larger - faster)
    """
    return K / (t - t_start + c) ** p


def constant_rate(t, lam=5.0):
    """Constant Poisson rate."""
    return lam


def swarm_rate(t, peaks=None, K=8.0, sigma=2.0):
    """
    Swarm-like behavior: sum of Gaussian bursts.
    peaks: list of burst centers (default: a few spread across [0, t_end])
    """
    if peaks is None:
        peaks = [10.0, 30.0, 60.0, 85.0]
    return sum(K * np.exp(-0.5 * ((t - mu) / sigma) ** 2) for mu in peaks) + 0.5

# ─── Geometry functions ───────────────────────────────────────────────────────

def geometry_single_fault(n, strike=0.0, half_length=5.0, width=0.2, center=(0.,0.)):
    """
    Events along a single linear fault with arbitrary strike.
    strike: fault azimuth in degrees (0=N, 90=E, 180=S, 270=W), clockwise from North.
    Events are sampled along the fault axis and rotated accordingly.
    """
    along = np.random.uniform(-half_length, half_length, n)
    across = np.random.normal(0, width, n)
    # Strike measured clockwise from North → angle from +x axis (East)
    angle_rad = np.radians(90.0 - strike)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    x = along * cos_a - across * sin_a
    y = along * sin_a + across * cos_a
    return x + center[0], y + center[1]


def geometry_cross_fault(n, half_length=5.0, width=0.2):
    """Events on a cross of EW and NS faults."""
    fault = np.random.choice(['EW', 'NS'], size=n)
    along = np.random.uniform(-half_length, half_length, n)
    across = np.random.normal(0, width, n)
    x = np.where(fault == 'EW', along, across)
    y = np.where(fault == 'EW', across, along)
    return x, y


def geometry_diffuse(n, center=(0.,0.,),scale=3.0):
    """Diffuse cloud of events."""
    x = np.random.normal(0, scale, n)
    y = np.random.normal(0, scale, n)
    return x + center[0], y + center[1]


def geometry_edge_cluster(n, center=(4.0, 4.0), spread=0.5, half_length=5.0, width=0.2):
    """Most events clustered near one edge, some on a background fault."""
    n_edge = int(0.8 * n)
    n_fault = n - n_edge
    x_edge = np.random.normal(center[0], spread, n_edge)
    y_edge = np.random.normal(center[1], spread, n_edge)
    x_fault = np.random.uniform(-half_length, half_length, n_fault)
    y_fault = np.random.normal(0, width, n_fault)
    return np.concatenate([x_edge, x_fault]), np.concatenate([y_edge, y_fault])


# ─── Thinning sampler ─────────────────────────────────────────────────────────

def sample_times(rate_fn, t_start, t_end, rate_kwargs):
    """Thinning algorithm for inhomogeneous Poisson process."""
    t = t_start
    arrival_times = []
    lambda_max = rate_fn(t_start, **rate_kwargs)

    while t < t_end:
        dt = np.random.exponential(1.0 / max(lambda_max, 1e-6))
        t += dt
        if t >= t_end:
            break
        lam_t = rate_fn(t, **rate_kwargs)
        if np.random.uniform() < lam_t / lambda_max:
            arrival_times.append(t)
        lambda_max = lam_t

    return np.array(arrival_times)


# ─── Master sequence generator ────────────────────────────────────────────────

def generate_sequence(
    t_start=0.0,
    t_end=100.0,
    rate_fn=omori_rate,
    rate_kwargs=None,
    geometry_fn=geometry_cross_fault,
    geometry_kwargs=None,
    mag_scale=1.0,
    mag_min=2.0,
    xy_bounds=(-6.0, 6.0),
):
    """
    Generate a synthetic earthquake sequence.

    Parameters
    ----------
    rate_fn       : callable(t, **rate_kwargs) -> float
    rate_kwargs   : dict of kwargs forwarded to rate_fn
    geometry_fn   : callable(n, **geometry_kwargs) -> (x_vals, y_vals)
    geometry_kwargs : dict of kwargs forwarded to geometry_fn
    """
    if rate_kwargs is None:
        rate_kwargs = {}
    if geometry_kwargs is None:
        geometry_kwargs = {}

    arrival_times = sample_times(rate_fn, t_start, t_end, rate_kwargs)

    if len(arrival_times) == 0:
        return None

    arrival_times = np.unique(np.sort(arrival_times))
    arrival_times = arrival_times[arrival_times > t_start]

    inter_times = np.diff(arrival_times, prepend=[t_start], append=[t_end])
    assert np.all(inter_times >= 0), f"Negative inter-times detected"

    n = len(arrival_times)
    mag_vals  = (np.random.exponential(scale=mag_scale, size=n) + mag_min).astype(np.float32)
    x_vals, y_vals = geometry_fn(n, **geometry_kwargs)
    x_vals = x_vals.astype(np.float32)
    y_vals = y_vals.astype(np.float32)

    lo, hi = xy_bounds
    mag = eq.data.ContinuousMarks(mag_vals, bounds=[mag_min, 8.0], nll_bounds=[mag_min + 0.5, 8.0])
    x_loc = eq.data.ContinuousMarks(x_vals, bounds=[lo, hi], nll_bounds=[lo + 1, hi - 1])
    y_loc = eq.data.ContinuousMarks(y_vals, bounds=[lo, hi], nll_bounds=[lo + 1, hi - 1])

    return eq.data.Sequence(inter_times, t_start=t_start, mag=mag, x_loc=x_loc, y_loc=y_loc)
