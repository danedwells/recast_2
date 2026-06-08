import pytest
import numpy as np
import torch
import eq
from eq.data.synthetic import (
    omori_rate,
    constant_rate,
    swarm_rate,
    geometry_single_fault,
    geometry_cross_fault,
    geometry_diffuse,
    geometry_edge_cluster,
    generate_sequence,
)

RNG_SEED = 42

def _to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return arr

# ─── Rate functions ───────────────────────────────────────────────────────────

class TestOmoriRate:
    def test_positive(self):
        assert omori_rate(0.0) > 0
        assert omori_rate(10.0) > 0

    def test_decreasing(self):
        """Rate must strictly decrease over time."""
        times = [0.0, 1.0, 5.0, 20.0, 100.0]
        rates = [omori_rate(t) for t in times]
        assert all(r1 > r2 for r1, r2 in zip(rates, rates[1:]))

    def test_c_shift(self):
        """At t=0, rate should equal K / c^p."""
        K, c, p = 10.0, 0.5, 1.2
        expected = K / c ** p
        assert np.isclose(omori_rate(0.0, K=K, c=c, p=p), expected)


class TestConstantRate:
    def test_returns_lam(self):
        assert constant_rate(0.0, lam=3.0) == 3.0
        assert constant_rate(99.9, lam=3.0) == 3.0

    def test_time_independent(self):
        lam = 7.5
        assert constant_rate(0.0, lam=lam) == constant_rate(50.0, lam=lam)


class TestSwarmRate:
    def test_positive(self):
        assert swarm_rate(0.0) > 0

    def test_peaks_higher_than_baseline(self):
        """Rate at a peak center should exceed the background floor."""
        peak = 30.0
        rate_at_peak = swarm_rate(peak, peaks=[peak], K=8.0, sigma=2.0)
        rate_far = swarm_rate(peak + 100.0, peaks=[peak], K=8.0, sigma=2.0)
        assert rate_at_peak > rate_far

    def test_default_peaks(self):
        """Default call should not raise and should return a positive value."""
        assert swarm_rate(10.0) > 0


# ─── Geometry functions ───────────────────────────────────────────────────────

class TestGeometrySingleFault:
    def test_output_shape(self):
        x, y = geometry_single_fault(100)
        assert x.shape == (100,) and y.shape == (100,)

    def test_strike_0_north(self):
        """strike=0 (North): spread should be in y, narrow in x."""
        np.random.seed(RNG_SEED)
        x, y = geometry_single_fault(10000, strike=0.0, half_length=5.0, width=0.01)
        assert np.std(y) > np.std(x) * 5

    def test_strike_90_east(self):
        """strike=90 (East): spread should be in x, narrow in y."""
        np.random.seed(RNG_SEED)
        x, y = geometry_single_fault(10000, strike=90.0, half_length=5.0, width=0.01)
        assert np.std(x) > np.std(y) * 5

    def test_strike_180_same_as_0(self):
        """strike=180 (South) has the same axis as strike=0 (North)."""
        np.random.seed(RNG_SEED)
        x0, y0 = geometry_single_fault(10000, strike=0.0, half_length=5.0, width=0.01)
        np.random.seed(RNG_SEED)
        x180, y180 = geometry_single_fault(10000, strike=180.0, half_length=5.0, width=0.01)
        # Spread structure should be the same (axis is identical modulo direction)
        assert np.isclose(np.std(x0), np.std(x180), rtol=0.05)
        assert np.isclose(np.std(y0), np.std(y180), rtol=0.05)

    def test_extent_bounded_by_half_length(self):
        np.random.seed(RNG_SEED)
        x, y = geometry_single_fault(1000, strike=45.0, half_length=5.0, width=0.1)
        radius = np.sqrt(x**2 + y**2)
        # All points should be within half_length + a few widths of origin
        assert np.max(radius) < 5.0 * np.sqrt(2) + 1.0


class TestGeometryCrossFault:
    def test_output_shape(self):
        x, y = geometry_cross_fault(200)
        assert x.shape == (200,) and y.shape == (200,)

    def test_bimodal_structure(self):
        """Cross fault should have variance in both x and y."""
        np.random.seed(RNG_SEED)
        x, y = geometry_cross_fault(5000, half_length=5.0, width=0.1)
        assert np.std(x) > 1.0
        assert np.std(y) > 1.0


class TestGeometryDiffuse:
    def test_output_shape(self):
        x, y = geometry_diffuse(100)
        assert x.shape == (100,) and y.shape == (100,)

    def test_scale_controls_spread(self):
        np.random.seed(RNG_SEED)
        x1, _ = geometry_diffuse(5000, scale=1.0)
        np.random.seed(RNG_SEED)
        x2, _ = geometry_diffuse(5000, scale=5.0)
        assert np.std(x2) > np.std(x1) * 3


class TestGeometryEdgeCluster:
    def test_output_shape(self):
        x, y = geometry_edge_cluster(100)
        assert x.shape == (100,) and y.shape == (100,)

    def test_majority_near_center(self):
        """80% of events should be near the specified center."""
        np.random.seed(RNG_SEED)
        center = (4.0, 4.0)
        x, y = geometry_edge_cluster(1000, center=center, spread=0.3)
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        near = np.mean(dist < 1.5)
        assert near > 0.7


# ─── generate_sequence ────────────────────────────────────────────────────────


class TestGenerateSequence:
    def test_returns_sequence(self):
        np.random.seed(RNG_SEED)
        seq = generate_sequence(rate_fn=constant_rate, rate_kwargs={"lam": 10.0})
        assert seq is not None
        assert isinstance(seq, eq.data.Sequence)

    def test_inter_times_positive(self):
        np.random.seed(RNG_SEED)
        seq = generate_sequence(rate_fn=constant_rate, rate_kwargs={"lam": 10.0})
        assert (seq.inter_times >= 0).all()

    def test_inter_times_sum_to_duration(self):
        np.random.seed(RNG_SEED)
        t_start, t_end = 5.0, 55.0
        seq = generate_sequence(
            t_start=t_start, t_end=t_end,
            rate_fn=constant_rate, rate_kwargs={"lam": 10.0}
        )
        assert np.isclose(seq.inter_times.sum(), t_end - t_start, rtol=1e-5)

    def test_marks_dtype_float32(self):
        np.random.seed(RNG_SEED)
        seq = generate_sequence(rate_fn=constant_rate, rate_kwargs={"lam": 5.0})
        assert seq.mag.data.dtype == torch.float32
        assert seq.x_loc.data.dtype == torch.float32
        assert seq.y_loc.data.dtype == torch.float32

    def test_marks_count_matches_events(self):
        np.random.seed(RNG_SEED)
        seq = generate_sequence(rate_fn=constant_rate, rate_kwargs={"lam": 5.0})
        n_events = len(seq)
        assert len(seq.mag.data) == n_events
        assert len(seq.x_loc.data) == n_events
        assert len(seq.y_loc.data) == n_events

    def test_custom_geometry(self):
        np.random.seed(RNG_SEED)
        seq = generate_sequence(
            rate_fn=constant_rate, rate_kwargs={"lam": 5.0},
            geometry_fn=geometry_single_fault,
            geometry_kwargs={"strike": 45.0},
        )
        assert seq is not None

    def test_low_rate_may_return_none(self):
        """Very low rate over short window can produce empty sequence."""
        np.random.seed(0)
        results = [
            generate_sequence(
                t_end=0.1,
                rate_fn=constant_rate, rate_kwargs={"lam": 0.01}
            )
            for _ in range(50)
        ]
        assert any(r is None for r in results)