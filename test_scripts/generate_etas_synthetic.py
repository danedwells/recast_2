"""
generate_etas_synthetic.py
==========================
One-time script: generates N independent synthetic ETAS catalogs using the
etas_2 library and saves them for use by test_replicate_ETAS.py.

BACKGROUND — what ETAS is and why we generate from it:

    ETAS (Epidemic-Type Aftershock Sequence, Ogata 1988) is the classical
    spatio-temporal model for earthquake sequences.  It treats seismicity as a
    self-exciting point process with two types of events:

      1. Background (immigrant) events — spatially uniform Poisson process with
         rate  μ  [events / day / km²].

      2. Aftershocks — each event of magnitude m triggers further events at rate
              K(m) = k₀ · 10^(a · (m − Mc))   [expected offspring count]
         with two additional kernels:
              Temporal:   h(Δt) ∝ (Δt + c)^(−p)          Omori-Utsu law
              Spatial:    g(r)  ∝ (r² + d_g)^(−ρ−1)      power-law kernel
                          where  d_g = d · exp(γ · (m − Mc))

    The etas_2 library (../etas_2/) implements ETAS as a branching process:
    each generation of events spawns the next via Poisson draws, and the
    process continues until no new offspring remain.

    We use etas_2 here as "ground truth" to generate perfectly labeled
    synthetic sequences.  test_replicate_ETAS.py then trains RECAST on those
    sequences and asks: how well can the neural model recover ETAS statistics?

RUN THIS SCRIPT ONCE before running test_replicate_ETAS.py.

    Usage (must have etas_2 dependencies installed in myeq3.12):
        conda run -n myeq3.12 python generate_etas_synthetic.py

    If you see "ModuleNotFoundError: No module named 'seismostats'", install:
        conda run -n myeq3.12 pip install \\
            "seismostats @ git+https://github.com/swiss-seismological-service/SeismoStats.git"

OUTPUT:
    case_studies/etas2_synthetic/raw_catalogs.parquet
        One row per event.
        Columns: catalog_id (int), latitude, longitude, time (datetime), magnitude

    case_studies/etas2_synthetic/true_params.json
        The ETAS parameters used for generation, plus derived spatial constants
        (polygon centroid, box dimensions) needed for the km-projection in
        test_replicate_ETAS.py.

PARAMETER UNITS (etas_2 convention):
    log10_mu   log10 of background rate [events / day / km²]
    log10_k0   log10 of base aftershock productivity [dimensionless, see note below]
    a          magnitude scaling of productivity  (= alpha in RECAST)
    log10_c    log10 of Omori c parameter [days]
    omega      Omori shape: p = 1 + omega  (omega > 0 → p > 1 → decaying kernel)
    log10_tau  log10 of Omori exponential taper [days]; large value ≈ pure power-law
    log10_d    log10 of base spatial length scale [km]
    gamma      magnitude scaling of spatial length scale
    rho        spatial power-law decay exponent

    Note on k0 vs RECAST k:
        etas_2 normalizes the Omori kernel to integrate to 1 over (0, ∞), so k0 is
        the expected total offspring count for a Mc event.  RECAST does NOT normalize
        its Omori kernel — it uses the un-normalised (Δt + c)^(−p) directly.
        Therefore RECAST's fitted k ≈ k0 / ∫(t+c)^(-p)dt = k0 × (p−1) / c^(1−p).
        p, c, and alpha are directly comparable; k and mu have different units/norms.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

# ── Add etas_2 to path so it can be imported without `pip install -e .` ────────
# Adjust this path if your repo layout differs.
ETAS2_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "etas_2")
)
if ETAS2_ROOT not in sys.path:
    sys.path.insert(0, ETAS2_ROOT)

from etas.simulation import generate_catalog

# ── Output directory ────────────────────────────────────────────────────────────
# Stored alongside the real case-study parquets so test scripts find them easily.
OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "case_studies", "etas2_synthetic"
)
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these to change the experiment
# ═══════════════════════════════════════════════════════════════════════════════

# ── Spatial domain ──────────────────────────────────────────────────────────────
# A 4° × 4° box in the Southern California desert.  Using a simple rectangle
# avoids coastline / border effects and gives a clean, featureless domain for
# the spatial model to learn in.
# etas_2 convention: Polygon vertices are (latitude, longitude) pairs, i.e.
# the x-axis is latitude and the y-axis is longitude.  This is consistent
# throughout etas_2's internals (bounds, point-in-polygon checks, etc.).
LAT_MIN, LAT_MAX = 33.0, 37.0   # 4° N-S range  ≈ 444 km
LON_MIN, LON_MAX = -118.5, -114.5  # 4° E-W range ≈ 364 km at 35°N

# Polygon centroid — used as the coordinate origin (0, 0) in km space by the
# test script.  Stored in true_params.json so the test script reads it there.
LAT_CTR = (LAT_MIN + LAT_MAX) / 2   # 35.0°N
LON_CTR = (LON_MIN + LON_MAX) / 2   # −116.5°W

# km/degree at the centroid latitude.  Standard approximations:
#   KM_PER_LAT ≈ 111.1 (nearly constant globally)
#   KM_PER_LON = 111.1 × cos(lat) (shrinks toward poles)
KM_PER_LAT = 111.1
KM_PER_LON = KM_PER_LAT * np.cos(np.radians(LAT_CTR))  # ≈ 91.0 km/° at 35°N

# ── Catalog generation settings ─────────────────────────────────────────────────
N_CATALOGS = 300    # total number of independent 30-day catalogs to generate
T_DAYS     = 30     # duration of each catalog window [days]
MC         = 3.6    # magnitude of completeness (catalog lower threshold)
BETA_MAIN  = np.log(10)  # Gutenberg-Richter β = b × ln(10);  b=1 → β≈2.303
DELTA_M    = 0.1    # magnitude bin width used internally by etas_2
RANDOM_SEED = 42    # fix numpy seed before generation for reproducibility

# ── True ETAS parameters (etas_2 convention) ────────────────────────────────────
# These are close to the California calibration (etas_2/config/simulate_catalog_config.json)
# with two adjustments:
#   1. log10_mu raised from −7.5 to −5.5 so the 4°×4° box produces ~15–30
#      events per 30-day window, giving RECAST enough data to learn from.
#   2. omega set to +0.08 (p = 1.08) so the Omori kernel strictly decays and
#      its integral converges — required for RECAST's temporal compensator.
#      (The California calibration has omega=−0.03, p=0.97, which needs
#       the exponential taper log10_tau to converge.)
TRUE_PARAMS = {
    "log10_mu"  : -5.5,   # background rate  [events/day/km²]
    "log10_k0"  : -2.49,  # aftershock productivity base
    "a"         :  1.69,  # magnitude scaling of productivity (= alpha in RECAST)
    "log10_c"   : -2.95,  # Omori c  [days]; 10^-2.95 ≈ 0.0011 days ≈ 1.6 min
    "omega"     :  0.08,  # Omori shape; p = 1 + omega = 1.08
    "log10_tau" :  3.99,  # taper scale [days]; 10^3.99 ≈ 9772 days ≈ irrelevant here
    "log10_d"   : -0.35,  # spatial base scale [km]; 10^-0.35 ≈ 0.45 km
    "gamma"     :  1.22,  # magnitude scaling of spatial scale
    "rho"       :  0.51,  # spatial power-law decay exponent
}

# ═══════════════════════════════════════════════════════════════════════════════
# DERIVED METADATA (not changed by user; computed from configuration above)
# ═══════════════════════════════════════════════════════════════════════════════

# Box half-extents in km — used by the test script to set axis limits.
HALF_KM_LAT = (LAT_MAX - LAT_MIN) / 2 * KM_PER_LAT   # ≈ 222 km North/South
HALF_KM_LON = (LON_MAX - LON_MIN) / 2 * KM_PER_LON   # ≈ 182 km East/West

# Effective area of the box [km²] — used to convert etas_2's μ [ev/day/km²]
# to RECAST's μ [ev/day] for the parameter comparison table in the test script.
BOX_AREA_KM2 = (LAT_MAX - LAT_MIN) * KM_PER_LAT * (LON_MAX - LON_MIN) * KM_PER_LON

print(f"Box: {LAT_MIN}–{LAT_MAX}°N, {LON_MIN}–{LON_MAX}°W")
print(f"  Area          : {BOX_AREA_KM2:.0f} km²")
print(f"  Center (km=0) : {LAT_CTR}°N, {LON_CTR}°W")
print(f"  KM_PER_LON    : {KM_PER_LON:.2f}")
print(f"  E-W half-ext  : ±{HALF_KM_LON:.1f} km")
print(f"  N-S half-ext  : ±{HALF_KM_LAT:.1f} km")
print(f"Expected background events per 30-day catalog: "
      f"{10**TRUE_PARAMS['log10_mu'] * BOX_AREA_KM2 * T_DAYS:.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD THE SHAPELY POLYGON
# ═══════════════════════════════════════════════════════════════════════════════

# etas_2 internally passes this Polygon to geopandas.GeoDataFrame, which uses
# (x, y) = (latitude, longitude).  So Polygon vertices must be (lat, lon) pairs.
polygon = Polygon([
    [LAT_MIN, LON_MIN],
    [LAT_MAX, LON_MIN],
    [LAT_MAX, LON_MAX],
    [LAT_MIN, LON_MAX],
    [LAT_MIN, LON_MIN],   # close the ring
])

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE CATALOGS
# ═══════════════════════════════════════════════════════════════════════════════

np.random.seed(RANDOM_SEED)  # fix global numpy state for reproducibility

all_catalogs = []
t_wall_start = time.time()
T_ORIGIN = pd.Timestamp("2000-01-01")   # arbitrary absolute start — only relative
                                         # times matter inside RECAST sequences

print(f"\nGenerating {N_CATALOGS} × {T_DAYS}-day ETAS catalogs...")
for i in range(N_CATALOGS):
    # Each catalog gets its own non-overlapping time window.
    # The window label doesn't affect the statistics — each catalog is seeded
    # independently — but it avoids confusing pandas when concatenating.
    t_start = T_ORIGIN + pd.Timedelta(days=i * T_DAYS)
    t_end   = t_start  + pd.Timedelta(days=T_DAYS)

    cat = generate_catalog(
        polygon          = polygon,
        timewindow_start = t_start,
        timewindow_end   = t_end,
        parameters       = TRUE_PARAMS,
        mc               = MC,
        beta_main        = BETA_MAIN,
        delta_m          = DELTA_M,
    )

    # generate_catalog may return an empty DataFrame when no events occur.
    # These empty catalogs are kept (marked by catalog_id) so the test script
    # can report the true fraction of empty sequences.
    cat = cat[["latitude", "longitude", "time", "magnitude"]].copy()
    cat["catalog_id"] = i
    all_catalogs.append(cat)

    if (i + 1) % 50 == 0:
        elapsed = time.time() - t_wall_start
        print(f"  [{i+1}/{N_CATALOGS}]  events this catalog: {len(cat):3d}  "
              f"elapsed: {elapsed:.1f}s")

# ── Summary statistics ──────────────────────────────────────────────────────────
all_df = pd.concat(all_catalogs, ignore_index=True)
n_nonempty = sum(len(c) > 0 for c in all_catalogs)
counts_per_cat = all_df.groupby("catalog_id").size()

print(f"\nDone in {time.time() - t_wall_start:.1f}s")
print(f"  Total events  : {len(all_df):,}")
print(f"  Non-empty seqs: {n_nonempty}/{N_CATALOGS}")
print(f"  Events per seq: min={counts_per_cat.min()}, "
      f"median={counts_per_cat.median():.0f}, max={counts_per_cat.max()}")

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════

parquet_path = os.path.join(OUT_DIR, "raw_catalogs.parquet")
all_df.to_parquet(parquet_path)
print(f"\nSaved catalog : {parquet_path}")

# Save all metadata needed by test_replicate_ETAS.py in one place.
# This way the test script never hard-codes the generation parameters.
meta = {
    # The true ETAS parameters (etas_2 convention)
    "true_params": TRUE_PARAMS,
    # Coordinate system info — the test script reads these to project lat/lon → km
    "lat_ctr": LAT_CTR,
    "lon_ctr": LON_CTR,
    "km_per_lat": KM_PER_LAT,
    "km_per_lon": KM_PER_LON,
    "half_km_lat": HALF_KM_LAT,
    "half_km_lon": HALF_KM_LON,
    "box_area_km2": BOX_AREA_KM2,
    # Data generation settings
    "n_catalogs": N_CATALOGS,
    "t_days": T_DAYS,
    "mc": MC,
    "mag_max": 8.0,          # maximum plausible magnitude (for ContinuousMarks)
    "richter_b": 1.0,        # b = beta_main / ln(10)
    "random_seed": RANDOM_SEED,
    # RECAST-comparable derived values (for the parameter table in test script)
    # p and c are directly comparable; mu needs unit conversion (see docstring).
    "recast_p_true"   : 1 + TRUE_PARAMS["omega"],          # Omori p
    "recast_c_true"   : 10 ** TRUE_PARAMS["log10_c"],      # Omori c [days]
    "recast_alpha_true": TRUE_PARAMS["a"],                  # = alpha in RECAST
    "recast_mu_rate_true": (                                # background [events/day]
        10 ** TRUE_PARAMS["log10_mu"] * BOX_AREA_KM2
    ),
}
json_path = os.path.join(OUT_DIR, "true_params.json")
with open(json_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Saved metadata: {json_path}")
print("\nRun test_replicate_ETAS.py next.")
