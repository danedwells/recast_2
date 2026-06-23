#%%
import eq
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
torch.set_num_threads(1)
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os
import shutil
from plot_helpers import *

# ─── Paths ─────────────────────────────────────────────────────────────────────
HOME_DIR = "/home/a01738353/2024_NEHRP/RECAST/"
ANSS_DIR = f"{HOME_DIR}data/ANSS_MultiCatalog"
DATA_DIR = f"{HOME_DIR}case_studies/Ridgecrest_2019_catalog.parquet"
SAVE_DIR = f"{HOME_DIR}results/case_studies/Ridgecrest"
FIGS_DIR = f"{HOME_DIR}figs/case_studies/Ridgecrest"

for d in [SAVE_DIR, FIGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Ridgecrest M7.1 mainshock — fixed coordinate origin ───────────────────────
# July 6, 2019 03:19:53 UTC — ci38457511
MS_LAT = 35.7695
MS_LON = -117.5993
KM_PER_DEG_LAT = 111.1
KM_PER_DEG_LON = KM_PER_DEG_LAT * np.cos(np.radians(MS_LAT))
RADIUS_KM = 250.0
MAG_MIN = 4.5   # matches ANSS global catalog completeness
MAG_MAX = 8.0
JITTER_KM = 150.0  # random spatial translation applied per sequence during training


# ─── Helper: load a parquet catalog as a mainshock-centered Sequence ───────────
def load_mainshock_sequence(path, ms_lat, ms_lon, mc=4.5, radius_km=250.0):
    """Load events from parquet; project into local km relative to the mainshock.

    Coordinate convention matches ANSS_MultiCatalog.get_catalog_batch:
      x_loc = km east  of the mainshock
      y_loc = km north of the mainshock
    so training and test sequences share the same semantic coordinate frame.
    """
    df = pd.read_parquet(path)
    df = df.sort_values("time").reset_index(drop=True)
    df = df[df.mag >= mc].reset_index(drop=True)

    km_per_lat = 111.1
    km_per_lon = km_per_lat * np.cos(np.radians(ms_lat))
    x_km = ((df.longitude.values - ms_lon) * km_per_lon).astype(np.float32)
    y_km = ((df.latitude.values  - ms_lat) * km_per_lat).astype(np.float32)
    dist_km = np.sqrt(x_km**2 + y_km**2)

    keep = dist_km <= radius_km
    df   = df[keep].reset_index(drop=True)
    x_km = x_km[keep] - 100
    y_km = y_km[keep] - 100

    # Time: fractional days from first event; 1-hour lead so first inter-time > 0
    t0 = df["time"].iloc[0]
    offset = 1.0 / 24.0
    arrival_times = (df["time"] - t0).dt.total_seconds().values / 86400.0 + offset
    t_end = float(arrival_times[-1]) + offset
    inter_times = np.diff(arrival_times, prepend=[0.0], append=[t_end])

    mag = eq.data.ContinuousMarks(
        df.mag.values.astype(np.float32),
        bounds=[mc, MAG_MAX],
        nll_bounds=[mc, MAG_MAX],
    )
    seq = eq.data.Sequence(
        inter_times,
        t_start=0.0,
        mag=mag,
        x_loc=torch.as_tensor(x_km),
        y_loc=torch.as_tensor(y_km),
    )
    return seq, df


#%%
# ─── Load ANSS multi-sequence training catalog ─────────────────────────────────
from eq.catalogs.anss import ANSS_MultiCatalog

ANSS_PARAMS = dict(
    root_dir=ANSS_DIR,
    num_sequences=5000,
    train_frac=0.6, val_frac=0.2, test_frac=0.2,
    train_daterange=[pd.Timestamp("1990-01-01"), pd.Timestamp("2010-01-01")],
    val_daterange=[pd.Timestamp("2010-01-01"),   pd.Timestamp("2015-01-01")],
    test_daterange=[pd.Timestamp("2015-01-01"),  pd.Timestamp("2020-01-01")],
    t_end_days=365,
    radius_kilometers=250,
    mag_completeness=4.5,
    minimum_mainshock_mag=6.5,
    random_state=123,
)

anss_catalog = ANSS_MultiCatalog(**ANSS_PARAMS)

# Cached .pt files may predate the x_loc/y_loc addition — regenerate if stale.
# The raw CSV is cached at data/raw/anss_global_catalog.csv so this takes ~10 min.
if 'x_loc' not in anss_catalog.train[0]:
    print("Stale ANSS cache (no x_loc/y_loc) — regenerating from cached CSV...")
    shutil.rmtree(ANSS_DIR)
    anss_catalog = ANSS_MultiCatalog(**ANSS_PARAMS)


def _keep_sequence(seq, max_depth_km=50.0, min_aftershocks=10):
    """Return True if the sequence passes quality filters.

    Filters:
      1. Proxy mainshock depth ≤ max_depth_km — removes megathrust interface
         and intraslab subduction events.  The mainshock is the event nearest
         to the coordinate origin (x_loc=0, y_loc=0).
      2. At least min_aftershocks events in the NLL window — removes sequences
         from remote / poorly-instrumented regions that contribute no signal.
    """
    dist2 = seq.x_loc ** 2 + seq.y_loc ** 2
    proxy_depth = float(seq.depth[dist2.argmin()].item())
    if proxy_depth > max_depth_km:
        return False
    n_after = int((seq.arrival_times >= seq.t_nll_start).sum())
    return n_after >= min_aftershocks


for split in ("train", "val", "test"):
    dataset = getattr(anss_catalog, split)
    before = len(dataset)
    kept = [seq for seq in dataset if _keep_sequence(seq)]
    setattr(anss_catalog, split, eq.data.InMemoryDataset(kept))
    print(f"{split}: {before} → {len(kept)} sequences after depth≤50 km + ≥10 aftershocks filter")

print(f"Sequence keys       : {list(anss_catalog.train[0].keys())}")
s0 = anss_catalog.train[0]
print(f"x_loc range example : [{s0.x_loc.min():.0f}, {s0.x_loc.max():.0f}] km")
print(f"Events per sequence (train[0]): {s0.num_events}")


#%%
# ─── Normalization stats — pooled across all training sequences ────────────────
# Because each sequence is projected relative to its own mainshock,
# x_mean ≈ y_mean ≈ 0 by construction.
all_x   = np.concatenate([s.x_loc.numpy() for s in anss_catalog.train])
all_y   = np.concatenate([s.y_loc.numpy() for s in anss_catalog.train])
all_tau = np.concatenate([s.inter_times[:-1].numpy() for s in anss_catalog.train])

x_mean, y_mean = 0.0, 0.0
# Inflate std to match training distribution: jitter adds uniform(±JITTER_KM)
# variance = JITTER_KM² / 3, so std_effective = sqrt(std_raw² + JITTER_KM²/3)
_jitter_std = JITTER_KM / np.sqrt(3)
x_std    = float(np.sqrt(all_x.std()**2 + _jitter_std**2))
y_std    = float(np.sqrt(all_y.std()**2 + _jitter_std**2))
tau_mean = float(all_tau.mean())

print(f"x_std={x_std:.1f} km  y_std={y_std:.1f} km  (includes {_jitter_std:.1f} km jitter)")
print(f"tau_mean={tau_mean:.4f} days")


#%%
# ─── Load Ridgecrest test sequence (mainshock-centered, same convention as ANSS) ─
seq, df = load_mainshock_sequence(DATA_DIR, MS_LAT, MS_LON, mc=MAG_MIN, radius_km=RADIUS_KM)
print(f"Ridgecrest events (M≥{MAG_MIN}): {seq.num_events}")
print(f"Duration : {seq.t_end:.2f} days")
print(f"x range  : [{seq.x_loc.min().item():.1f}, {seq.x_loc.max().item():.1f}] km")
print(f"y range  : [{seq.y_loc.min().item():.1f}, {seq.y_loc.max().item():.1f}] km")

# Quick look at the test sequence
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].stem(seq.arrival_times.numpy(), seq.mag.numpy(),
             linefmt='grey', markerfmt='o', basefmt=' ')
axes[0].set_xlabel("Days since mainshock")
axes[0].set_ylabel("Magnitude")
axes[0].set_title(f"Ridgecrest — M≥{MAG_MIN} M-t plot")

sc = axes[1].scatter(seq.x_loc.numpy(), seq.y_loc.numpy(),
                     s=seq.mag.numpy()**2,
                     c=seq.arrival_times.numpy(), cmap='viridis', alpha=0.6)
axes[1].axhline(0, color='k', lw=0.5, ls='--'); axes[1].axvline(0, color='k', lw=0.5, ls='--')
axes[1].set_xlabel("East of mainshock (km)")
axes[1].set_ylabel("North of mainshock (km)")
axes[1].set_title("Ridgecrest — Spatial (mainshock at origin)")
plt.colorbar(sc, ax=axes[1], label="Days since mainshock")
plt.tight_layout()
plt.show()


#%%
# ─── Model ─────────────────────────────────────────────────────────────────────
model = eq.models.RecurrentTPP(
    context_size=256,
    num_components_time=8,
    num_components_space=32,
    tau_mean=tau_mean,
    x_mean=x_mean,
    y_mean=y_mean,
    x_std=x_std,
    y_std=y_std,
)


#%%
# ─── Training — mini-batch over ANSS analog library ───────────────────────────
from torch.utils.data import DataLoader
from functools import partial


def collate_with_jitter(sequences, max_km=100.0):
    """Collate sequences with a random per-sequence spatial translation.

    Shifts all events in each sequence by the same random (dx, dy), so the
    mainshock is no longer always at (0, 0). The model must learn to locate
    the mainshock from the event stream rather than memorising the origin as a
    prior. Relative positions within a sequence are preserved.
    """
    jittered = []
    for seq in sequences:
        dx = float(np.random.uniform(-max_km, max_km))
        dy = float(np.random.uniform(-max_km, max_km))
        # Carry all extra fields (mag, mag_bounds, lat, lon, …) through unchanged
        extra = {k: v for k, v in seq.items()
                 if k not in seq.default_sequence_attrs}
        extra['x_loc'] = seq.x_loc + dx
        extra['y_loc'] = seq.y_loc + dy
        jittered.append(eq.data.Sequence(
            seq.inter_times.clone(),
            t_start=seq.t_start,
            t_nll_start=seq.t_nll_start,
            **extra,
        ))
    return eq.data.Batch.from_list(jittered)


_RETRAIN_ = True
model.train()

if _RETRAIN_:
    epochs = 100          # one epoch = ~94 mini-batches of 32 sequences each
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Training on: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    train_loader = DataLoader(
        anss_catalog.train,
        batch_size=32,
        shuffle=True,
        collate_fn=partial(collate_with_jitter, max_km=JITTER_KM),
    )

    running_loss = []
    for epoch in range(epochs):
        epoch_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            nll = model.nll_loss(batch).mean()
            nll.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(nll.item())
        epoch_mean = float(np.mean(epoch_losses))
        running_loss.append(epoch_mean)
        if epoch % 1 == 0:
            print(f"Epoch {epoch:3d}  NLL: {epoch_mean:.4f}")
        scheduler.step()

    torch.save(model.state_dict(), f"{SAVE_DIR}/model_anss.pt")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(running_loss)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean NLL per sequence")
    ax.set_title("Training loss — ANSS analog library")
    plt.tight_layout()
    plt.show()


#%%
# ─── Load trained model ────────────────────────────────────────────────────────
map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
model.load_state_dict(torch.load(f"{SAVE_DIR}/model_anss.pt", map_location=map_location))

#%%
model.eval()
model = model.cpu()

total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total:,}")

# ─── Inference — condition on mainshock, forecast aftershocks ──────────────────
# Condition on the first ~2.4 hours (mainshock + any immediate early events)
condition_days = seq.t_start + 0.5
forecast_days  = seq.t_end - condition_days
cond_seq = seq.get_subsequence(seq.t_start, condition_days)
print(f"Conditioning events: {cond_seq.num_events}")

with torch.no_grad():
    predicted_batch = model.sample(
        batch_size=1,
        duration=forecast_days,
        t_start=condition_days,
        past_seq=cond_seq,
        mag_completeness=MAG_MIN,
    )
print(len(predicted_batch))


#%%
# ─── Plot: observed vs. sampled ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(7, 11))

axes[0].stem(seq.arrival_times.numpy(), seq.mag.numpy(),
             linefmt='red', markerfmt='o', basefmt=' ', label='Observed')
axes[0].axvline(condition_days, color='k', linestyle='--', label='Condition cutoff')

for i in range(predicted_batch.batch_size):
    mask    = predicted_batch.mask[i].bool()
    p_times = predicted_batch.arrival_times[i][~mask].numpy()
    p_mags  = predicted_batch.mag[i][~mask].numpy()
    axes[0].stem(p_times, p_mags, linefmt='b', markerfmt='.', basefmt=' ',
                 label='Forecast' if i == 0 else None)

axes[0].set_xlabel("Days since mainshock")
axes[0].set_ylabel("Magnitude")
axes[0].legend()

obs_mask_train = seq.arrival_times.numpy() <= condition_days
axes[1].scatter(seq.x_loc.numpy()[obs_mask_train],  seq.y_loc.numpy()[obs_mask_train],
                s=12, color='salmon', alpha=0.6, label='Observed (condition)')
axes[1].scatter(seq.x_loc.numpy()[~obs_mask_train], seq.y_loc.numpy()[~obs_mask_train],
                s=12, color='red', alpha=0.8, label='Observed (forecast period)')

for i in range(predicted_batch.batch_size):
    mask = predicted_batch.mask[i].bool()
    x = predicted_batch.x_loc[i][~mask].numpy()
    y = predicted_batch.y_loc[i][~mask].numpy()
    axes[1].scatter(x, y, s=8, alpha=0.3, color='b', label='Forecast' if i == 0 else None)

past_batch   = eq.data.Batch.from_list([cond_seq])
context_vec  = model.get_context(past_batch)[:, 0, :]
plot_xy_mixture(model, context_vec, axes[1], alpha=0.05)
axes[1].set_xlabel("East of mainshock (km)")
axes[1].set_ylabel("North of mainshock (km)")
axes[1].set_xlim([-100, 100])
axes[1].set_ylim([-100, 100])
axes[1].axhline(0, color='k', lw=0.5, ls='--')
axes[1].axvline(0, color='k', lw=0.5, ls='--')
axes[1].legend()
plt.tight_layout()
plt.show()


#%%
# ─── Magnitude-frequency distribution — observed vs. forecast ─────────────────
obs_fore_mags = seq.mag.numpy()[seq.arrival_times.numpy() > condition_days]

pred_mags_list = []
for i in range(predicted_batch.batch_size):
    mask = predicted_batch.mask[i].bool()
    pred_mags_list.append(predicted_batch.mag[i][~mask].numpy())
pred_mags = np.concatenate(pred_mags_list)

def aki_bvalue(mags, mc=MAG_MIN):
    m = mags[mags >= mc]
    return np.log10(np.e) / (m.mean() - mc) if len(m) > 1 else np.nan

def plot_mfd(ax, mags, mc, label, color):
    m = np.sort(mags[mags >= mc])[::-1]
    if len(m) == 0:
        return
    ax.semilogy(m, np.arange(1, len(m) + 1), 'o-', color=color,
                label=label, markersize=4, linewidth=1.2)

b_obs  = aki_bvalue(obs_fore_mags)
b_pred = aki_bvalue(pred_mags)

fig, ax = plt.subplots(figsize=(6, 4))
plot_mfd(ax, obs_fore_mags, MAG_MIN, f'Observed  b={b_obs:.2f}',  'red')
plot_mfd(ax, pred_mags,     MAG_MIN, f'Forecast  b={b_pred:.2f}', 'steelblue')
ax.set_xlabel('Magnitude')
ax.set_ylabel('Cumulative count  N ≥ M')
ax.set_title('Magnitude–Frequency Distribution')
ax.legend()
plt.tight_layout()
plt.show()


#%%
# ─── Spatial mixture: zero context vs. mainshock-conditioned ───────────────────
# TAU_VIZ controls which inter-event time the spatial decoder is shown at.
# Small τ (e.g. 0.1 days = 2.4 h) → near-source immediate aftershock zone.
# Large τ (e.g. 30 days) → diffuse background.
TAU_VIZ = 0.5   # days — change this to explore

model.cpu()
model.eval()

context_zero = torch.zeros(1, 1, model.context_size)

# Condition on the observed sequence (mainshock + early events) to get the
# post-mainshock context vector.
past_batch   = eq.data.Batch.from_list([cond_seq])
with torch.no_grad():
    context_cond = model.get_context(past_batch)[:, [-1], :]  # (1, 1, C)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for ax, ctx, title in [
    (axes[0], context_zero, 'Prior  (zero context)'),
    (axes[1], context_cond,
     f'After conditioning seq  ({cond_seq.num_events} events, τ_viz={TAU_VIZ} d)'),
]:
    # A few ANSS training sequences for spatial reference
    for i in range(min(8, len(anss_catalog.train))):
        s = anss_catalog.train[i]
        ax.scatter(s.x_loc.numpy(), s.y_loc.numpy(), s=2, alpha=0.12, c='grey', zorder=1)

    plot_xy_mixture(model, ctx, ax, alpha=0.05, tau=TAU_VIZ)

    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.axvline(0, color='k', lw=0.6, ls='--')
    ax.set_xlim([-250, 250])
    ax.set_ylim([-250, 250])
    ax.set_xlabel('East of mainshock (km)')
    ax.set_ylabel('North of mainshock (km)')
    ax.set_title(title)

# Overlay Ridgecrest observations on the conditioned panel
axes[1].scatter(seq.x_loc.numpy(), seq.y_loc.numpy(),
                s=seq.mag.numpy()**2, c='dodgerblue', alpha=0.7,
                edgecolors='k', linewidths=0.3, zorder=5, label='Ridgecrest M≥4.5')
axes[1].legend(fontsize=8)

plt.suptitle(f'Spatial Gaussian mixture  |  τ = {TAU_VIZ} days', fontsize=12)
plt.tight_layout()
plt.savefig(f'{FIGS_DIR}/spatial_mixture_tau{TAU_VIZ:.1f}.png', dpi=150, bbox_inches='tight')
plt.show()


#%%
# ─── Temporal mixture: zero context vs. Ridgecrest-conditioned ─────────────────
# context_zero / context_cond computed in the spatial cell above.

# Reference histograms
anss_inter = np.concatenate([
    anss_catalog.train[i].inter_times[:-1].numpy()
    for i in range(min(50, len(anss_catalog.train)))
])
# Only inter-times for events after the conditioning window, so the histogram
# is comparable to what the conditioned distribution is predicting.
post_cond_seq = seq.get_subsequence(condition_days, seq.t_end)
rc_inter = post_cond_seq.inter_times[:-1].numpy()

T_MAX  = 5.0   # x-axis limit (days) — shrink to zoom in
Y_MAX  = 30    # density y-axis

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, ctx, title, obs, obs_label in [
    (axes[0], context_zero,
     'Prior (zero context)',
     anss_inter, 'ANSS training (50 seqs)'),
    (axes[1], context_cond,
     f'After conditioning seq  ({cond_seq.num_events} events)',
     rc_inter,   f'Ridgecrest M≥{MAG_MIN} (post-conditioning)'),
]:
    plot_time_mixture(model, ctx, ax, t_max=T_MAX)
    bins = np.logspace(-3, np.log10(T_MAX * 4), 60)
    ax.hist(obs, bins=bins, density=True, alpha=0.35, color='red',
            rwidth=0.9, label=obs_label)
    ax.set_xlabel('Inter-event time (days)')
    ax.set_ylabel('Density')
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0, Y_MAX])
    ax.set_title(title)
    ax.legend(fontsize=8)

plt.suptitle('Weibull inter-event time mixture', fontsize=12)
plt.tight_layout()
plt.savefig(f'{FIGS_DIR}/time_mixture.png', dpi=150, bbox_inches='tight')
plt.show()


#%%
"""
########################################
Test effective memory
########################################
"""
# Condition on the first day of Ridgecrest (mainshock + first day of aftershocks)
condition_days = seq.t_start + 1
cond_seq_mem   = seq.get_subsequence(seq.t_start, condition_days)
MEM_DURATION   = 60    # days to simulate forward in each panel
MEM_BATCH      = 100   # trajectories

model.eval()

# Left panel: zero context — samples from the prior, no knowledge of mainshock
# Right panel: Ridgecrest-conditioned — samples after seeing day-0 → day-1

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, past, title in [
    (axes[0], None,          'Zero context (no history)'),
    (axes[1], cond_seq_mem,  f'Ridgecrest-conditioned  ({cond_seq_mem.num_events} events, day 0–1)'),
]:
    t0 = 0.0 if past is None else past.t_end
    with torch.no_grad():
        pb = model.sample(
            batch_size=MEM_BATCH,
            duration=MEM_DURATION,
            t_start=t0,
            past_seq=past,
            mag_completeness=MAG_MIN,
        )
    for i in range(len(pb)):
        mask   = pb.mask[i].bool()
        times  = pb.arrival_times[i][~mask].numpy()
        if len(times) < 2:
            continue
        pinter    = np.diff(times)
        parrival  = times[:-1] - t0   # relative to start of this panel
        ax.scatter(parrival, pinter, marker='o', s=15, c='b', alpha=0.3)

    ax.set_xlabel('Days since conditioning start')
    ax.set_ylabel('Inter-event time (days)')
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_title(title)

plt.suptitle('Effective memory — does the RNN remember the mainshock?', fontsize=12)
plt.tight_layout()
plt.savefig(f'{FIGS_DIR}/effective_memory.png', dpi=150, bbox_inches='tight')
plt.show()


#%%
"""
Generate a forecast
"""
condition_days = seq.t_start + 1
forecast_days  = seq.t_end - condition_days + 400
cond_seq = seq.get_subsequence(seq.t_start, condition_days)

model.eval()
with torch.no_grad():
    predicted_batch = model.sample(
        batch_size=500,
        duration=forecast_days,
        t_start=condition_days,
        past_seq=cond_seq,
        mag_completeness=MAG_MIN,
    )


#%%
"""
Spatial forecast — magnitude and timing per grid cell
"""
nx, ny     = 35,35
mag_thresh = MAG_MIN

# Use the test sequence spatial extent (mainshock-centered)
x_pad = max(abs(seq.x_loc.min().item()), abs(seq.x_loc.max().item())) * 1.1
y_pad = max(abs(seq.y_loc.min().item()), abs(seq.y_loc.max().item())) * 1.1
x_bins = np.linspace(-x_pad, x_pad, nx + 1)
y_bins = np.linspace(-y_pad, y_pad, ny + 1)

all_t, all_x, all_y, all_m = [], [], [], []
for i in range(predicted_batch.batch_size):
    mask = predicted_batch.mask[i].bool()
    t = predicted_batch.arrival_times[i][~mask].numpy()
    x = predicted_batch.x_loc[i][~mask].numpy()
    y = predicted_batch.y_loc[i][~mask].numpy()
    m = predicted_batch.mag[i][~mask].numpy()
    keep = m >= mag_thresh
    all_t.append(t[keep]);  all_x.append(x[keep])
    all_y.append(y[keep]);  all_m.append(m[keep])

all_t = np.concatenate(all_t) - condition_days
all_x = np.concatenate(all_x)
all_y = np.concatenate(all_y)
all_m = np.concatenate(all_m)

# Filter to events within the grid domain before binning; np.clip was
# routing out-of-domain events to boundary cells, inflating edge counts.
in_domain = (
    (all_x >= x_bins[0]) & (all_x < x_bins[-1]) &
    (all_y >= y_bins[0]) & (all_y < y_bins[-1])
)
all_t = all_t[in_domain]
all_x = all_x[in_domain]
all_y = all_y[in_domain]
all_m = all_m[in_domain]

ix = np.digitize(all_x, x_bins) - 1
iy = np.digitize(all_y, y_bins) - 1

mag_grid    = np.full((nx, ny), np.nan)
time_grid   = np.full((nx, ny), np.nan)
count_grid  = np.zeros((nx, ny))
moment_grid = np.zeros((nx, ny))

moment = 10 ** (1.5 * all_m + 9.1)

for ci in range(nx):
    for cj in range(ny):
        sel = (ix == ci) & (iy == cj)
        if sel.sum() > 0:
            mag_grid[ci, cj]    = all_m[sel].mean()
            time_grid[ci, cj]   = all_t[sel].mean()
            count_grid[ci, cj]  = sel.sum() / predicted_batch.batch_size
            moment_grid[ci, cj] = moment[sel].sum() / predicted_batch.batch_size

count_grid  = np.where(count_grid  > 0, count_grid,  np.nan)
moment_grid = np.where(moment_grid > 0, moment_grid, np.nan)

extent = [x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]]

import cartopy.feature as cfeature
from shapely.geometry import box as shapely_box

_coast = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')


def plot_coast_km(ax):
    """Overlay NaturalEarth coastline converted from lon/lat to mainshock-relative km."""
    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    lon_lo = xl / KM_PER_DEG_LON + MS_LON - 1
    lon_hi = xr / KM_PER_DEG_LON + MS_LON + 1
    lat_lo = yb / KM_PER_DEG_LAT + MS_LAT - 1
    lat_hi = yt / KM_PER_DEG_LAT + MS_LAT + 1
    bbox = shapely_box(lon_lo, lat_lo, lon_hi, lat_hi)
    for geom in _coast.geometries():
        if not geom.intersects(bbox):
            continue
        parts = geom.geoms if hasattr(geom, 'geoms') else [geom]
        for part in parts:
            coords = np.array(part.coords)
            x_c = (coords[:, 0] - MS_LON) * KM_PER_DEG_LON
            y_c = (coords[:, 1] - MS_LAT) * KM_PER_DEG_LAT
            ax.plot(x_c, y_c, 'k-', linewidth=0.7, alpha=0.8)
    ax.set_xlim(xl, xr); ax.set_ylim(yb, yt)


def add_obs(ax):
    ax.scatter(seq.x_loc.numpy(), seq.y_loc.numpy(),
               s=seq.mag.numpy()**2, c='dodgerblue', alpha=0.6,
               edgecolors='k', linewidths=0.3, label='Observed')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.axvline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel('East of mainshock (km)')
    ax.set_ylabel('North of mainshock (km)')
    ax.legend(fontsize=8)


fig, axes = plt.subplots(2, 2, figsize=(13, 10))

cmap1 = plt.cm.hot_r.copy();   cmap1.set_bad('lightgrey')
im1 = axes[0, 0].imshow(mag_grid.T, origin='lower', extent=extent,
                          aspect='auto', cmap=cmap1, vmin=mag_thresh)
plt.colorbar(im1, ax=axes[0, 0], label='Mean magnitude')
add_obs(axes[0, 0]); plot_coast_km(axes[0, 0])
axes[0, 0].set_title('Mean magnitude per cell')

cmap2 = plt.cm.viridis.copy(); cmap2.set_bad('lightgrey')
im2 = axes[0, 1].imshow(time_grid.T, origin='lower', extent=extent,
                          aspect='auto', cmap=cmap2)
plt.colorbar(im2, ax=axes[0, 1], label='Days after conditioning sequence')
add_obs(axes[0, 1]); plot_coast_km(axes[0, 1])
axes[0, 1].set_title('Mean EQ time per cell')

cmap3 = plt.cm.Blues.copy();   cmap3.set_bad('lightgrey')
im3 = axes[1, 0].imshow(count_grid.T, origin='lower', extent=extent,
                          aspect='auto', cmap=cmap3)
plt.colorbar(im3, ax=axes[1, 0], label='Mean event count')
add_obs(axes[1, 0]); plot_coast_km(axes[1, 0])
axes[1, 0].set_title('Expected event count per cell')

log_moment = np.log10(moment_grid)
cmap4 = plt.cm.YlOrRd.copy();  cmap4.set_bad('lightgrey')
im4 = axes[1, 1].imshow(log_moment.T, origin='lower', extent=extent,
                          aspect='auto', cmap=cmap4)
plt.colorbar(im4, ax=axes[1, 1], label='log$_{10}$ M$_0$ (N·m)')
add_obs(axes[1, 1]); plot_coast_km(axes[1, 1])
axes[1, 1].set_title('Total seismic moment per cell')

plt.tight_layout()
plt.savefig(f'{FIGS_DIR}/forecast_spatial.png', dpi=150)
plt.show()


#%%
"""
Paper-style M-t + cumulative forecast figure  (RECAST)
"""

def forecast_cumulative(batch, t_cut, t_grid):
    """Cumulative event count (relative to t_cut) for each trajectory at each t_grid point."""
    cum = np.zeros((batch.batch_size, len(t_grid)), dtype=int)
    for i in range(batch.batch_size):
        mask = batch.mask[i].bool()
        times = batch.arrival_times[i][~mask].numpy()
        fore_rel = np.sort(times[times > t_cut] - t_cut)
        cum[i] = np.searchsorted(fore_rel, t_grid, side='right')
    return cum

T_SHOW  = 14
N_GRID  = 400

t_cut   = condition_days
t_grid  = np.linspace(0, T_SHOW, N_GRID)
obs_rel = seq.arrival_times.numpy() - t_cut
obs_mag = seq.mag.numpy()

obs_fore_rel = np.sort(obs_rel[obs_rel > 0])
obs_cum      = np.searchsorted(obs_fore_rel, t_grid, side='right')

recast_cum = forecast_cumulative(predicted_batch, t_cut, t_grid)
r_q10, r_q25, r_q75, r_q90 = np.percentile(recast_cum, [10, 25, 75, 90], axis=0)

fig, ax = plt.subplots(figsize=(8, 4))
fig.subplots_adjust(top=0.88)
ax2 = ax.twinx()

ax.axvspan(-T_SHOW, 0, color='lightgrey', alpha=0.45, zorder=0)
ax.axvline(0, color='k', linestyle='--', linewidth=0.8, zorder=4)

in_window = (obs_rel >= -T_SHOW) & (obs_rel <= T_SHOW)
ax.scatter(obs_rel[in_window], obs_mag[in_window],
           s=obs_mag[in_window] ** 2 * 0.15, c='k', alpha=0.65, zorder=5)

ax2.fill_between(t_grid, r_q10, r_q90, alpha=0.18, color='steelblue', zorder=1)
ax2.fill_between(t_grid, r_q25, r_q75, alpha=0.38, color='steelblue', zorder=2)
ax2.step(np.concatenate([[0], t_grid]), np.concatenate([[0], obs_cum]),
         where='post', color='k', linewidth=1.5, zorder=5)

ax.set_xlim(-T_SHOW, T_SHOW)
ax.set_ylim(MAG_MIN - 0.3, MAG_MAX + 0.3)
ax.set_xlabel('Time (days)')
ax.set_ylabel('Magnitude')
ax2.set_ylabel('Cumulative number of events')
ax.text(0.02, 0.96, 'RECAST', transform=ax.transAxes,
        color='steelblue', fontsize=12, fontweight='bold', va='top')

fig.text(0.28, 0.93, 'Conditioning interval', ha='center', fontsize=10)
fig.text(0.70, 0.93, 'Forecast interval',     ha='center', fontsize=10)

plt.savefig(f'{FIGS_DIR}/forecast_summary.png', dpi=150, bbox_inches='tight')
plt.show()


#%%
"""
Evaluate conditional intensity
"""
nx_eval, ny_eval = 50, 50
_ep = 50
x_lin = np.linspace(-x_pad - _ep, x_pad + _ep, nx_eval)
y_lin = np.linspace(-y_pad - _ep, y_pad + _ep, ny_eval)
x_grid_np, y_grid_np = np.meshgrid(x_lin, y_lin)
x_grid = torch.tensor(x_grid_np, dtype=torch.float32)
y_grid = torch.tensor(y_grid_np, dtype=torch.float32)

t_eval = torch.linspace(0.01, forecast_days, 200)

model.eval()
with torch.no_grad():
    lam_intensity = model.evaluate_conditional_intensity(
        cond_seq, t_eval, x_grid, y_grid,
    )


#%%
"""
Conditional intensity — temporal decay + spatial map
"""
import matplotlib.colors

lam    = lam_intensity.cpu().numpy()
t_eval_np = t_eval.cpu().numpy()
t_last_obs = float(cond_seq.arrival_times[-1].item())

dx = float(x_lin[1] - x_lin[0])
dy = float(y_lin[1] - y_lin[0])

lambda_t  = lam.sum(axis=(-2, -1)) * dx * dy   # temporal marginal (T,)
lambda_xy = lam.mean(axis=0)                    # spatial marginal  (ny, nx)

obs_fore  = seq.arrival_times.numpy() >= condition_days
obs_t_rel = seq.arrival_times.numpy()[obs_fore] - t_last_obs
obs_x_f   = seq.x_loc.numpy()[obs_fore]
obs_y_f   = seq.y_loc.numpy()[obs_fore]
obs_m_f   = seq.mag.numpy()[obs_fore]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.semilogy(t_eval_np, lambda_t, color='steelblue', lw=1.5, label='λ(t)')
ax.set_xlabel('Days after end of conditioning sequence')
ax.set_ylabel('Predicted rate  (events / day)', color='steelblue')
ax.tick_params(axis='y', labelcolor='steelblue')
ax2 = ax.twinx()
ax2.scatter(obs_t_rel, obs_m_f, s=obs_m_f**2 * 0.4, c='k', alpha=0.45, zorder=5)
ax2.set_ylabel('Magnitude', color='k')
ax2.set_ylim(MAG_MIN - 0.5, MAG_MAX + 0.5)
ax.set_xlim(0, min(float(t_eval_np[-1]), 100))
ax.set_title('Temporal marginal  λ(t)')

ax = axes[1]
extent_ci = [x_lin[0], x_lin[-1], y_lin[0], y_lin[-1]]
im = ax.imshow(
    np.clip(lambda_xy, 1e-10, None),
    origin='lower', extent=extent_ci, aspect='auto',
    cmap='hot_r', norm=matplotlib.colors.LogNorm(),
)
plt.colorbar(im, ax=ax, label='Mean rate  (events / day / km²)')
ax.scatter(obs_x_f, obs_y_f, s=obs_m_f**2, c='dodgerblue', alpha=0.7,
           edgecolors='k', linewidths=0.3, zorder=5, label='Observed aftershocks')
ax.axhline(0, color='w', lw=0.5, ls='--'); ax.axvline(0, color='w', lw=0.5, ls='--')
ax.set_xlabel('East of mainshock (km)')
ax.set_ylabel('North of mainshock (km)')
ax.set_title('Spatial marginal  Λ(x,y)  [time-averaged]')
ax.set_xlim([-80,80])
ax.set_ylim([-80,80])
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{FIGS_DIR}/conditional_intensity.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
