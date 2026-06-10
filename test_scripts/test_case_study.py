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
from plot_helpers import *

# ─── Data loading ─────────────────────────────────────────────────────────────

HOME_DIR = "/home/a01738353/2024_NEHRP/RECAST/"

# Test sequence
DATA_DIR = f"{HOME_DIR}case_studies/Ferndale_2022_catalog.parquet"

# Background (training?) sequence
BACK_DIR = f"{HOME_DIR}case_studies/background_seismicity.parquet"

# Output directories
SAVE_DIR = f"{HOME_DIR}results/case_studies/Ferndale"
FIGS_DIR = f"{HOME_DIR}figs/case_studies/Ferndale"
for directory in [SAVE_DIR,FIGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Set the reference location in lat, lon. Convert to cartesian
_sf = 0.1 # scale factor to compress distances
KM_PER_DEG_LAT = 111.1 * _sf
LAT_CENTROID = 40.5 #35.8006 
LON_CENTROID = -124.3 #-117.6130
KM_PER_DEG_LON = KM_PER_DEG_LAT * np.cos(np.radians(LAT_CENTROID))

MAG_MIN = 3.0
MAG_MAX = 8.0

# Bounds: data extent + 20% padding
def padded_bounds(vals, frac=0.2):
    lo, hi = vals.min(), vals.max()
    pad = (hi - lo) * frac
    return [float(lo - pad), float(hi + pad)]


def load_sequence(path,bounds=None, t_start = None):
    df = pd.read_parquet(path)
    df = df.sort_values("time").reset_index(drop=True)

    if bounds is not None:
        #Filter by bounds
        df = df[df['latitude'].between(bounds[0],bounds[1]) & 
                df['longitude'].between(bounds[2],bounds[3])]

    # Time: fractional days from first event
    t0 = df["time"].iloc[0]
    arrival_times = (df["time"] - t0).dt.total_seconds().values / 86400.0

    # Shift by 1 hour so the first event has a positive inter-time from t_start=0
    offset = 1.0 / 24.0
    arrival_times = arrival_times + offset
    if t_start is None:
        t_start = 0.0
    arrival_times += t_start

    # Add a 1-hour trailing window so the survival inter-time is also positive
    t_end = float(arrival_times[-1]) + offset 
    inter_times = np.diff(arrival_times, prepend=[t_start], append=[t_end])

    # Space: km from centroid
    x_km = ((df["longitude"].values - LON_CENTROID) * KM_PER_DEG_LON).astype(np.float32)
    y_km = ((df["latitude"].values - LAT_CENTROID) * KM_PER_DEG_LAT).astype(np.float32)

    # put some bounds on the geographical extent
    x_bounds = padded_bounds(x_km)
    y_bounds = padded_bounds(y_km)

    mag_vals = df["mag"].values.astype(np.float32)
    mag = eq.data.ContinuousMarks(mag_vals, bounds=[MAG_MIN, MAG_MAX], nll_bounds=[MAG_MIN, MAG_MAX])
    x_loc = eq.data.ContinuousMarks(x_km, bounds=x_bounds, nll_bounds=x_bounds)
    y_loc = eq.data.ContinuousMarks(y_km, bounds=y_bounds, nll_bounds=y_bounds)

    seq = eq.data.Sequence(inter_times, t_start=t_start, mag=mag, x_loc=x_loc, y_loc=y_loc)
    return seq, df, x_bounds, y_bounds, t_start, t_end


#%%

df = pd.read_parquet(DATA_DIR)
bounds = [df['latitude'].min()-0.1, df['latitude'].max()+0.1,
          df['longitude'].min()-0.1, df['longitude'].max()+0.1]
#bounds = [y_bounds[0],y_bounds[1],x_bounds[0],x_bounds[1]]
bseq, bdf, _, _, _, t_end, = load_sequence(path=BACK_DIR, bounds = bounds)

seq, df, _, _, _, _ = load_sequence(path=DATA_DIR, t_start = t_end)
print(f"Events: {seq.num_events}")
print(f"Duration: {seq.t_end:.2f} days")
print(f"Mean inter-event: {seq.inter_times[:-1].mean().item():.4f} days")
print(f"x range: [{seq.x_loc.min().item():.1f}, {seq.x_loc.max().item():.1f}] km")
print(f"y range: [{seq.y_loc.min().item():.1f}, {seq.y_loc.max().item():.1f}] km")




# Quick look at the sequence
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].stem(seq.arrival_times.numpy(), seq.mag.numpy(), linefmt='grey', markerfmt='o', basefmt=' ')
axes[0].set_xlabel("Days since mainshock")
axes[0].set_ylabel("Magnitude")
axes[0].set_title("Ferndale — M-t plot")

axes[1].scatter(seq.x_loc.numpy(), seq.y_loc.numpy(), s=seq.mag.numpy()**2,
                c=seq.arrival_times.numpy(), cmap='viridis', alpha=0.6)
axes[1].set_xlabel("East (km)")
axes[1].set_ylabel("North (km)")
axes[1].set_title("Ferndale — Spatial")
plt.colorbar(axes[1].collections[0], ax=axes[1], label="Days since mainshock")
plt.tight_layout()
plt.show()

# Quick look at the background (training?)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].stem(bseq.arrival_times.numpy(), bseq.mag.numpy(), linefmt='grey', markerfmt='o', basefmt=' ')
axes[0].set_xlabel("Days since mainshock")
axes[0].set_ylabel("Magnitude")
axes[0].set_title("background — M-t plot")

axes[1].scatter(bseq.x_loc.numpy(), bseq.y_loc.numpy(), s=bseq.mag.numpy()**2,
                c=bseq.arrival_times.numpy(), cmap='viridis', alpha=0.6)
axes[1].set_xlabel("East (km)")
axes[1].set_ylabel("North (km)")
axes[1].set_title("Background — Spatial")
plt.colorbar(axes[1].collections[0], ax=axes[1], label="Days since mainshock")
plt.tight_layout()
plt.show()


#%%
batch = eq.data.Batch.from_list([bseq])

# Spatial normalization stats derived from data
x_mean = float(bseq.x_loc.mean().item())
y_mean = float(bseq.y_loc.mean().item())
x_std  = float(bseq.x_loc.std().item())
y_std  = float(bseq.y_loc.std().item())
tau_mean = float(bseq.inter_times[:-1].mean().item())

print(f"x_mean={x_mean:.2f}  x_std={x_std:.2f}")
print(f"y_mean={y_mean:.2f}  y_std={y_std:.2f}")
print(f"tau_mean={tau_mean:.4f} days")

model = eq.models.RecurrentTPP(
    context_size=64,
    num_components_time=32,
    num_components_space=32,
    tau_mean=tau_mean,
    x_mean=x_mean,
    y_mean=y_mean,
    x_std=x_std,
    y_std=y_std,
)



#%%
"""
Training Loop
"""

_RETRAIN_ = True
model.train()

if _RETRAIN_:
    epochs = 2000
    running_training_loss = []

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    batch = batch.to(device)
    print(f"Training on: {device}")

    for epoch in range(epochs):
        optimizer.zero_grad()
        nll = model.nll_loss(batch).sum()
        nll.backward()
        optimizer.step()
        running_training_loss.append(nll.item())
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}  NLL: {nll.item():.4f}")
        scheduler.step()

    torch.save(model.state_dict(), f"{SAVE_DIR}/model.pt")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(running_training_loss)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL")
    ax.set_title("Training loss")
    plt.tight_layout()
    plt.show()


#%%
"""
Inference — condition on the observed sequence, sample forward
"""
model = model.cpu()
batch = batch.cpu()
map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
model.load_state_dict(torch.load(f"{SAVE_DIR}/model.pt", map_location=map_location))
model.eval()

total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total:,}")

# Condition on the first 20 days, forecast the remaining ~6 days
condition_days = seq.t_start + 1
forecast_days = seq.t_end - condition_days
cond_seq = seq.get_subsequence(seq.t_start, condition_days)

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
# Plot: observed vs. sampled
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Magnitude-time
for i in range(predicted_batch.batch_size):
    mask = predicted_batch.mask[i].bool()
    p_times = predicted_batch.arrival_times[i][~mask].numpy()
    p_mags  = predicted_batch.mag[i][~mask].numpy()
    axes[0].stem(p_times, p_mags, linefmt='lightblue', markerfmt='.', basefmt=' ', label='Forecast' if i == 0 else None)

axes[0].stem(seq.arrival_times.numpy(), seq.mag.numpy(), linefmt='red', markerfmt='o', basefmt=' ', label='Observed')
axes[0].axvline(condition_days, color='k', linestyle='--', label='Condition cutoff')
axes[0].set_xlabel("Days since mainshock")
axes[0].set_ylabel("Magnitude")
axes[0].legend()

# Spatial
for i in range(predicted_batch.batch_size):
    mask = predicted_batch.mask[i].bool()
    x = predicted_batch.x_loc[i][~mask].numpy()
    y = predicted_batch.y_loc[i][~mask].numpy()
    axes[1].scatter(x, y, s=8, alpha=0.3, color='lightblue', label='Forecast' if i == 0 else None)

# Observed: pre- and post-cutoff in different colors
obs_mask_train = seq.arrival_times.numpy() <= condition_days
axes[1].scatter(seq.x_loc.numpy()[obs_mask_train],  seq.y_loc.numpy()[obs_mask_train],
                s=12, color='salmon', alpha=0.6, label='Observed (condition)')
axes[1].scatter(seq.x_loc.numpy()[~obs_mask_train], seq.y_loc.numpy()[~obs_mask_train],
                s=12, color='red', alpha=0.8, label='Observed (forecast period)')

axes[1].set_xlabel("East (km)")
axes[1].set_ylabel("North (km)")
axes[1].legend()
plt.tight_layout()
plt.show()

# %%


# Usage — use the zero context (prior, no history) or a specific event's context
context_vec = torch.zeros(1, 1, model.context_size)


# Specific event - maybe most recent event? This *does* affect it, which is good to know
past_batch = eq.data.Batch.from_list([bseq])
context_vec = model.get_context(past_batch)[:,-1,:] # take last event


fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(bseq.x_loc, bseq.y_loc, s=
           5, alpha=0.3, c='red', label='Training data')
plot_xy_mixture(model, context_vec, ax)
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
plt.tight_layout()
plt.show()

#%%

# Usage — use the zero context (prior, no history) or a specific event's context
context_vec = torch.zeros(1, 1, model.context_size)


# Specific event - maybe most recent event? This *does* affect it, which is good to know
past_batch = eq.data.Batch.from_list([bseq])
context_vec = model.get_context(past_batch)[:,-1,:] # take last event



fig, ax = plt.subplots(figsize=(8, 4))
plot_time_mixture(model, context_vec, ax, t_max=5.0)

# Overlay observed inter-time histogram
all_inter = bseq.inter_times[:-1].numpy()
bins = np.logspace(-2,3,100)
ax.hist(all_inter, bins=bins, density=True, alpha=0.3, color='red', rwidth=0.9,
         label='Observed')

ax.set_xlabel('Inter-event time')
ax.set_ylabel('Density')
ax.set_xlim([0, 0.5])
ax.set_ylim([0,30])
ax.legend()
plt.tight_layout()
plt.show()

#%%

"""
########################################
Test effective memory
########################################
"""

# Condition on the first 20 days, forecast the remaining ~6 days
condition_days = seq.t_start + 1
forecast_days = seq.t_end - condition_days + 400
cond_seq = seq.get_subsequence(seq.t_start, condition_days)

model.eval()
with torch.no_grad():
    predicted_batch = model.sample(
        batch_size=100,          # number of simulated futures
        duration=forecast_days,
        t_start=condition_days,
        past_seq=cond_seq,       # condition on observed history
        mag_completeness=3.,   # magnitude threshold
    )

fig,ax = plt.subplots(figsize=(8,6))
for i in range(len(predicted_batch)):

    mask = predicted_batch.mask[i].bool()
    pinter_times = np.diff(predicted_batch.arrival_times[i][~mask].numpy())
    parrival_times = predicted_batch.arrival_times[i][~mask].numpy()[:-1]


    ax.scatter(parrival_times,pinter_times,marker='o',s=20, c='b',alpha=0.5)
ax.set_xlabel("Arrival Times")
ax.set_xlim([7650, 7850])
ax.set_ylim([0,80])
ax.set_ylabel("Inter times")
plt.show()

fig.savefig("test_effective_memory.png")

# %%
"""
Generate a forecast
"""

# Condition on the first 20 days, forecast the remaining ~6 days
condition_days = seq.t_start + 1
forecast_days = seq.t_end - condition_days + 400
cond_seq = seq.get_subsequence(seq.t_start, condition_days)

model.eval()
with torch.no_grad():
    predicted_batch = model.sample(
        batch_size=500,          # number of simulated futures
        duration=forecast_days,
        t_start=condition_days,
        past_seq=cond_seq,       # condition on observed history
        mag_completeness=3.,   # magnitude threshold
    )

# %%
"""
Spatial forecast — magnitude and timing per grid cell
"""

nx, ny     = 6, 6
mag_thresh = 3.0

x_bins = np.linspace(bseq.x_loc.min(), bseq.x_loc.max(), nx + 1)
y_bins = np.linspace(bseq.y_loc.min(), bseq.y_loc.max(), ny + 1)

# Pool all events across trajectories
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

all_t = np.concatenate(all_t) - condition_days   # days since conditioning
all_x = np.concatenate(all_x)
all_y = np.concatenate(all_y)
all_m = np.concatenate(all_m)

ix = np.clip(np.digitize(all_x, x_bins) - 1, 0, nx - 1)
iy = np.clip(np.digitize(all_y, y_bins) - 1, 0, ny - 1)

mag_grid    = np.full((nx, ny), np.nan)
time_grid   = np.full((nx, ny), np.nan)
count_grid  = np.zeros((nx, ny))
moment_grid = np.zeros((nx, ny))

# Hanks-Kanamori: M0 (N·m) = 10^(1.5*M + 9.1)
moment = 10 ** (1.5 * all_m + 9.1)

for ci in range(nx):
    for cj in range(ny):
        sel = (ix == ci) & (iy == cj)
        if sel.sum() > 0:
            mag_grid[ci, cj]    = all_m[sel].mean()
            time_grid[ci, cj]   = all_t[sel].mean()
            count_grid[ci, cj]  = sel.sum() / predicted_batch.batch_size  # mean count per trajectory
            moment_grid[ci, cj] = moment[sel].sum() / predicted_batch.batch_size

# Use NaN for empty cells on all grids
count_grid  = np.where(count_grid  > 0, count_grid,  np.nan)
moment_grid = np.where(moment_grid > 0, moment_grid, np.nan)

extent = [x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]]

import cartopy.feature as cfeature
from shapely.geometry import box as shapely_box

_coast = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')

def plot_coast_km(ax):
    """Overlay NaturalEarth coastline on an axis whose coordinates are in compressed km."""
    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    # Convert axis extent back to lon/lat with 1-degree padding for clipping
    lon_lo = xl / KM_PER_DEG_LON + LON_CENTROID - 1
    lon_hi = xr / KM_PER_DEG_LON + LON_CENTROID + 1
    lat_lo = yb / KM_PER_DEG_LAT + LAT_CENTROID - 1
    lat_hi = yt / KM_PER_DEG_LAT + LAT_CENTROID + 1
    bbox = shapely_box(lon_lo, lat_lo, lon_hi, lat_hi)
    for geom in _coast.geometries():
        if not geom.intersects(bbox):
            continue
        parts = geom.geoms if hasattr(geom, 'geoms') else [geom]
        for part in parts:
            coords = np.array(part.coords)
            x_c = (coords[:, 0] - LON_CENTROID) * KM_PER_DEG_LON
            y_c = (coords[:, 1] - LAT_CENTROID) * KM_PER_DEG_LAT
            ax.plot(x_c, y_c, 'k-', linewidth=0.7, alpha=0.8)
    ax.set_xlim(xl, xr);  ax.set_ylim(yb, yt)   # restore — coast can nudge limits

def add_obs(ax):
    ax.scatter(seq.x_loc.numpy(), seq.y_loc.numpy(),
               s=seq.mag.numpy()**2, c='dodgerblue', alpha=0.6,
               edgecolors='k', linewidths=0.3, label='Observed')
    ax.set_xlabel('East (km)');  ax.set_ylabel('North (km)')
    ax.legend(fontsize=8)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Panel 1 — mean magnitude
cmap1 = plt.cm.hot_r.copy();  cmap1.set_bad('lightgrey')
im1 = axes[0, 0].imshow(mag_grid.T, origin='lower', extent=extent,
                         aspect='auto', cmap=cmap1, vmin=mag_thresh)
plt.colorbar(im1, ax=axes[0, 0], label='Mean magnitude')
add_obs(axes[0, 0])
plot_coast_km(axes[0, 0])
axes[0, 0].set_title('Mean magnitude per cell')

# Panel 2 — mean arrival time
cmap2 = plt.cm.viridis.copy();  cmap2.set_bad('lightgrey')
im2 = axes[0, 1].imshow(time_grid.T, origin='lower', extent=extent,
                         aspect='auto', cmap=cmap2)
plt.colorbar(im2, ax=axes[0, 1], label='Days after conditioning sequence')
add_obs(axes[0, 1])
plot_coast_km(axes[0, 1])
axes[0, 1].set_title('Mean EQ time per cell')

# Panel 3 — mean event count per trajectory
cmap3 = plt.cm.Blues.copy();  cmap3.set_bad('lightgrey')
im3 = axes[1, 0].imshow(count_grid.T, origin='lower', extent=extent,
                         aspect='auto', cmap=cmap3)
plt.colorbar(im3, ax=axes[1, 0], label='Mean event count')
add_obs(axes[1, 0])
plot_coast_km(axes[1, 0])
axes[1, 0].set_title('Expected event count per cell')

# Panel 4 — total seismic moment (log10)
log_moment = np.log10(moment_grid)
cmap4 = plt.cm.YlOrRd.copy();  cmap4.set_bad('lightgrey')
im4 = axes[1, 1].imshow(log_moment.T, origin='lower', extent=extent,
                         aspect='auto', cmap=cmap4)
plt.colorbar(im4, ax=axes[1, 1], label='log$_{10}$ M$_0$ (N·m)')
add_obs(axes[1, 1])
plot_coast_km(axes[1, 1])
axes[1, 1].set_title('Total seismic moment per cell')

plt.tight_layout()
plt.savefig(f'{FIGS_DIR}/forecast_spatial.png', dpi=150)
plt.show()

# %%
