#%%
import eq
import numpy as np
import torch
torch.set_num_threads(1)
import matplotlib.pyplot as plt

#%%

"""
sequences = []
for _ in range(3):
    # regenerate with different random seeds
    ...
    sequences.append(seq)

"""
n_events = 100
noise = 0.5  # controls how much scatter around the fault lines

n_events = 50

# Generate independent arrival times for each fault
ew_inter = np.random.exponential(scale=2.0, size=n_events)
ew_arrivals = np.cumsum(ew_inter) + 10.0

ns_inter = np.random.exponential(scale=2.0, size=n_events)
ns_arrivals = np.cumsum(ns_inter) + 10.0

# Spatial locations
ew_x = np.random.uniform(-5, 5, n_events)
ew_y = np.random.normal(0, 0.2, n_events)

ns_x = np.random.normal(0, 0.2, n_events)
ns_y = np.random.uniform(-5, 5, n_events)

# Merge and sort by arrival time
all_arrivals = np.concatenate([ew_arrivals, ns_arrivals])
all_x = np.concatenate([ew_x, ns_x])
all_y = np.concatenate([ew_y, ns_y])

sort_idx = np.argsort(all_arrivals)
all_arrivals = all_arrivals[sort_idx]
all_x = all_x[sort_idx]
all_y = all_y[sort_idx]

# Generate arrival times — just need monotonically increasing times
n_total = 2 * n_events
inter_times_raw = np.random.exponential(scale=1.0, size=n_total)
arrival_times = np.cumsum(inter_times_raw) + 10.0  # t_start = 10.0

t_start = 10.0
t_end = arrival_times[-1] + 1.0
inter_times = np.diff(arrival_times, prepend=[t_start], append=[t_end])

# Sort events by arrival time (faults are interleaved randomly)
sort_idx = np.argsort(arrival_times)
all_x = all_x[sort_idx]
all_y = all_y[sort_idx]

# Plot up the locations
fig,ax = plt.subplots(figsize=(8,6))
cbar = ax.scatter(all_x,all_y,marker='o',s=10,c=arrival_times,cmap='viridis')
plt.colorbar(cbar, ax=ax, label='arrival time')
plt.show()

loc_x = eq.data.ContinuousMarks(all_x,bounds = [-6,6], nll_bounds=[-5,5])
loc_y = eq.data.ContinuousMarks(all_y,bounds = [-6,6], nll_bounds = [-5,5])
mag_vals = np.random.exponential(scale=1.0, size=n_total) + 2.0
mag = eq.data.ContinuousMarks(mag_vals, bounds=[2.0, 8.0], nll_bounds=[2.5, 8.0])

# Plot up the magnitudes
fig,ax = plt.subplots(figsize=(8,6))
ax.stem(arrival_times, mag_vals, linefmt='grey', markerfmt='o', basefmt=' ')
ax.set_xlabel('Time')
ax.set_ylabel('Magnitude')
ax.set_title('Event Magnitudes')
plt.tight_layout()
plt.show()


inter_times = np.diff(arrival_times, prepend=[t_start], append=[t_end])
seq = eq.data.Sequence(inter_times, t_start=0.0, mag=mag, x_loc=loc_x, y_loc = loc_y)

# What's in seq?
print(seq.__dict__)
#%%
batch = eq.data.Batch.from_list(2*[seq])


#%%
rnn_type= "GRU"
import torch.nn as nn

context_size = 32
num_rnn_inputs=1


model = eq.models.RecurrentTPP(num_components=4,tau_mean=0.8)
#%%
epochs = 500

running_training_loss = []
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)


for epoch in range(epochs):

    optimizer.zero_grad()    
                           # zero the gradient buffers
    nll = model.nll_loss(batch).sum()               # compute the loss
    nll.backward()                                  # compute the gradients
    optimizer.step()                                # update the weights
    running_training_loss.append(nll.item())        # save the training loss
    if epoch % 100 == 0:
        print(f"Epoch:  {epoch}  Loss: {nll.item()}")


"""
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(running_training_loss)
plt.show()
"""
#%%
"""
Inference loop
"""

model.eval()
with torch.no_grad():
    predicted_batch = model.sample(
        batch_size=1,          # number of simulated futures
        duration=50.0,          # how far into the future to simulate
        t_start=seq.t_end,      # start from end of observed sequence
        past_seq=seq,           # condition on observed history
        mag_completeness=2.,   # magnitude threshold
    )
    context = model.get_context(batch)
    dist = model.get_inter_time_dist(context)
    # Sample directly from the distribution
    samples = dist.sample()
    print("Predicted mean from samples:", samples.mean().item())
    
    # Check the scale parameters directly
    params = model.hypernet_time(context)
    scale, shape, weight_logits = torch.split(
        params,
        [model.num_components, model.num_components, model.num_components],
        dim=-1
    )
    scale = torch.nn.functional.softplus(scale.clamp_min(-5.0))
    print("Mean scale:", scale.mean().item())
    print("Mean shape:", torch.nn.functional.softplus(shape.clamp_min(-5.0)).mean().item())

"""
print("tau_mean passed to model:", model.tau_mean)
print("Observed mean inter-time:", seq.inter_times[:-1].mean().item())
print("Predicted mean inter-time:", predicted_batch.inter_times[~predicted_batch.mask.bool()].mean().item())
#print("padding_mask sum:", padding_mask.sum().item())
print("end_idx:", predicted_batch.end_idx)
print("inter_times non-zero:", (predicted_batch.inter_times > 0).sum().item())
print("inter_times mean (unmasked):", predicted_batch.inter_times[predicted_batch.inter_times > 0].mean().item())
print("predicted_batch.t_start:", predicted_batch.t_start)
print("predicted_batch.t_end:", predicted_batch.t_end)
print("arrival_times[0, :5] outside sample:", predicted_batch.arrival_times[0, :5])
"""
#%%

fig, axes = plt.subplots(2, 1, figsize=(9, 8) )

# --- Arrival times (lollipop) ---
#axes[0].stem(seq.arrival_times.numpy(), seq.mag.numpy(), 
#             linefmt='grey', markerfmt='o', basefmt=' ')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Magnitude')
axes[0].set_xlim([seq.t_start, seq.t_end+160])
axes[0].set_ylim([0, 8.5])

# Overlay observed events
axes[0].stem(seq.arrival_times.numpy(), seq.mag.numpy(), 
                linefmt='red', markerfmt='o', basefmt=' ')
axes[1].scatter(seq.x_loc.numpy(), seq.y_loc.numpy(), 
                s=20, c='red', zorder=5, label='Observed')

# --- 2D locations ---
for i in range(predicted_batch.batch_size):
    mask = predicted_batch.mask[i].bool()
    x = predicted_batch.x_loc[i][~mask].numpy()
    y = predicted_batch.y_loc[i][~mask].numpy()
    p_times = predicted_batch.arrival_times[i][~mask].numpy()
    p_mags =  predicted_batch.mag[i][~mask].numpy()
    axes[1].scatter(x, y, s=10, alpha=0.5, c='steelblue')
    axes[0].stem(p_times, p_mags, 
                linefmt='grey', markerfmt='o', basefmt=' ')


axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].set_xlim([-6, 6])
axes[1].set_ylim([-6, 6])
axes[1].legend()

plt.tight_layout()
plt.show()
#%%