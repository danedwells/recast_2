#%%
import eq
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
torch.set_num_threads(1)
import matplotlib.pyplot as plt
from eq.data.synthetic import *
#%%

"""
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

seq0 = generate_sequence(
    t_start=0, 
    t_end=50,
    rate_fn=omori_rate,
    rate_kwargs=dict(K=50.0, c=0.1, p=1.1),
    geometry_fn=geometry_single_fault,
    geometry_kwargs=dict(center=(-2,0),strike=20,half_length=4, width=0.5)
)
print(f"Cross fault / Omori:    {len(seq0)} events")
seq1= generate_sequence(
    t_start = 0,
    t_end = 50,
    rate_fn=omori_rate,
    rate_kwargs=dict(K=30.0, c=0.1, p=1.1),
    geometry_fn=geometry_single_fault,
    geometry_kwargs=dict(center=(0,2),strike=90,half_length=4, width=0.5)
)
print(f"Cross fault / Omori:    {len(seq0)} events")

seq2 = generate_sequence(
    t_start=0, 
    t_end=50,
    rate_fn=omori_rate,
    rate_kwargs=dict(K=50.0, c=0.1, p=1.1),
    geometry_fn=geometry_diffuse,
    geometry_kwargs=dict(center=(3,-2),scale=1.0)
)




sequences = [seq0, seq1, seq2]#, seq1] #, seq1, seq2]
arrival_times = np.concatenate([seq.arrival_times.numpy() for seq in sequences])
mag_vals = np.concatenate([seq.mag.numpy() for seq in sequences])
all_x = np.concatenate([seq.x_loc.numpy() for seq in sequences])
all_y = np.concatenate([seq.y_loc.numpy() for seq in sequences])
t_start = sequences[0].t_start
t_end = sequences[0].t_end

xlims = sequences[0].x_loc_bounds
ylims = sequences[0].y_loc_bounds

# Plot up the locations
fig,ax = plt.subplots(figsize=(8,6))

cbar = ax.scatter(all_x,all_y,marker='o',s=10,c=arrival_times,cmap='viridis')
plt.colorbar(cbar, ax=ax, label='arrival time')
ax.set_xlim(xlims)
ax.set_ylim(ylims)
plt.show()

loc_x = eq.data.ContinuousMarks(all_x,bounds = [-6,6], nll_bounds=[-5,5])
loc_y = eq.data.ContinuousMarks(all_y,bounds = [-6,6], nll_bounds = [-5,5])
#mag_vals = np.random.exponential(scale=1.0, size=n_total) + 2.0
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
#seq = eq.data.Sequence(inter_times, t_start=0.0, mag=mag, x_loc=loc_x, y_loc = loc_y)

# What's in seq?
print(sequences[0].__dict__)
seq = sequences[0]
#%%
batch = eq.data.Batch.from_list(sequences)


#%%
rnn_type= "GRU"
import torch.nn as nn

context_size = 32
num_rnn_inputs=1


model = eq.models.RecurrentTPP(num_components_time=4,num_components_space=4,tau_mean=0.8)
#%%
epochs = 500

running_training_loss = []

# TODO - consider separate space and time optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)
scheduler = CosineAnnealingLR(optimizer, T_max = epochs)


for epoch in range(epochs):

    optimizer.zero_grad()    
                           # zero the gradient buffers
    nll = model.nll_loss(batch).sum()               # compute the loss
    nll.backward()                                  # compute the gradients
    optimizer.step()                                # update the weights
    running_training_loss.append(nll.item())        # save the training loss
    if epoch % 100 == 0:
        print(f"Epoch:  {epoch}  Loss: {nll.item()}")
    scheduler.step()


"""
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(running_training_loss)
plt.show()
"""
#%%
"""
Inference loop
"""
tc_start = 100
tc_end = 110
# cond_seq = generate_sequence(
#     t_start=100, 
#     t_end=110,
#     rate_fn=omori_rate,
#     rate_kwargs=dict(t_start = tc_start, K=50.0, c=0.1, p=1.1),
#     geometry_fn=geometry_single_fault,
#     geometry_kwargs=dict(center=(1,2),strike=100,half_length=4, width=0.2)
# )

cond_seq = generate_sequence(
    t_start=tc_start, 
    t_end=tc_end,
    rate_fn=omori_rate,
    rate_kwargs=dict(t_start = tc_start, K=200.0, c=0.1, p=1.1),
    geometry_fn=geometry_diffuse,
    geometry_kwargs=dict(center=(3,-2),scale=0.5)
)


model.eval()
with torch.no_grad():
    predicted_batch = model.sample(
        batch_size=1,          # number of simulated futures
        duration=90.0,          # how far into the future to simulate
        t_start=cond_seq.t_end,      # start from end of observed sequence
        past_seq=cond_seq,           # condition on observed history
        mag_completeness=2.,   # magnitude threshold
    )
"""    context = model.get_context(batch)
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

fig, axes = plt.subplots(2, 1, figsize=(5,9) )



# --- Arrival times (lollipop) ---
#axes[0].stem(seq.arrival_times.numpy(), seq.mag.numpy(), 
#             linefmt='grey', markerfmt='o', basefmt=' ')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Magnitude')
axes[0].set_xlim([sequences[0].t_start-5, sequences[0].t_end+160])
axes[0].set_ylim([0, 9])

# Overlay observed events
for i, sequence in enumerate(sequences):
    axes[0].stem(sequence.arrival_times.numpy(), sequence.mag.numpy(),
                 linefmt='red', markerfmt='o', basefmt=' ',
                 )
    axes[0].stem(cond_seq.arrival_times.numpy(), cond_seq.mag.numpy(),
                 linefmt='blue', markerfmt='o', basefmt=' ',
                 )
    axes[1].scatter(sequence.x_loc.numpy(), sequence.y_loc.numpy(),
                    s=10, c='red', zorder=1, alpha=0.1,
                    label='Training' if i == 0 else None)
    axes[1].scatter(cond_seq.x_loc.numpy(), cond_seq.y_loc.numpy(),
                    s=15, c='blue', zorder=2, alpha=0.15,
                    label='Conditioning' if i == 0 else None)

# --- 2D locations ---
for i in range(predicted_batch.batch_size):
    mask = predicted_batch.mask[i].bool()
    x = predicted_batch.x_loc[i][~mask].numpy()
    y = predicted_batch.y_loc[i][~mask].numpy()
    p_times = predicted_batch.arrival_times[i][~mask].numpy()
    p_mags =  predicted_batch.mag[i][~mask].numpy()
    axes[1].scatter(x, y, s=20, alpha=0.4, c='grey',zorder=3, 
                    label='Predicted' if i == 0 else None)
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