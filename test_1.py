#%%
import eq
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
torch.set_num_threads(1)
import matplotlib.pyplot as plt
from eq.data.synthetic import *
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import os
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
    t_end=80,
    rate_fn=omori_rate,
    rate_kwargs=dict(K=100.0, c=0.1, p=1.1),
    geometry_fn=geometry_single_fault,
    geometry_kwargs=dict(center=(-2,0),strike=20,half_length=4, width=0.5),
    mag_scale=1.0,
    mag_min=2.0,
)
print(f"Cross fault / Omori:    {len(seq0)} events")
seq1= generate_sequence(
    t_start = 0,
    t_end = 80,
    rate_fn=omori_rate,
    rate_kwargs=dict(K=60.0, c=0.1, p=1.1),
    geometry_fn=geometry_single_fault,
    geometry_kwargs=dict(center=(0,2),strike=90,half_length=4, width=0.5),
    mag_scale=1.0,
    mag_min=2.0,
)
print(f"Cross fault / Omori:    {len(seq0)} events")

seq2 = generate_sequence(
    t_start=0, 
    t_end=80,
    rate_fn=omori_rate,
    rate_kwargs=dict(K=100.0, c=0.1, p=1.1),
    geometry_fn=geometry_diffuse,
    geometry_kwargs=dict(center=(3,-2),scale=1.0),
    mag_scale=1.0,
    mag_min=2.0,
)
seq3= generate_sequence(
    t_start = 50,
    t_end = 200,
    rate_fn=omori_rate,
    rate_kwargs=dict(t_start = 50, K=80.0, c=0.1, p=1.1),
    geometry_fn=geometry_single_fault,
    geometry_kwargs=dict(center=(0,2),strike=90,half_length=4, width=0.5),
    mag_scale=1.0,
    mag_min=2.0,
)



sequences = [seq0, seq1, seq2, seq3]#, seq1] #, seq1, seq2]
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

print("Total events: ", len(all_x))

#%%
batch = eq.data.Batch.from_list(sequences)

rnn_type= "GRU"
context_size = 8
num_rnn_inputs=1


model = eq.models.RecurrentTPP(
    context_size = context_size,
    num_components_time=2,
    num_components_space=3,
    tau_mean=0.8)
save_dir = "test_1"

#%%
"""
Training Loop
"""

_RETRAIN_ = False

if _RETRAIN_:
    epochs = 500

    running_training_loss = []

    # TODO - consider separate space and time optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max = epochs)

    device = 'cuda' if torch.cuda.is_available else 'cpu'
    print(f"Training on device: {device}")
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



    os.makedirs(f"trained_models/{save_dir}", exist_ok=True)
    torch.save(model.state_dict(), f"trained_models/{save_dir}/model_f.pt")

#%%
"""
Inference loop
"""
model.load_state_dict(torch.load(f"trained_models/{save_dir}/model_f.pt"))

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

# cond_seq = generate_sequence(
#     t_start=tc_start, 
#     t_end=tc_end,
#     rate_fn=omori_rate,
#     rate_kwargs=dict(t_start = tc_start, K=50.0, c=0.1, p=1.1),
#     geometry_fn=geometry_diffuse,
#     geometry_kwargs=dict(center=(3,-2),scale=0.5)
# )

cond_seq = None
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total:,}")
print(f"Trainable parameters: {trainable:,}")

model.eval()
with torch.no_grad():
    predicted_batch = model.sample(
        batch_size=1,          # number of simulated futures
        duration=300.0,          # how far into the future to simulate
        t_start= 100, #cond_seq.t_end,      # start from end of observed sequence
        past_seq=None, # cond_seq,           # condition on observed history
        mag_completeness=2.,   # magnitude threshold
    )

#%%

fig, axes = plt.subplots(2, 1, figsize=(5,9) )

axes[0].set_xlabel('Time')
axes[0].set_ylabel('Magnitude')
axes[0].set_xlim([sequences[0].t_start-5, sequences[0].t_end+160])
axes[0].set_ylim([0, 9])

# Overlay observed events
for i, sequence in enumerate(sequences):
    axes[0].stem(sequence.arrival_times.numpy(), sequence.mag.numpy(),
                 linefmt='red', markerfmt='o', basefmt=' ',
                 )

    axes[1].scatter(sequence.x_loc.numpy(), sequence.y_loc.numpy(),
                    s=10, c='red', zorder=1, alpha=0.1,
                    label='Training' if i == 0 else None)
    if cond_seq is not None:
        axes[0].stem(cond_seq.arrival_times.numpy(), cond_seq.mag.numpy(),
                    linefmt='blue', markerfmt='o', basefmt=' ',
                    )
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
"""
########################################
# Plot Gaussian distributions fitted to data
########################################
"""
def plot_xy_mixture(model, context_vec, ax, n_std=2.0, alpha=0.3):
    """
    Plot 2D Gaussian mixture components as ellipses.
    context_vec: single context vector, shape (1, 1, context_size)
    """
    with torch.no_grad():
        params = model.hypernet_xy(context_vec)
        C = model.num_components_space

        means, l_diag, l_offdiag, weight_logits = torch.split(
            params, [2*C, 2*C, C, C], dim=-1
        )
        means = means.unflatten(-1, (C, 2)).squeeze()          # (C, 2)
        l_diag = torch.nn.functional.softplus(l_diag.clamp_min(-5.0))
        l_diag = l_diag.unflatten(-1, (C, 2)).squeeze()        # (C, 2)
        l_offdiag = l_offdiag.unflatten(-1, (C, 1)).squeeze()  # (C,)
        weights = torch.softmax(weight_logits.squeeze(), dim=-1)  # (C,)

        for k in range(C):
            # Reconstruct L and covariance
            L = torch.zeros(2, 2)
            L[0, 0] = l_diag[k, 0]
            L[1, 1] = l_diag[k, 1]
            L[1, 0] = l_offdiag[k]
            cov = (L @ L.T).numpy()

            mean = means[k].numpy()
            weight = weights[k].item()

            # Denormalize mean back to original space
            mean[0] = mean[0] * model.x_std.item() + model.x_mean.item()
            mean[1] = mean[1] * model.y_std.item() + model.y_mean.item()

            # Eigendecomposition for ellipse orientation
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 2 * n_std * np.sqrt(eigenvalues)

            ellipse = Ellipse(
                xy=mean,
                width=width * model.x_std.item(),   # scale back to data space
                height=height * model.y_std.item(),
                angle=angle,
                alpha=0.5*alpha * weight * C,            # weight controls opacity
                color='blue',
                zorder=3,
            )
            ax.add_patch(ellipse)

#%%
# Usage — use the zero context (prior, no history) or a specific event's context
context_vec = torch.zeros(1, 1, model.context_size)


# Specific event - maybe most recent event? This *does* affect it, which is good to know
past_batch = eq.data.Batch.from_list([sequences[3]])
context_vec = model.get_context(past_batch)[:,-1,:] # take last event


fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(all_x, all_y, s=5, alpha=0.3, c='red', label='Training data')
plot_xy_mixture(model, context_vec, ax)
ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
plt.tight_layout()
plt.show()
# %%
"""
########################################
# Plot Weibull distributions fitted to data
########################################
"""
def plot_time_mixture(model, context_vec, ax, t_max=10.0, n_points=500):
    """
    Plot Weibull mixture density over time.
    context_vec: single context vector, shape (1, 1, context_size)
    """
    with torch.no_grad():
        params = model.hypernet_time(context_vec)
        C = model.num_components_time

        scale, shape, weight_logits = torch.split(
            params,
            [C, C, C],
            dim=-1
        )
        scale = torch.nn.functional.softplus(scale.clamp_min(-5.0)).squeeze()  # (C,)
        shape = torch.nn.functional.softplus(shape.clamp_min(-5.0)).squeeze()  # (C,)
        weights = torch.softmax(weight_logits.squeeze(), dim=-1)               # (C,)

        # Evaluate mixture density over time grid
        t = torch.linspace(1e-3, t_max, n_points)  # (T,)
        
        # Weibull log_prob for each component — shape (T, C)
        t_expanded = t.unsqueeze(-1).expand(-1, C)
        component_log_prob = (
            torch.log(shape)
            + (shape - 1) * torch.log(t_expanded / scale)
            - (t_expanded / scale) ** shape
            - torch.log(scale)
        )
        
        # Mixture density
        log_weights = torch.log(weights).unsqueeze(0)  # (1, C)
        mixture_log_prob = torch.logsumexp(component_log_prob + log_weights, dim=-1)  # (T,)
        mixture_prob = mixture_log_prob.exp().numpy()

        # Plot mixture and individual components
        ax.plot(t.numpy(), mixture_prob, 'b-', linewidth=2, label='Mixture')
        for k in range(C):
            component_prob = (weights[k] * component_log_prob[:, k].exp()).numpy()
            ax.plot(t.numpy(), component_prob, 'b-', alpha=0.2, linewidth=0.5)

# Usage — use the zero context (prior, no history) or a specific event's context
context_vec = torch.zeros(1, 1, model.context_size)


# Specific event - maybe most recent event? This *does* affect it, which is good to know
past_batch = eq.data.Batch.from_list([sequences[0]])
context_vec = model.get_context(past_batch)[:,-1,:] # take last event



fig, ax = plt.subplots(figsize=(8, 4))
plot_time_mixture(model, context_vec, ax, t_max=5.0)

# Overlay observed inter-time histogram
all_inter = np.concatenate([seq.inter_times[:-1].numpy() for seq in sequences])
ax.hist(all_inter, bins=100, density=True, alpha=0.3, color='red', label='Observed')

ax.set_xlabel('Inter-event time')
ax.set_ylabel('Density')
ax.set_xlim([0, 5.0])
ax.legend()
plt.tight_layout()
plt.show()
# %%
"""
########################################
Test effective memory
########################################
"""
tc_start = 200
tc_end = 205
cond_seq = generate_sequence(
    t_start=tc_start, 
    t_end=tc_end,
    rate_fn=omori_rate,
    rate_kwargs=dict(t_start = tc_start, K=50.0, c=0.1, p=1.1),
    geometry_fn=geometry_diffuse,
    geometry_kwargs=dict(center=(3,-2),scale=0.5),
    mag_scale=1.0,
    mag_min=2.0,
)

model.eval()
with torch.no_grad():
    predicted_batch = model.sample(
        batch_size=1,          # number of simulated futures
        duration=300.0,          # how far into the future to simulate
        t_start= 100, #cond_seq.t_end,      # start from end of observed sequence
        past_seq=cond_seq, # cond_seq,           # condition on observed history
        mag_completeness=2.,   # magnitude threshold
    )

mask = predicted_batch.mask[i].bool()
pinter_times = np.diff(predicted_batch.arrival_times[i][~mask].numpy())
parrival_times = predicted_batch.arrival_times[i][~mask].numpy()[:-1]

fig,ax = plt.subplots(figsize=(8,6))
ax.scatter(parrival_times,pinter_times,marker='o',s=20)
ax.set_xlabel("Arrival Times")
ax.set_ylabel("Inter times")
plt.show()



#%%

model.eval()
with torch.no_grad():
    # Build context from real sequence history
    past_batch = eq.data.Batch.from_list([sequences[3]])
    context = model.get_context(past_batch)  # (1, L, context_size)
    
    # Take context after the last real event
    last_context = context[:, [-1], :]  # (1, 1, context_size)
    h0 = last_context.transpose(0, 1).contiguous()  # (1, 1, context_size) for RNN

    # # How much does a single past event shift the predicted distribution?
    # context_zero = torch.zeros(1, 1, model.context_size)

    # Feed one event at varying distances in the past
    predicted_means = []
    lags = np.logspace(-2,3,50)
    for lag in lags:
        
        fake_inter = torch.tensor([[lag]]).float()          # (1, 1)
        fake_mag = torch.tensor([[7.0]]).float()            # (1, 1)
        fake_x = torch.tensor([[2.0]]).float()             # (1, 1)
        fake_y = torch.tensor([[-2.0]]).float()             # (1, 1)
        
        rnn_input = torch.cat([
            model.encode_time(fake_inter),                                    # (1, 1, 1)
            model.encode_magnitude(fake_mag, torch.tensor([2.0])),            # (1, 1, 1)
            model.encode_xy(fake_x, fake_y)                                   # (1, 1, 2)
        ], dim=-1)   
        
        context_after, _ = model.rnn(rnn_input,h0)
        # Does it make sense to run a single event...?
        # Maybe we want a whole series of events, then look at what happens
        # as t_sincelastevent -> infinity

        dist = model.get_inter_time_dist(context_after)
        samples = dist.sample((1000,))
        #print(f"Lag {lag:.1f}: predicted mean = {dist.sample().mean().item():.3f}")
        mean = dist.sample().mean().item()
        mean = samples.mean().item()
        predicted_means.append(mean)

fig,ax = plt.subplots(figsize=(8,6))
ax.semilogx(lags,np.array(predicted_means),c='k',linewidth=2)
ax.set_ylabel("Predicted mean")
ax.set_ylim([0,40])
plt.show()

#%%