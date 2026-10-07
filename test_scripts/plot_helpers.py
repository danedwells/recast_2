import eq
import numpy as np
import torch
torch.set_num_threads(1)
import matplotlib.pyplot as plt
from eq.data.synthetic import *
import matplotlib.patches as patches
from matplotlib.patches import Ellipse


"""
########################################
# Plot Gaussian distributions fitted to data
########################################
"""
def plot_xy_mixture(model, context_vec, ax, n_std=2.0, alpha=0.3, tau=None):
    """
    Plot 2D Gaussian mixture components as ellipses.
    context_vec: single context vector, shape (1, 1, context_size)
    tau: inter-event time for spatial conditioning (None uses tau_mean → zero encoding)
    """
    with torch.no_grad():
        # hypernet_xy takes [context, encode_time(tau)]; default tau=tau_mean gives 0 encoding
        if tau is None:
            tau_enc = torch.zeros(*context_vec.shape[:-1], 1, dtype=context_vec.dtype,
                                  device=context_vec.device)
        else:
            tau_t = torch.as_tensor(tau, dtype=context_vec.dtype, device=context_vec.device)
            tau_enc = model.encode_time(tau_t).expand(*context_vec.shape[:-1], 1)
        context_input = torch.cat([context_vec, tau_enc], dim=-1)
        params = model.hypernet_xy(context_input)
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


"""
########################################
# Plot Student-t (or Gaussian) mixture as density contours
########################################
"""
def plot_xy_mixture_student_t(model, context_vec, ax, tau=None, extent=None,
                              mass_levels=(0.5, 0.9, 0.99), n_grid=300,
                              show_means=True, cmap='Blues'):
    """
    Plot the spatial mixture p(x, y | tau, context) as highest-density contours.

    Student-t components have no finite n-sigma ellipse (covariance is infinite
    for df <= 2), so instead the density is evaluated on a km grid via
    model.get_xy_dist (the exact decoder, so heavy tails are shown faithfully)
    and contoured at the density thresholds enclosing `mass_levels` of the
    probability mass inside the grid. Also works for spatial_dist='gaussian'.

    context_vec: single context vector, shape (1, 1, context_size)
    tau: inter-event time for spatial conditioning (None uses tau_mean, i.e. zero encoding)
    extent: (xmin, xmax, ymin, ymax) in km; default model.x/y_mean ± 3 std
    mass_levels: fractions of grid mass enclosed by each contour
    show_means: mark component means, sized by mixture weight
    """
    with torch.no_grad():
        ctx = context_vec.reshape(-1)                       # (context_size,)
        tau_t = (model.tau_mean if tau is None
                 else torch.as_tensor(tau)).to(dtype=ctx.dtype, device=ctx.device)
        xy_dist = model.get_xy_dist(ctx, tau_t)             # batch_shape (), standardized units

        xm, ym = model.x_mean.item(), model.y_mean.item()
        xs, ys = model.x_std.item(), model.y_std.item()
        if extent is None:
            extent = (xm - 3 * xs, xm + 3 * xs, ym - 3 * ys, ym + 3 * ys)
        gx = torch.linspace(extent[0], extent[1], n_grid)
        gy = torch.linspace(extent[2], extent[3], n_grid)
        X, Y = torch.meshgrid(gx, gy, indexing='xy')        # (n_grid, n_grid), rows = y
        xy_norm = torch.stack([(X - xm) / xs, (Y - ym) / ys], dim=-1).to(ctx)
        dens = (xy_dist.log_prob(xy_norm).exp() / (xs * ys)).cpu().numpy()  # per km²

        # Density thresholds enclosing the requested fractions of grid mass
        cell = (gx[1] - gx[0]).item() * (gy[1] - gy[0]).item()
        flat = np.sort(dens.ravel())[::-1]
        cum = np.cumsum(flat) * cell
        cum /= cum[-1]
        thresholds = [flat[min(np.searchsorted(cum, m), len(flat) - 1)] for m in mass_levels]
        order = np.argsort(thresholds)                      # contour needs ascending levels
        levels = [thresholds[i] for i in order]
        fmt = {thresholds[i]: f"{int(round(mass_levels[i] * 100))}%" for i in range(len(levels))}

        colors = plt.get_cmap(cmap)(np.linspace(0.45, 0.95, len(levels)))  # keep outer contours visible
        cs = ax.contour(X.numpy(), Y.numpy(), dens, levels=levels, colors=colors,
                        linewidths=1.2, zorder=3)
        ax.clabel(cs, cs.levels, fmt=fmt, fontsize=7)

        if show_means:
            means = xy_dist.component_distribution.loc.cpu().numpy()        # (C, 2) standardized
            w = xy_dist.mixture_distribution.probs.cpu().numpy()            # (C,)
            ax.scatter(means[:, 0] * xs + xm, means[:, 1] * ys + ym,
                       s=10 + 400 * w, c='k', alpha=0.5, zorder=4)


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
