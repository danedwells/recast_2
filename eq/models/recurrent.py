from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from sklearn.preprocessing import StandardScaler

import eq
import eq.distributions as dist

from eq.models.tpp_model import TPPModel


class RecurrentTPP(TPPModel):

    """Neural TPP model with an recurrent encoder.
    TPP = temporal point process

    Args:
        input_magnitude: Should magnitude be used as model input?
        predict_magnitude: Should the model predict the magnitude?
        num_extra_features: Number of extra features to use as input.
        context_size: Size of the RNN hidden state.
        num_components: Number of mixture components in the output distribution.
        rnn_type: Type of the RNN. Possible choices {'GRU', 'RNN', 'LSTM'}
        dropout_proba: Dropout probability.
        tau_mean: Mean inter-event times in the dataset.
        mag_mean: Mean earthquake magnitude in the dataset.                             # FLAG
        richter_b: Fixed b value of the Gutenberg-Richter distribution for magnitudes.
        mag_completeness: Magnitude of completeness of the catalog.
        learning_rate: Learning rate used in optimization.
    """

    def __init__(
        self,
        input_magnitude: bool = True,
        predict_magnitude: bool = True,
        num_extra_features: Optional[int] = None,
        context_size: int = 32,
        num_components_time: int = 32,
        num_components_space: int = 32,
        rnn_type: str = "GRU",
        dropout_proba: float = 0.5,
        tau_mean: float = 1.0,
        x_mean: float = 0.0,
        y_mean: float = 0.0,
        x_std: float = 1.0,
        y_std: float = 1.0,
        richter_b: float = 1.0,
        learning_rate: float = 5e-2,
        *args,
        **kwargs
    ):
        super().__init__()
        self.debug = kwargs.get("debug", False)

        self.input_magnitude = input_magnitude
        self.predict_magnitude = predict_magnitude
        self.num_extra_features = num_extra_features
        self.context_size = context_size
        self.num_components_time = num_components_time
        self.num_components_space = num_components_space
        self.register_buffer("tau_mean", torch.tensor(tau_mean, dtype=torch.float64))
        self.register_buffer("log_tau_mean", self.tau_mean.log())
        self.register_buffer("richter_b", torch.tensor(richter_b, dtype=torch.float64))
        
        # Spatial encoding statistics (register as buffers like tau_mean)
        self.register_buffer("x_mean", torch.tensor(x_mean, dtype=torch.float32))
        self.register_buffer("x_std",  torch.tensor(x_std,  dtype=torch.float32))
        self.register_buffer("y_mean", torch.tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std",  torch.tensor(y_std,  dtype=torch.float32))
        
        
        self.learning_rate = learning_rate

        # Decoder for the time distribution
        self.num_time_params = 3 * self.num_components_time
        self.hypernet_time = nn.Linear(context_size, self.num_time_params)

        # Option 1
        # Decoder for the spatial distribution - two hypernets, separates x and y
        # self.num_x_params = 3 * self.num_components_space  # mean, log_std, weight per component
        # self.num_y_params = 3 * self.num_components_space
        # self.hypernet_x = nn.Linear(context_size, self.num_x_params)
        # self.hypernet_y = nn.Linear(context_size, self.num_y_params)

        # Option 2
        # Decoder for the spatial distribution - combined with covariance.
        self.num_xy_params = self.num_components_space * (
            2          # mean vector (mu_x, mu_y)
            + 2        # log-diagonal of L (l_11, l_22) — softplus'd to stay positive
            + 1        # off-diagonal of L (l_21) — unconstrained
            + 1        # mixture weight logit
        )
        # Option B — joint spatio-temporal distribution: p(τ,x,y|c) = p(τ|c)·p(x,y|τ,c).
        # Spatial decoder now takes context + encoded τ so the mixture means/covariances
        # can vary with inter-event time, capturing aftershock clustering (small τ → near
        # source, large τ → diffuse background).  Revert by restoring context_size below.
        # self.hypernet_xy = nn.Linear(context_size, self.num_xy_params)
        self.hypernet_xy = nn.Linear(context_size + 1, self.num_xy_params)
        self.scale = StandardScaler()

        # RNN input features
        if self.input_magnitude:
            # Decoder for 
            self.num_mag_params = 1  # (1 rate)
            self.hypernet_mag = nn.Linear(context_size, self.num_mag_params)

        if rnn_type not in ["RNN", "GRU", "LSTM"]:
            raise ValueError(
                f"rnn_type must be one of ['RNN', 'GRU', 'LSTM'] " f"(got {rnn_type})"
            )
        """self.num_rnn_inputs = (
            1
            + int(self.input_magnitude)
            + (0 if self.num_extra_features is None else self.num_extra_features)
        )"""

        self.num_rnn_inputs = (
            1                                        # time
            + int(self.input_magnitude)              # magnitude
            + 2                                      # x and y
            + (0 if self.num_extra_features is None else self.num_extra_features)
        )

        # Initialize the RNN model - this is a single layer, one of RNN GRU or LSTM
        # TODO - investigate more complicated models?
        self.rnn = getattr(nn, rnn_type)(
            self.num_rnn_inputs,
            context_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_proba)

    def encode_time(self, inter_times):
        # inter_times has shape (...)
        # output has shape (..., 1)
        log_tau = torch.log(torch.clamp_min(inter_times, 1e-10)).unsqueeze(-1)
        return log_tau - self.log_tau_mean
    
    def encode_xy(self, x_loc, y_loc):
        # x_loc, y_loc each have shape (...)
        # output has shape (..., 2)
        x_norm = (x_loc - self.x_mean) / self.x_std
        y_norm = (y_loc - self.y_mean) / self.y_std
        return torch.stack([x_norm, y_norm], dim=-1)
    
    def encode_magnitude(self, mag, mag_completeness: Union[float, torch.tensor]):
        # Subtracts magnitude of completeness from the magnitudes 
        # This means each has an 'effective' magnitude of completeness of 0,
        # and that the magnitude of every other earthquake has beeen reduced
        # by the same amount.
        # mag has shape (...)
        # mag_completeness
        # output has shape (..., 1)
        if type(mag) is float:
            out = mag.unsqueeze(-1) - mag_completeness
        else:
            out = (mag - mag_completeness.unsqueeze(1)).unsqueeze(-1)
        return out

    def encode_extra_features(self, extra_feat):
        # Place holder for any encoding of extra features that may be needed
        # e.g. normalization, log-transform, aggregation, etc.
        # extra_feat has shape (..., num_extra_features)
        # output has shape (..., num_extra_features)
        return extra_feat

    def get_context(self, batch, return_hidden = False):
        """Get context embedding for each event in the batch of padded sequences.

        Returns:
            context: Context vectors, shape (batch_size, seq_len, context_size)
        """
        # Get time
        feat_list = [self.encode_time(batch.inter_times)] # Returns Log of inter-event times - minus the mean (continuously computed)
        
        # Get magnitude
        if self.input_magnitude:
            feat_list.append(self.encode_magnitude(batch.mag, batch.mag_bounds[:, 0]))
        
        # Get locations
        feat_list.append(self.encode_xy(batch.x_loc, batch.y_loc))  # always included
        
        # Get anything additional
        if self.num_extra_features is not None:
            feat_list.append(self.encode_extra_features(batch.extra_feat))
        features = torch.cat(feat_list, dim=-1)

        # Temporary debug
        if self.debug:
            for i, f in enumerate(feat_list):
                print(f"feat_list[{i}] dtype: {f.dtype}, shape: {f.shape}")
    
        # Rnn output is (output, h_n) or (output, (h_n,c_n))
        rnn_out = self.rnn(features) # Get hidden state, discard last entry for 2nd dim
        rnn_output = rnn_out[0][:,:-1,:]
        hidden = rnn_out[1]
        output = F.pad(rnn_output, (0, 0, 1, 0))  # (B, L, C) # adds zero at the beginning
        if return_hidden == True:
            return self.dropout(output), hidden
        
        return self.dropout(output)  # (B, L, C)

    def get_inter_time_dist(self, context):
        """Get the distribution over the inter-event times given the context."""
        params = self.hypernet_time(context)
        # Very small params may lead to numerical problems, clamp to avoid this
        # params = clamp_preserve_gradients(params, -6.0, np.inf)
        scale, shape, weight_logits = torch.split(
            params,
            [self.num_components_time, self.num_components_time, self.num_components_time],
            dim=-1,
        )
        # How does this work? Not yet sure
        scale = F.softplus(scale.clamp_min(-5.0))
        shape = F.softplus(shape.clamp_min(-5.0)) # Maybe higher min?
        weight_logits = F.log_softmax(weight_logits, dim=-1)
        component_dist = dist.Weibull(scale=scale, shape=shape)
        mixture_dist = Categorical(logits=weight_logits)
        return dist.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            component_distribution=component_dist,
        )

    def get_xy_dist(self, context, inter_times):
        """Get a 2D Gaussian mixture distribution over (x, y) given context and τ.

        Option B: p(x,y | τ, c) — spatial parameters conditioned on inter-event time
        so that aftershock proximity and background diffusion emerge from the same decoder.
        inter_times must broadcast against context's leading dimensions.
        """
        # Option B: concatenate encoded τ to context before decoding spatial params.
        # Revert by removing the cat and passing context directly to hypernet_xy.
        tau_encoded = self.encode_time(inter_times)          # (..., 1)
        context_tau = torch.cat([context, tau_encoded], dim=-1)  # (..., C+1)
        params = self.hypernet_xy(context_tau)  # (..., num_components * 6)
        C = self.num_components_space

        means, l_diag, l_offdiag, weight_logits = torch.split(
            params, [2*C, 2*C, C, C], dim=-1
        )

        # means: (..., C, 2)
        # Splits last dimension (-2) into (C,2), if before it had C*2
        means = means.unflatten(-1, (C, 2))

        # Build lower-triangular Cholesky factor L for each component
        # l_diag entries must be positive → softplus
        l_diag  = F.softplus(l_diag.clamp_min(-5.0))   # (..., 2*C)
        l_diag  = l_diag.unflatten(-1, (C, 2))          # (..., C, 2)
        l_offdiag = l_offdiag.unflatten(-1, (C, 1))     # (..., C, 1)

        # Assemble L: shape (..., C, 2, 2)
        *batch, c, _ = l_diag.shape # batch is all dimensions except last two
        L = torch.zeros(*batch, c, 2, 2,  # spreads batch dimensions back out
                        dtype=context.dtype, device=context.device)
        L[..., 0, 0] = l_diag[..., 0]   # l_11
        L[..., 1, 1] = l_diag[..., 1]   # l_22
        L[..., 1, 0] = l_offdiag[..., 0] # l_21 (lower off-diagonal)

        weight_logits = F.log_softmax(weight_logits, dim=-1)

        component_dist = torch.distributions.MultivariateNormal(
            loc=means,
            scale_tril=L,   # accepts Cholesky directly — no need to form Σ explicitly
        )

        # We need a mixture distribution for the gaussians
        mixture_dist = Categorical(logits=weight_logits)

        # Combine the Gaussians 
        return dist.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            component_distribution=component_dist,
        )

    def get_magnitude_dist(self, context, mag_completeness):
        """Returns the GutenberRichter distribution for each context."""
        log_rate = self.hypernet_mag(context).squeeze(-1)  # (B, L)
        b = self.richter_b * torch.ones_like(log_rate) # (B, L)
        mag_min = mag_completeness.unsqueeze(1) * torch.ones_like(b[0, :])  # FLAG 
        return dist.GutenbergRichter(b=b, mag_min=mag_min)

    def nll_loss(self, batch: eq.data.Batch) -> torch.Tensor:
        """
        Compute negative log-likelihood (NLL) for a batch of event sequences.

        Args:
            batch: Batch of padded event sequences.

        Returns:
            nll: NLL of each sequence, shape (batch_size,)
        """
        context = self.get_context(batch)  # (B, L, C) Encodes everything, runs through rnn
        # Inter-event times
        inter_time_dist = self.get_inter_time_dist(context) # Get the weibull distributions
        log_pdf = inter_time_dist.log_prob(  # Get negative log likelihood
            batch.inter_times.clamp_min(1e-10)
        )  # (B, L) # KDC: is this clamping twice?
        log_like = (log_pdf * batch.mask).sum(-1)


        # Survival time from last event until t_end
        arange = torch.arange(batch.batch_size)
        
        # 2D spatial term — Option B: condition spatial dist on observed τ at each event
        xy = self.encode_xy(batch.x_loc, batch.y_loc)  # (B, L, 2)
        # xy_dist = self.get_xy_dist(context)          # old: p(x,y|c) independent of τ
        xy_dist = self.get_xy_dist(context, batch.inter_times)  # new: p(x,y|τ,c)
        log_pdf_xy = xy_dist.log_prob(xy)              # (B, L)
        spatial_term  = (log_pdf_xy * batch.mask).sum(-1)
        spatial_weight = 0.1
        log_like = log_like + spatial_weight*spatial_term


        # LAST REAL EVENT - survival
        # Go to last real event in each sequence (batch.end_idx). Get the context
        last_surv_context = context[arange, batch.end_idx, :]

        # Get inter-event time
        last_surv_dist = self.get_inter_time_dist(last_surv_context)

        # Get the log survival probability
        last_log_surv = last_surv_dist.log_survival(
            batch.inter_times[arange, batch.end_idx]
        )
        log_like = log_like + last_log_surv.squeeze(-1)  # (B,)

        #  DEBUG
        time_nll = -(log_pdf * batch.mask).sum(-1).mean()
        xy_nll = -(log_pdf_xy * batch.mask).sum(-1).mean()
        lls_nll = last_log_surv
        #print(f"Time NLL: {time_nll:.3f}, XY NLL: {xy_nll:.3f}")#, LLS : {lls_nll:.3f}")

        # FIRST REAL EVENT - survival. Remove anything before.
        # Remove survival time from t_prev to t_nll_start
        if torch.any(batch.t_nll_start != batch.t_start):
            # Get the context for each sequence at the start (start_idx)
            prev_surv_context = context[arange, batch.start_idx, :]
            prev_surv_dist = self.get_inter_time_dist(prev_surv_context)
            prev_surv_time = batch.inter_times[arange, batch.start_idx] - (
                batch.arrival_times[arange, batch.start_idx] - batch.t_nll_start
            )
            prev_log_surv = prev_surv_dist.log_survival(prev_surv_time)
            log_like = log_like - prev_log_surv

        return -log_like / (batch.t_end - batch.t_nll_start)  # (B,)

    def sample(
        self,
        batch_size: int,
        duration: float,
        t_start: float = 0.0,
        past_seq: Optional[eq.data.Sequence] = None,
        return_sequences: bool = False,
        mag_completeness: Optional[float] = None,
    ) -> Union[eq.data.Batch, List[eq.data.Sequence]]:
        """Simulate a batch of event sequences from the model.

        Args:
            batch_size: Number of sequences to generate.
            duration: Length of the interval on which to simulate the TPP.
            t_start: Start of the interval on which to simulate the TPP.
            past_seq: If provided, events are sampled conditioned on the past sequence.
            return_sequences: If True, returns samples as List[eq.data.Sequence].
                If False, returns samples as eq.data.Batch.

        Returns:
            batch: Sequences generated from the model.

        """

        if self.input_magnitude != self.predict_magnitude:
            raise ValueError(
                "Sampling is impossible if input_magnitude != predict_magnitude"
            )
        if self.num_extra_features is not None:
            raise ValueError("Sampling is not currently supported for extra features")

        if past_seq is not None:
            t_start = past_seq.t_end
            past_batch = eq.data.Batch.from_list([past_seq])
            if mag_completeness is not None:
                mag_threshold = torch.as_tensor(
                    mag_completeness, device=self.device, dtype=past_seq.mag_bounds.dtype
                )
            else:
                mag_threshold = past_seq.mag_bounds[0].to(self.device)
            past_context, h_n = self.get_context(past_batch,
                                                 return_hidden=True) # (1, 1, C)
            assert past_context.shape[-1] == h_n.shape[-1]
            current_state = past_context[:, [-1], :] 
            current_state = current_state.expand(batch_size, -1, -1)  # (B, 1, C)
            time_remaining = past_seq.t_end - past_seq.arrival_times[-1]
        else:
            current_state = torch.zeros(batch_size, 1, self.context_size)
            time_remaining = None
            mag_threshold = (
                torch.as_tensor(
                    mag_completeness, device=self.device, dtype=current_state.dtype
                )
                if mag_completeness is not None
                else None
            )

        if self.predict_magnitude and mag_threshold is None:
            raise ValueError("mag_completeness must be provided when sampling magnitudes")
        if mag_threshold is not None and mag_threshold.ndim == 0:
            mag_threshold = mag_threshold.expand(batch_size)

        t_end = t_start + duration

        inter_times = torch.empty(batch_size, 0, device=self.device)
        locations = torch.empty(batch_size, 0, 2, device=self.device)  # (B, 0, 2)
        if self.predict_magnitude:
            magnitudes = torch.empty(batch_size, 0, device=self.device)
        else:
            magnitudes = None

        generated = False
        while not generated:
            inter_time_dist = self.get_inter_time_dist(current_state)
            if time_remaining is None:
                next_inter_times = inter_time_dist.sample()   # (B, 1)
            else:
                next_inter_times = inter_time_dist.sample_conditional(
                    lower_bound=time_remaining
                )
                next_inter_times -= time_remaining
                time_remaining = None
            next_inter_times.clamp_max_(t_end - t_start)
            inter_times = torch.cat([inter_times, next_inter_times], dim=1)

            rnn_input_list = [self.encode_time(next_inter_times)]

            if self.predict_magnitude:
                mag_dist = self.get_magnitude_dist(current_state, mag_threshold)
                next_mag = mag_dist.sample()                  # (B, 1)
                next_mag = next_mag.clamp_min_(mag_threshold.unsqueeze(1))
                magnitudes = torch.cat([magnitudes, next_mag], dim=1)
                rnn_input_list.append(self.encode_magnitude(next_mag, mag_threshold))

            # Sample spatial locations and feed back into RNN
            # xy_dist = self.get_xy_dist(current_state)      # old: p(x,y|c)
            xy_dist = self.get_xy_dist(current_state, next_inter_times)  # new: p(x,y|τ,c)
            next_xy = xy_dist.sample()                        # (B, 1, 2)
            locations = torch.cat([locations, next_xy], dim=1)
            next_x = next_xy[..., 0]                          # (B, 1)
            next_y = next_xy[..., 1]                          # (B, 1)
            rnn_input_list.append(self.encode_xy(next_x, next_y))  # (B, 1, 2)

            # RNN_input now has [inter_time, magnitude, and (x,y)]
            with torch.no_grad():
                reached = inter_times.sum(-1).min()
                generated = reached >= t_end - t_start
                rnn_input = torch.cat(rnn_input_list, dim=-1)

            current_state = self.rnn(
                rnn_input, current_state.transpose(0, 1).contiguous()
            )[0]
            current_state = self.dropout(current_state)       # (B, 1, C)

        duration = t_end - t_start
        unclipped_arrival_times = inter_times.cumsum(-1)
        padding_mask = unclipped_arrival_times >= duration
        inter_times = torch.masked_fill(inter_times, padding_mask, 0.0)
        end_idx = (1 - padding_mask.long()).sum(-1)
        last_surv_time = duration - inter_times.sum(-1)
        inter_times[torch.arange(batch_size), end_idx] = last_surv_time

        """
        print("t_start type:", type(t_start), t_start)
        print("inter_times dtype:", inter_times.dtype)
        print("cumsum:", inter_times.cumsum(-1)[0, :5])
        print("cumsum + t_start:", (inter_times.cumsum(-1) + t_start)[0, :5])
        print("arrival_times[0, :5] inside sample:", inter_times.cumsum(-1) + t_start)  # ← add t_start
        print("t_start inside sample:", t_start)
        """

        batch = eq.data.Batch(
            inter_times=inter_times,
            arrival_times=inter_times.cumsum(-1) + t_start,  # ← add t_start
            t_start=torch.full([batch_size], t_start, device=self.device).float(),
            t_end=torch.full([batch_size], t_end, device=self.device).float(),
            t_nll_start=torch.full([batch_size], t_start, device=self.device).float(),
            mask=padding_mask.float(),
            start_idx=torch.zeros(batch_size, device=self.device).long(),
            end_idx=end_idx,
            mag=magnitudes,
            x_loc=locations[..., 0],
            y_loc=locations[..., 1],
        )
        if return_sequences:
            return batch.to_list()
        else:
            return batch

    def evaluate_intensity(
        self,
        sequence: eq.data.Sequence,
        num_grid_points: int = 100,
        eps: float = 1e-4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = eq.data.Batch.from_list([sequence])
        context = self.get_context(batch).squeeze(0)  # (L, C)
        inter_time_dist = self.get_inter_time_dist(context)

        # Evaluate each hazard function at times x = [eps, ..., tau_i]
        x = batch.inter_times * torch.linspace(eps, 1, num_grid_points)[:, None]
        intensity = inter_time_dist.log_hazard(x).T.reshape(-1).exp()

        # Shift the inter-event times x to get the global times
        offsets = torch.cat([torch.tensor([0.0]), sequence.arrival_times])
        grid = (x + offsets).T.reshape(-1)
        return grid, intensity

    def evaluate_spatial_intensity(
        self,
        sequence: eq.data.Sequence,
        grid_x: torch.Tensor,
        grid_y: torch.Tensor,
        tau: float,
    ) -> torch.Tensor:
        """Evaluate the forecast spatial intensity over a 2D grid at a given τ.

        Computes p(x, y | τ, all observed events) at each grid point under
        Option B, where the spatial distribution is conditioned on inter-event
        time τ = t_query - t_last_event.  Density is returned in geographic
        coordinate units, accounting for the Jacobian of the normalization.

        Args:
            sequence: Observed history.
            grid_x:   2-D tensor of x-coordinates (e.g. longitude), shape (H, W).
            grid_y:   2-D tensor of y-coordinates (e.g. latitude),  shape (H, W).
            tau:      Time since the last observed event (same units as sequence).

        Returns:
            density:  Probability density at each grid point, shape (H, W).
        """
        batch = eq.data.Batch.from_list([sequence])
        context = self.get_context(batch).squeeze(0)  # (L, C)

        # context[-1] is conditioned on all observed events — same pattern as
        # sample(), which uses past_context[:, -1, :] as its forecast starting state.
        forecast_context = context[-1]  # (C,)

        tau_tensor = torch.tensor(tau, dtype=forecast_context.dtype,
                                  device=forecast_context.device)
        # old: xy_dist = self.get_xy_dist(forecast_context)
        xy_dist = self.get_xy_dist(forecast_context, tau_tensor)  # p(x,y|τ,c)

        # Normalise grid coordinates to match the space the mixture was decoded in
        x_norm = (grid_x - self.x_mean) / self.x_std  # (H, W)
        y_norm = (grid_y - self.y_mean) / self.y_std  # (H, W)
        xy_norm = torch.stack([x_norm, y_norm], dim=-1)  # (H, W, 2)

        log_p_norm = xy_dist.log_prob(xy_norm)  # (H, W)

        # Convert density from normalised space to geographic space:
        # p(x,y) = p(z) * |dz/dx| = p(z) / (σ_x * σ_y)
        log_jacobian = -(self.x_std.log() + self.y_std.log())
        return (log_p_norm + log_jacobian).exp()  # (H, W)

    def evaluate_compensator(
        self, sequence: eq.data.Sequence, num_grid_points: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = eq.data.Batch.from_list([sequence])
        context = self.get_context(batch).squeeze(0)  # (L, C)
        inter_time_dist = self.get_inter_time_dist(context)

        # Note that on each interval the compensator can be computed as:
        # compensator = -log_surv(x)
        # or
        # compensator = int(hazard(x)) dx

        # Evaluate each log survival function at times x = [eps, ..., tau_i]
        x = batch.inter_times * torch.linspace(1e-4, 1, num_grid_points)[:, None]
        log_surv = inter_time_dist.log_survival(x)
        # Compute the cumulative sum of log survival functions to get the compensator
        surv_offsets = torch.cat(
            [torch.tensor([0.0]), log_surv[-1].cumsum(dim=-1)[:-1]]
        )
        compensator = -(log_surv + surv_offsets).T.reshape(-1)

        # Shift the inter-event times x to get the global times
        offsets = torch.cat([torch.tensor([0.0]), sequence.arrival_times])
        grid = (x + offsets).T.reshape(-1)
        return grid, compensator
    
    def evaluate_conditional_intensity(
        self,
        sequence: eq.data.Sequence,
        t_grid: torch.Tensor,
        x_grid: torch.Tensor,
        y_grid: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the joint spatio-temporal conditional intensity on a grid.

        Combines evaluate_intensity() and evaluate_spatial_intensity() via the
        standard factorization of the conditional intensity into a temporal
        hazard rate and a spatial density given time:

            λ(t, r | H) = f(t | c) / (1 - F(t | c))  ×  g(r | t, c)
                        = hazard(t | c)  ×  g(r | t, c)

        where c is the context summarizing the observed history H (taken
        after the last observed event, as in evaluate_spatial_intensity),
        t is elapsed time since that last event, and r = (x, y) is location.
        Because the spatial decoder is conditioned on τ (Option B), g(r|t,c)
        is re-evaluated for every t in t_grid.

        Args:
            sequence: Observed history H.
            t_grid: 1-D tensor of times since the last observed event, shape (T,).
            x_grid: 2-D tensor of x-coordinates, shape (R, C).
            y_grid: 2-D tensor of y-coordinates, shape (R, C).

        Returns:
            intensity: λ(t, r | H) evaluated at each (t, x, y), shape (T, R, C).
        """
        batch = eq.data.Batch.from_list([sequence])
        context = self.get_context(batch).squeeze(0)  # (L, C)
        forecast_context = context[-1]  # (C,) — condition on the full observed history

        t_grid = t_grid.to(dtype=forecast_context.dtype, device=forecast_context.device)

        # Temporal hazard rate: f(t|c) / (1 - F(t|c))
        inter_time_dist = self.get_inter_time_dist(forecast_context)
        hazard = inter_time_dist.log_hazard(t_grid).exp()  # (T,)

        # Normalize the spatial grid once, matching the space the mixture decodes in
        x_norm = (x_grid - self.x_mean) / self.x_std  # (R, C)
        y_norm = (y_grid - self.y_mean) / self.y_std
        xy_norm = torch.stack([x_norm, y_norm], dim=-1)  # (R, C, 2)
        log_jacobian = -(self.x_std.log() + self.y_std.log())

        # g(r|t,c) depends on t (Option B), so re-decode the spatial mixture per t
        intensity = torch.empty(
            t_grid.shape[0], *x_grid.shape,
            dtype=forecast_context.dtype, device=forecast_context.device,
        )
        for i, t in enumerate(t_grid):
            xy_dist = self.get_xy_dist(forecast_context, t)   # g(r|t,c)
            log_g = xy_dist.log_prob(xy_norm) + log_jacobian  # (R, C)
            intensity[i] = hazard[i] * log_g.exp()

        return intensity
