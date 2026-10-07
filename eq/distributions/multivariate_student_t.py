import torch
from torch.distributions import Chi2

from .distribution import Distribution


class MultivariateStudentT(Distribution):
    """Multivariate Student-t distribution, parameterized like torch's
    MultivariateNormal (loc + lower-Cholesky scale_tril) plus a degrees-of-
    freedom parameter. Heavier tails than a Gaussian for the same scale;
    converges to MultivariateNormal(loc, scale_tril) as df -> infinity.

    Drop-in replacement for torch.distributions.MultivariateNormal as the
    component distribution of a MixtureSameFamily: same (loc, scale_tril)
    shape conventions, plus log_prob/sample.
    """

    arg_constraints = {}
    has_rsample = True

    def __init__(self, loc, scale_tril, df, validate_args=None):
        # loc: (..., D)   scale_tril: (..., D, D)   df: scalar or (...)
        self.loc = loc
        self.scale_tril = scale_tril
        self.df = torch.as_tensor(df, dtype=loc.dtype, device=loc.device).expand(
            loc.shape[:-1]
        )
        batch_shape = loc.shape[:-1]
        event_shape = loc.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def dim(self):
        return self.event_shape[0]

    def _mahalanobis_sq(self, value):
        diff = (value - self.loc).unsqueeze(-1)  # (..., D, 1)
        z = torch.linalg.solve_triangular(self.scale_tril, diff, upper=False)
        return z.pow(2).sum(dim=(-2, -1))  # (...,)

    def log_prob(self, value):
        d = self.dim
        df = self.df
        maha2 = self._mahalanobis_sq(value)
        log_det = self.scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        log_norm = (
            torch.lgamma((df + d) / 2)
            - torch.lgamma(df / 2)
            - 0.5 * d * torch.log(df * torch.pi)
            - log_det
        )
        return log_norm - 0.5 * (df + d) * torch.log1p(maha2 / df)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)  # sample_shape + batch_shape + event_shape
        z = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        Lz = (self.scale_tril @ z.unsqueeze(-1)).squeeze(-1)
        df = self.df.expand(shape[:-1])
        chi2 = Chi2(df).rsample()
        scale = torch.sqrt(df / chi2)
        return self.loc + Lz * scale.unsqueeze(-1)

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)
