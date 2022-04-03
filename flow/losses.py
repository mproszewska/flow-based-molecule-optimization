import torch
from torch.distributions import Normal


def naive_loss(z, logdet, mu, sigma, mu_rest=None):
    """First element of z corresponds to property (mu = property value),
    the rest have N(0,1) distribution"""
    device = z.device
    z_a = z[:, : mu.shape[1]]
    z_rest = z[:, mu.shape[1] :]
    dist_a = Normal(mu, sigma * torch.ones_like(mu, device=device))
    dist_rest = Normal(mu_rest if mu_rest is not None else torch.tensor([0.0]).to(device) , torch.tensor([1.0]).to(device))
    logp_a = dist_a.log_prob(z_a)
    logp_rest = dist_rest.log_prob(z_rest)
    logpz = logp_a.sum(-1, keepdim=True) + logp_rest.sum(-1, keepdim=True)
    logpz = (logpz + logdet).mean()
    return (
        -logpz,
        {
            "-logpz": -logpz,
            "-logp_a": -logp_a.mean(),
            "MSE": torch.mean((z_a - mu) ** 2),
            "sigma": torch.tensor(sigma),
        },
    )


def N01_loss(z, logdet, mu_rest=None):
    device = z.device
    dist = Normal(mu_rest if mu_rest is not None else torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
    logpz = dist.log_prob(z)
    logpz = logpz.sum(-1, keepdim=True)
    logpz = (logpz + logdet).mean()
    return (
        -logpz,
        {"-logpz": -logpz},
    )
