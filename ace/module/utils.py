"""Utilities for ACE modules."""
import torch
from scipy.special import gamma


def gram_matrix(x: torch.Tensor, y: torch.Tensor, gammas) -> torch.Tensor:
    """
    Calculate the maximum mean discrepancy gram matrix with multiple gamma values.

    Args:
    ----
        x: Tensor with shape (B, P, M) or (P, M).
        y: Tensor with shape (B, R, M) or (R, M).
        gammas: 1-D tensor with the gamma values.

    Returns
    -------
        A tensor with shape (B, P, R) or (P, R) for the distance between pairs of data
        points in `x` and `y`.
    """
    # if not torch.is_tensor(gammas):
    #     gammas = torch.ones(x.shape[1], dtype=torch.float32) * gammas
    #     gammas = gammas.to(x.device)

    # gammas = gammas.unsqueeze(1)
    pairwise_distances = torch.cdist(x, y, p=2.0)

    pairwise_distances_sq = torch.square(pairwise_distances)
    # tmp = torch.matmul(gammas, torch.reshape(pairwise_distances_sq, (1, -1)))
    tmp = gammas * torch.reshape(pairwise_distances_sq, (1, -1))
    tmp = torch.reshape(torch.sum(torch.exp(-tmp), 0), pairwise_distances_sq.shape) ### shape of tmp: (1, batch_size*batch_size)
    return tmp

def hsic(x, y): #, gamma=1.):
    m = x.shape[0]
    d_x = x.shape[1]
    g_x = 2 * gamma(0.5 * (d_x+1)) / gamma(0.5 * d_x)
    K = gram_matrix(x, x, gammas=1./(2. * g_x))

    d_y = y.shape[1]
    g_y = 2 * gamma(0.5 * (d_y+1)) / gamma(0.5 * d_y)
    L = gram_matrix(y, y, gammas=1./(2. * g_y))
    
    H = torch.eye(m) - 1.0 / m * torch.ones((m, m))
    H = H.float().cuda()
    HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
    return HSIC