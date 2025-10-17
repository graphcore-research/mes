from typing import Callable
import torch as pt


def optimize_adam(
    *,
    theta: pt.Tensor,
    loss_fn: Callable,
    max_iters: int = 200,
    tol: float = 1e-9,
    lr: float = 1e-2,
    wd: float = 0.0,
) -> tuple[pt.Tensor, float]:
    """
    Optimize the given parameters using Adam.

    Args:
        theta: pt.Tensor, shape (n_x, 4)
        loss_fn: Callable, the loss function to optimize
        max_iters: int, the maximum number of iterations
        tol: float, the tolerance for the optimization
        lr: float, the learning rate for the optimization

    Returns:
        theta: pt.Tensor, shape (n_x, 4)
        final_loss: float, the optimized loss
    """
    opt = pt.optim.Adam([theta], lr=lr, amsgrad=True, weight_decay=wd)
    prev_loss = float("inf")
    L = loss_fn(theta)

    loss_min = L.item()
    theta_min = theta.detach().clone()

    for _ in range(max_iters):
        opt.zero_grad(set_to_none=True)
        L = loss_fn(theta)
        L.backward()
        opt.step()

        # Early stopping
        if abs(prev_loss - L.item()) < tol:
            break
        prev_loss = L.item()

        if L.item() < loss_min:
            loss_min = L.item()
            theta_min = theta.detach().clone()

    return theta_min, loss_min
