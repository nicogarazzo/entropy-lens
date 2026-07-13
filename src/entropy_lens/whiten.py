"""Activation-aware whitening (SVD-LLM style) for the data-metric law.

Rationale (mesa conjunta 2026-07-12): LLM weight matrices have flat SVD
spectra in the Frobenius metric (high S1), so naive truncation destroys the
model at any parameter-saving budget. The output-relevant error is not
||dW||_F but ||dW @ L||_F, where L L^T = C = sum(x x^T) over calibration
activations. Truncating the whitened matrix M = W @ L is Eckart-Young
optimal in the data metric, and the Entropy-Compression Law is refit on
S1_eff = S1(sigma(M)).

Memory design: covariances are accumulated per *input group*, not per
projection. Within a decoder layer, q/k/v share the attention input,
gate/up share the MLP input, so only 4 covariances per layer are needed
(attn_in, o_in, mlp_in, down_in). down_proj's covariance is the big one
(intermediate_size^2, ~822 MB fp32 for Mistral 7B), so calibration is run
in layer chunks to bound GPU memory.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Which module inside the decoder block sees each covariance group, and
# which canonical projections consume it.
_HOOK_GROUPS = {
    "attn_in": "self_attn.q_proj",
    "o_in": "self_attn.o_proj",
    "mlp_in": "mlp.gate_proj",
    "down_in": "mlp.down_proj",
}

PROJ_TO_GROUP = {
    "q_proj": "attn_in",
    "k_proj": "attn_in",
    "v_proj": "attn_in",
    "o_proj": "o_in",
    "gate_proj": "mlp_in",
    "up_proj": "mlp_in",
    "down_proj": "down_in",
}


def _get_submodule(block: torch.nn.Module, path: str) -> torch.nn.Module:
    mod = block
    for attr in path.split("."):
        mod = getattr(mod, attr)
    return mod


def collect_covariances(
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    layer_indices: list[int],
    device: str = "cuda",
) -> dict[tuple[int, str], torch.Tensor]:
    """Accumulate C = sum(x^T x) of the inputs to each projection group.

    Args:
        model: a LLaMA-family CausalLM already on `device`.
        batches: list of input_ids tensors, each (1, seq_len).
        layer_indices: which decoder layers to instrument this pass.
        device: where the forward runs; covariances accumulate there in fp32.

    Returns:
        {(layer_idx, group): C} with C moved to CPU fp32, shape (in, in).
    """
    accs: dict[tuple[int, str], torch.Tensor] = {}
    handles = []

    def make_hook(key):
        def hook(module, args):
            x = args[0]
            x = x.reshape(-1, x.shape[-1]).to(torch.float32)
            c = x.T @ x
            if key in accs:
                accs[key] += c
            else:
                accs[key] = c
        return hook

    for i in layer_indices:
        block = model.model.layers[i]
        for group, path in _HOOK_GROUPS.items():
            mod = _get_submodule(block, path)
            handles.append(mod.register_forward_pre_hook(make_hook((i, group))))

    try:
        with torch.no_grad():
            for b in batches:
                model(b.to(device))
    finally:
        for h in handles:
            h.remove()

    return {k: v.cpu() for k, v in accs.items()}


def cholesky_factor(
    cov: torch.Tensor,
    damp: float = 1e-6,
    max_damp: float = 1e-1,
    channel_prescale: bool = True,
) -> tuple[torch.Tensor, bool, float]:
    """Factor C ~= L L^T with escalating ridge damping and eigh fallback.

    Activation covariances are routinely rank-deficient, so a plain Cholesky
    will fail on some layers. We add lambda*I with lambda = damp * mean(diag)
    and escalate x10 until it succeeds; if it never does, fall back to an
    eigendecomposition with clamped eigenvalues (L is then NOT triangular).

    Why channel_prescale (found empirically, 2026-07-12 on Mistral 7B): raw
    activation covariances have wildly heterogeneous per-channel variance
    (the "massive activations" phenomenon in late transformer layers: a
    handful of channels run 10-100x hotter than the rest). A ridge lambda*I
    floors ALL channels equally, so it either does nothing to the hot
    channels or over-regularizes the cold ones once escalated enough to fix
    the hot ones -- this is why mlp_in damping needed on Mistral 7B rose
    ~180x from layer 0 to layer 28 (0.0088 -> 1.58) and correlated at -0.88
    with S1_eff: heavy uniform damping was collapsing the whitened spectrum
    as an artifact, not a real compressibility signal. The fix (SmoothQuant/
    AWQ-style): rescale each channel by its own std s_j = sqrt(diag(C)_j)
    before damping, so lambda*I becomes a per-channel-relative ridge
    lambda*diag(s^2) in the original units. Cold channels no longer need to
    borrow the hot channels' damping budget.

    Args:
        cov: (n, n) symmetric PSD matrix.
        damp: initial ridge as a fraction of mean(diag(normalized cov)).
        max_damp: largest ridge fraction to try before the eigh fallback.
        channel_prescale: rescale by per-channel std before damping (see above).

    Returns:
        (L, triangular, lam) with C + lam*diag(s^2) = L @ L.T (s = ones if
        channel_prescale=False, so C + lam*I as before). `triangular` tells
        downstream solvers whether L is lower-triangular.
    """
    c = cov.to(torch.float64)
    c = 0.5 * (c + c.T)
    n = c.shape[0]

    if channel_prescale:
        s = c.diagonal().clamp_min(1e-30).sqrt()
        c_n = c / torch.outer(s, s)
    else:
        s = torch.ones(n, dtype=torch.float64)
        c_n = c

    mean_diag = float(c_n.diagonal().mean().clamp_min(1e-30))
    eye = torch.eye(n, dtype=torch.float64)

    lam = damp * mean_diag
    while lam <= max_damp * mean_diag:
        try:
            L_n = torch.linalg.cholesky(c_n + lam * eye)
            return s.unsqueeze(1) * L_n, True, lam
        except torch.linalg.LinAlgError:
            lam *= 10.0

    logger.warning("Cholesky failed up to damp=%g; using eigh fallback", max_damp)
    evals, evecs = torch.linalg.eigh(c_n)
    evals = evals.clamp_min(damp * mean_diag)
    L_n = evecs @ torch.diag(torch.sqrt(evals))
    return s.unsqueeze(1) * L_n, False, damp * mean_diag


def whitened_svdvals(
    weight: torch.Tensor,
    L: torch.Tensor,
    device: str | None = None,
) -> np.ndarray:
    """Singular values of the whitened matrix M = W @ L.

    W is (out, in) as stored by nn.Linear; L is (in, in) from the covariance
    of that projection's input, so the product is well-posed.

    Returns:
        1D numpy array, descending, noise-filtered like extract._compute_svdvals.
    """
    w = weight.to(torch.float32)
    l = L.to(torch.float32)
    if device is not None:
        w, l = w.to(device), l.to(device)
    m = w @ l
    sv = torch.linalg.svdvals(m).cpu().numpy()
    if len(sv) > 0 and sv[0] > 0:
        sv = sv[sv > sv[0] * 1e-12]
    return sv


def whiten_truncate(
    weight: torch.Tensor,
    L: torch.Tensor,
    rank: int,
    triangular: bool = True,
) -> torch.Tensor:
    """Rank-`rank` truncation of W that is optimal in the data metric.

    Computes M = W @ L, truncates M via SVD (Eckart-Young in ||.||_F, which
    equals the data metric for W), and maps back: W_d = M_d @ L^{-1}.

    Args:
        weight: (out, in) tensor.
        L: (in, in) whitening factor from cholesky_factor.
        rank: singular values to keep, clamped to [1, min(out, in)].
        triangular: whether L is lower-triangular (enables the fast solve).

    Returns:
        Dense (out, in) reconstruction in the original dtype.
    """
    dtype = weight.dtype
    w = weight.to(torch.float32)
    l = L.to(torch.float32).to(w.device)
    rank = max(1, min(rank, min(w.shape)))

    m = w @ l
    u, s, vh = torch.linalg.svd(m, full_matrices=False)
    m_d = (u[:, :rank] * s[:rank]) @ vh[:rank]

    if triangular:
        w_d = torch.linalg.solve_triangular(l, m_d, upper=False, left=False)
    else:
        w_d = m_d @ torch.linalg.inv(l)
    return w_d.to(dtype)
