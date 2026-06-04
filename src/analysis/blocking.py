"""
src.analysis.blocking
------------------------------------------------------------------------
Reblocking analysis for estimating standard errors of correlated MD observables.

Based on the method of Flyvbjerg and Petersen (J. Chem. Phys. 91, 461, 1989).
Automatically detects the plateau block size at which the error estimate converges.
"""


import numpy as np
from scipy.optimize import minimize
from scipy.stats import gaussian_kde


#······································································#
# Low-level blocking primitives
#······································································#

def _valid_block_sizes(array: np.ndarray, multi: int = 1) -> tuple[int, np.ndarray, list[float]]:
    """Find all block sizes that evenly divide the data length.

    Parameters
    ----------
    array : np.ndarray
        1-D data array.
    multi : int, optional
        Number of replicas for multi-replica blocking.

    Returns
    -------
    tuple[int, np.ndarray, list[float]]
        ``(N, n_blocks, block_sizes)``
    """
    N = len(array)
    rep = N / multi
    candidates = np.arange(2 if multi == 1 else multi, N + 1)
    n_blocks, block_sizes = [], []
    for n in candidates:
        bs = N / n
        if (N % n == 0) and (rep % bs == 0):
            n_blocks.append(int(n))
            block_sizes.append(bs)
    return N, np.array(n_blocks), block_sizes

#······································································#

def _trim_for_blocking(array: np.ndarray, multi: int = 1) -> np.ndarray:
    """Trim the array tail until at least 20 valid block transformations exist.

    Reblocking requires many divisible block sizes; if the raw length is
    inconvenient a small number of frames are removed from each replica end.

    Parameters
    ----------
    array : np.ndarray
        1-D data array.
    multi : int, optional
        Number of replicas.

    Returns
    -------
    np.ndarray
        Possibly shortened array.
    """
    if len(_valid_block_sizes(array, multi)[1]) > 19:
        return array

    rep_len = len(array) // multi
    for c in range(1, 102):
        trimmed = np.concatenate([
            array[n * rep_len - rep_len: n * rep_len - c]
            for n in range(1, multi + 1)
        ])
        if len(_valid_block_sizes(trimmed, multi)[1]) > 19:
            return trimmed

    return array

#······································································#

def blocking(array: np.ndarray, multi: int = 1) -> np.ndarray:
    """Standard reblocking: error estimate as a function of block size.

    Parameters
    ----------
    array : np.ndarray
        1-D time series (already trimmed to a convenient length).
    multi : int, optional
        Number of replicas.

    Returns
    -------
    np.ndarray
        Shape ``(B, 3)`` array with columns
        ``[block_size, error_estimate, error_on_error]``, sorted from
        largest to smallest block size.
    """
    u = array.mean()
    N, n_blocks, block_sizes = _valid_block_sizes(array, multi)
    rows = []
    for nb, bs in zip(n_blocks, block_sizes):
        block_avs = np.array([
            array[int(bs * n - bs): int(bs * n)].mean() for n in range(1, nb + 1)
        ])
        err = np.sqrt(((block_avs - u) ** 2).sum() / (nb * (nb - 1)))
        rows.append([bs, err, err / np.sqrt(2 * (nb - 1))])
    return np.flip(np.array(rows), axis=0)

#······································································#

def fblocking(
    cv: np.ndarray,
    weights: np.ndarray,
    kbt: float,
    multi: int = 1,
    interval: tuple[float, float] | None = None,
) -> np.ndarray:
    """Reblocking for weighted (biased) data (e.g. WTMetaD trajectories).

    Parameters
    ----------
    cv : np.ndarray
        Collective variable time series.
    weights : np.ndarray
        Reweighting factors (not normalised).
    kbt : float
        Thermal energy k_B T in kJ/mol.
    multi : int, optional
        Number of replicas.
    interval : tuple[float, float], optional
        CV range for histogram; defaults to ``(cv.min(), cv.max())``.

    Returns
    -------
    np.ndarray
        Shape ``(B, 3)`` — ``[block_size, mean_fes_error, error_on_error]``.
    """
    if interval is None:
        interval = (cv.min(), cv.max())
    N, n_blocks, block_sizes = _valid_block_sizes(cv, multi)
    u, bins = np.histogram(cv, weights=weights, bins=50, range=interval)
    u = u / N
    rows = []
    for nb, bs in zip(n_blocks, block_sizes):
        his = np.zeros(len(bins) - 1)
        for n in range(nb):
            s, e = int(n * bs), int(n * bs + bs)
            hi = np.histogram(cv[s:e], weights=weights[s:e], bins=bins)[0] / len(cv[s:e])
            his += (hi - u) ** 2
        err_vec = np.sqrt(his / (nb * (nb - 1)))
        err_vec = kbt * err_vec / u
        rows.append([bs, err_vec.mean(), err_vec.mean() / np.sqrt(2 * (nb - 1))])
    return np.flip(np.array(rows), axis=0)

#······································································#

def autocorrelation(x: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation function of a 1-D time series.

    Parameters
    ----------
    x : np.ndarray
        1-D time series.

    Returns
    -------
    np.ndarray
        Autocorrelation values for lags 0 … N-1.
    """
    n = len(x)
    variance = x.var()
    x = x - x.mean()
    r = np.correlate(x, x, mode="full")[-n:]
    return r / (variance * np.arange(n, 0, -1))



#······································································#
# BlockAnalysis class
#······································································#

class BlockAnalysis:
    """Reblocking error analysis for correlated MD time series.

    Automatically selects the plateau block size and reports the
    standard error of the mean (SEM) with uncertainty.

    Parameters
    ----------
    x : np.ndarray
        1-D observable time series.
    multi : int, optional
        Number of concatenated replicas in ``x``.
    weights : np.ndarray, optional
        Per-frame weights (for reweighted/biased ensembles).
    bias : np.ndarray, optional
        Bias potential values; weights are derived as ``exp(-bias/kBT)``.
        Requires ``T`` to be set.
    T : float, optional
        Temperature in Kelvin (required when ``bias`` is provided).
    interval_low, interval_up : float, optional
        CV range clipping for ``get_pdf`` / ``get_fes``.
    dt : float, optional
        Time step scaling applied to block sizes (default 1).

    Attributes
    ----------
    av : float
        Mean of the observable (unweighted case only).
    sem : float
        Standard error of the mean at the plateau block size.
    bs : float
        Detected plateau block size.
    """

    def __init__(
        self,
        x: np.ndarray,
        multi: int = 1,
        weights: np.ndarray | None = None,
        bias: np.ndarray | None = None,
        T: float | None = None,
        interval_low: float | None = None,
        interval_up: float | None = None,
        dt: float = 1.0,
    ) -> None:
        self.multi = multi
        self.x = _trim_for_blocking(x, multi)
        self.w = weights

        self.interval = [self.x.min(), self.x.max()]
        if interval_low is not None and self.x.min() < interval_low:
            self.interval[0] = interval_low
        if interval_up is not None and self.x.max() > interval_up:
            self.interval[1] = interval_up

        if self.w is None and bias is not None:
            assert T is not None, "Temperature T required when bias is provided"
            bias = bias - bias.max()
            self.kbT = 0.008314463 * T
            self.w = np.exp(bias / self.kbT)
            self.w = _trim_for_blocking(self.w, multi)

        if self.w is None:
            self.stat = blocking(self.x, multi)
            self.av: float = float(self.x.mean())
        else:
            self.w = self.w / self.w.sum()
            self.stat = fblocking(self.x, self.w, self.kbT, multi, tuple(self.interval))  # type: ignore[arg-type]

        self.stat[..., 0] /= dt

    def SEM(self) -> None:
        """Find the standard error of the mean at the blocking plateau.

        Sets ``self.sem`` and ``self.bs``.  Prints a warning if convergence
        may not have been reached.
        """
        def _n_overlap(x_try: float, stat: np.ndarray) -> float:
            # Negative count of stat points whose error band contains x_try
            return -sum(
                1 for p in stat if p[1] - p[2] <= x_try <= p[1] + p[2]
            )

        c = np.zeros(len(self.stat))
        for i, b in enumerate(self.stat):
            larger_blocks = self.stat[self.stat[:, 0] > b[0]]
            bounds = [(b[1] - b[2], b[1] + b[2])]
            c[i] = -minimize(
                fun=_n_overlap,
                x0=b[1],
                args=(larger_blocks,),
                bounds=bounds,
            ).fun

        best = int(np.argmax(c))
        self.bs: float = float(self.stat[best, 0])
        self.sem: float = float(self.stat[best, 1])

        if self.bs > self.stat[-1, 0] / 3:
            print("WARNING: fixed point of the error may not have been reached!")

    def get_pdf(
        self, cv: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate the probability density function with block error bars.

        Parameters
        ----------
        cv : np.ndarray, optional
            Alternative CV for the histogram; uses ``self.x`` if not given.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(x_grid, pdf, pdf_error)``
        """
        lo, hi = self.interval
        x_grid = np.linspace(lo, hi, 100)
        data = _trim_for_blocking(cv, self.multi) if cv is not None else self.x
        u = gaussian_kde(data, bw_method="silverman", weights=self.w).evaluate(x_grid)

        N = len(self.x)
        Nb = int(N / self.bs)
        W = self.w.sum()
        S = (self.w ** 2).sum()
        blocks_pi = []
        for n in range(1, Nb + 1):
            s, e = int(self.bs * (n - 1)), int(self.bs * n)
            pdf_i = gaussian_kde(
                self.x[s:e], bw_method="silverman", weights=self.w[s:e]
            ).evaluate(x_grid)
            wi = self.w[s:e].sum()
            blocks_pi.append(wi * (pdf_i - u) ** 2)

        err = np.sqrt(np.array(blocks_pi).sum(axis=0) / (Nb * (W - S / W)))
        return x_grid, u, err

    def get_fes(
        self, max_kj: float = 25.0, cv: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the free-energy surface from the PDF.

        Parameters
        ----------
        max_kj : float, optional
            Truncate FES above this value in kJ/mol.
        cv : np.ndarray, optional
            Alternative CV array.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(x_grid, F, F_error)`` — truncated to ``max_kj``.
        """
        x, H, E = self.get_pdf(cv)
        F = -self.kbT * np.log(H)
        FE = self.kbT * E / H
        F -= F.min()
        mask = F < max_kj
        return x[mask], F[mask], FE[mask]

    def get_av_err(
        self, cv: np.ndarray | None = None
    ) -> tuple[float, float]:
        """Compute the weighted average and its propagated error.

        Parameters
        ----------
        cv : np.ndarray, optional
            Alternative CV.

        Returns
        -------
        tuple[float, float]
            ``(average, error)``
        """
        x, H, E = self.get_pdf(cv)
        H = H / H.sum()
        av = float(np.average(x, weights=H))
        err = float(np.sqrt(((H ** 2) * (E ** 2)).sum()))
        return av, err
