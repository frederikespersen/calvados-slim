"""
src.model
------------------------------------------------------------------------
Classes for modelling and evaluating dG-eps_go relationsships.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.gaussian_process.kernels import Hyperparameter, Kernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


#······································································#
# Curves
#······································································#

class Curve:
    """A dG-eps_go relationship, interpolated in both directions."""

    def __init__(self, dGs, eps_gos, interp1d_kws: dict=dict(kind='quadratic', bounds_error=False)) -> None:
        self.dG_grid_ = dGs
        self.eps_go_grid_ = eps_gos
        self.dG = interp1d(eps_gos, dGs, **interp1d_kws)
        self.eps_go = interp1d(dGs, eps_gos, **interp1d_kws)


#······································································#
# Functional PCA
#······································································#


class FPCATargetEncoder(BaseEstimator, TransformerMixin):
    """Evaluates callable target functions y(x) on a grid and projects them onto fPCA score space.

    Parameters
    ----------
    grid : array-like of shape (n_grid_points,)
        The grid points on which input functions are evaluated.
    n_components : int, default=2
        Number of principal functional components to keep.
    interp1d_kws : dict
        Passed to :class:`scipy.interpolate.interp1d` when decoding scores back to curves.
    """

    def __init__(self, grid, n_components=2, interp1d_kws: dict=dict(kind='quadratic', bounds_error=False)) -> None:
        self.grid = grid
        self.n_components = n_components
        self.interp1d_kws = interp1d_kws

    def fit(self, Y, y=None):
        """Evaluate curves, compute mean curve, and extract fPCA components via SVD.

        Parameters
        ----------
        Y : list or array-like of callables, shape (n_samples,)
            Target callable functions y(x).
        y : None
            Ignored.
        """

        self.grid_ = np.asarray(self.grid, dtype=float)
        if self.grid_.ndim != 1:
            raise ValueError('The grid must be a 1D array-like structure.')

        Y_grid = np.array([y_fn(self.grid_) for y_fn in Y], dtype=float)

        self.mean_curve_ = Y_grid.mean(axis=0)

        U, S, Vt = np.linalg.svd(Y_grid - self.mean_curve_, full_matrices=False)
        k = min(self.n_components, Vt.shape[0])

        self.components_ = Vt[:k]
        self.explained_variance_ratio_ = (S**2 / np.sum(S**2))[:k]

        return self

    def transform(self, Y):
        """Project input functions into fPCA score space.

        Parameters
        ----------
        Y : list or array-like of callables or 2D array
            Callable functions y(x) or pre-evaluated grid arrays.

        Returns
        -------
        scores : ndarray of shape (n_samples, n_components)
            The fPCA score representations of the curves.
        """
        check_is_fitted(self, attributes=['mean_curve_', 'components_'])

        if len(Y) > 0 and callable(Y[0]):
            Y_grid = np.array([y_fn(self.grid_) for y_fn in Y], dtype=float)
        else:
            Y_grid = np.asarray(Y, dtype=float)

        Y_centred = Y_grid - self.mean_curve_
        return Y_centred @ self.components_.T

    def inverse_transform(self, scores):
        """Reconstruct callable functions from predicted fPCA scores.

        Parameters
        ----------
        scores : ndarray of shape (n_samples, n_components)
            fPCA scores output by the regressor.

        Returns
        -------
        Y_callables : ndarray of shape (n_samples,), dtype=object
            Array of callable interpolator functions.
        """
        check_is_fitted(self, attributes=['mean_curve_', 'components_'])
        scores = np.atleast_2d(scores)

        Y_reconstructed = self.mean_curve_ + (scores @ self.components_)

        callables = [interp1d(self.grid_, row, **self.interp1d_kws)
                     for row in Y_reconstructed]

        return np.array(callables, dtype=object)


#······································································#
# Physical curve forms
#······································································#

class ParametricTargetEncoder(BaseEstimator, TransformerMixin):
    """Encodes ``eps_go(dG)`` curves as the parameters of a physical binding model.

    Shares the fit/transform/inverse_transform contract of :class:`FPCATargetEncoder`, so
    either can encode the target of a curve regressor. Decoded curves are monotone and
    concave by construction, as thermodynamic integration requires of ``dG(eps_go)`` when
    the Go term enters the Hamiltonian linearly as ``H = H_0 - eps_go n``::

        d dG / d eps     = -<n>           (mean number of formed native contacts)
        d^2 dG / d eps^2 = -Var(n) / kT   (concavity, by fluctuation-response)

    An fPCA reconstruction obeys neither identity, and can decode a curve that rises or
    bends the wrong way outside the training envelope. The cost is representation error
    wherever the real curve does not match the closure.

    Two closures for the growth of ``<n>`` with the coupling are available.

    ``'saturating'``: native contacts form independently, each with a logistic probability
    in ``eps_go``, so ``<n> = n_eff sigma((eps - eps_0) / w)``, whose integral is a
    softplus::

        dG(eps) = dG_bg - n_eff w [softplus((eps - eps_0)/w) - softplus(-eps_0/w)]

    ``'two_mode'``: the partition function is summed over two competing binding modes, one
    with few contacts but a favourable background energy, one with more contacts. Which
    dominates switches as the coupling grows::

        dG(eps) = -kT ln[ exp((n_1 eps + E_1)/kT) + exp((n_2 eps + E_2)/kT) ]

    Both nest the straight line, at a single dominant mode or a coupling far below onset.

    Parameters
    ----------
    grid : array-like of shape (n_grid_points,)
        The dG grid input curves are evaluated on.
    form : {'saturating', 'two_mode'}, default='saturating'
        Which closure to fit.
    kT : float, default=2.479
        Thermal energy in kJ/mol, used by ``'two_mode'`` only. The default is 298 K.
    n_eps : int, default=400
        Points on the eps_go sweep used to invert the fitted form for decoding.
    interp1d_kws : dict
        Passed to :class:`Curve`, matching :class:`FPCATargetEncoder`.

    Attributes
    ----------
    eps_range_ : pair of float
        Span of eps_go seen during ``fit``, padded; the range decoding sweeps over.
    """

    PARAMETERS = {'saturating': ('dG_bg', 'n_eff', 'eps_0', 'w'),
                  'two_mode': ('n_1', 'E_1', 'delta_n', 'E_2')}

    def __init__(self, grid, form='saturating', kT=2.479, n_eps=400,
                 interp1d_kws: dict = dict(kind='quadratic', bounds_error=False)) -> None:
        self.grid = grid
        self.form = form
        self.kT = kT
        self.n_eps = n_eps
        self.interp1d_kws = interp1d_kws

    #··································································#

    def _forward(self, parameters, eps):
        """dG at the given eps_go, under the chosen closure."""
        if self.form == 'saturating':
            dG_bg, n_eff, eps_0, w = parameters
            return dG_bg - n_eff * w * (np.logaddexp(0.0, (eps - eps_0) / w)
                                        - np.logaddexp(0.0, -eps_0 / w))
        if self.form == 'two_mode':
            n_1, E_1, delta_n, E_2 = parameters
            # delta_n >= 0 is imposed by the bounds, so mode 2 is always the contact-rich one
            return -self.kT * np.logaddexp((n_1 * eps + E_1) / self.kT,
                                           ((n_1 + delta_n) * eps + E_2) / self.kT)
        raise ValueError(f"form must be 'saturating' or 'two_mode', not {self.form!r}")

    def _bounds(self):
        """(lower, upper) per parameter; positivity keeps the curve monotone."""
        if self.form == 'saturating':
            return ([-np.inf, 1e-6, -np.inf, 1e-3], [np.inf, np.inf, np.inf, np.inf])
        return ([0.0, -np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf, np.inf])

    def _initial(self, eps, dG):
        """A straight line through the data, expressed in the closure's parameters."""
        slope, intercept = np.polyfit(eps, dG, 1)
        contacts = max(-slope, 1e-3)
        if self.form == 'saturating':
            return np.array([intercept, contacts, 0.0, 1.0])
        # at eps = 0 two equal modes give dG = -E - kT ln 2
        energy = -intercept - self.kT * np.log(2.0)
        return np.array([0.5 * contacts, energy, 0.5 * contacts, energy])

    def _fit_one(self, eps, dG):
        """Least-squares fit of one curve, in dG space; falls back to the linear start."""
        start = self._initial(eps, dG)
        try:
            result = least_squares(lambda p: self._forward(p, eps) - dG, start,
                                   bounds=self._bounds(), max_nfev=2000)
            return result.x
        except (ValueError, np.linalg.LinAlgError):
            return start

    #··································································#

    def fit(self, Y, y=None):
        """Record the grid and the eps_go span the decoded sweep has to cover.

        Parameters
        ----------
        Y : list of callables or 2D array
            Training curves, ``eps_go`` as a function of dG.
        y : None
            Ignored.
        """
        self.grid_ = np.asarray(self.grid, dtype=float)
        Y_grid = _curves_on_grid(Y, self.grid_)
        low, high = np.nanmin(Y_grid), np.nanmax(Y_grid)
        pad = 0.5 * (high - low) if high > low else 1.0
        self.eps_range_ = (low - pad, high + pad)
        return self

    def transform(self, Y):
        """Fit the closure to each curve and return its parameters.

        Returns
        -------
        parameters : ndarray of shape (n_samples, 4)
        """
        check_is_fitted(self, attributes=['eps_range_'])
        Y_grid = _curves_on_grid(Y, self.grid_)
        return np.vstack([self._fit_one(row, self.grid_) for row in Y_grid])

    def inverse_transform(self, parameters):
        """Rebuild callables by sweeping eps_go and inverting the monotone forward model."""
        check_is_fitted(self, attributes=['eps_range_'])
        parameters = np.atleast_2d(parameters)
        sweep = np.linspace(*self.eps_range_, self.n_eps)
        callables = [Curve(dGs=self._forward(row, sweep), eps_gos=sweep,
                           interp1d_kws=self.interp1d_kws).eps_go for row in parameters]
        return np.array(callables, dtype=object)


#······································································#
# Kernels for clustered designs
#······································································#

class OnColumns(Kernel):
    """Apply ``kernel`` to a column subset of X, ignoring every other column.

    :class:`GaussianProcessRegressor` hands the whole design matrix to every kernel it
    holds, so one matrix has to carry both the model features and any bookkeeping a second
    kernel needs, such as a cluster label. This restricts a kernel to the columns it is
    meant to see.

    Hyperparameters are the wrapped kernel's, forwarded unchanged apart from a
    ``kernel__`` prefix on their names.

    Parameters
    ----------
    kernel : sklearn.gaussian_process.kernels.Kernel
        The kernel to restrict.
    columns : list of int
        Column indices ``kernel`` is shown, in order.
    """

    def __init__(self, kernel, columns) -> None:
        self.kernel = kernel
        self.columns = columns

    def get_params(self, deep=True):
        params = dict(kernel=self.kernel, columns=self.columns)
        if deep:
            params.update({f'kernel__{name}': value
                           for name, value in self.kernel.get_params().items()})
        return params

    @property
    def hyperparameters(self):
        return [Hyperparameter('kernel__' + p.name, p.value_type, p.bounds, p.n_elements)
                for p in self.kernel.hyperparameters]

    @property
    def theta(self):
        return self.kernel.theta

    @theta.setter
    def theta(self, theta):
        self.kernel.theta = theta

    @property
    def bounds(self):
        return self.kernel.bounds

    def _select(self, X):
        return None if X is None else np.asarray(X, dtype=float)[:, self.columns]

    def __call__(self, X, Y=None, eval_gradient=False):
        return self.kernel(self._select(X), self._select(Y), eval_gradient=eval_gradient)

    def diag(self, X):
        return self.kernel.diag(self._select(X))

    def is_stationary(self):
        return self.kernel.is_stationary()


class ClusterEffect(Kernel):
    """``variance`` between rows sharing a cluster label, zero between rows that do not.

    The Gaussian-process form of a random intercept. Rows of one cluster are correlated
    through their shared label, so the process need not explain their difference in y
    through their difference in x, as a stationary kernel alone must.

    A cluster never seen in training matches nothing, so the term contributes zero to its
    posterior mean and only its prior variance to the posterior spread.

    Parameters
    ----------
    column : int, default=-1
        Column of X holding the cluster label. Compared by equality, so any labelling works
        as long as it is consistent between fit and predict.
    variance : float, default=1.0
        Between-cluster variance.
    variance_bounds : pair of float, default=(1e-5, 1e5)
        Bounds searched for ``variance``.
    """

    def __init__(self, column=-1, variance=1.0, variance_bounds=(1e-5, 1e5)) -> None:
        self.column = column
        self.variance = variance
        self.variance_bounds = variance_bounds

    @property
    def hyperparameter_variance(self):
        return Hyperparameter('variance', 'numeric', self.variance_bounds)

    def _labels(self, X):
        return np.asarray(X)[:, self.column]

    def __call__(self, X, Y=None, eval_gradient=False):
        labels = self._labels(X)
        other = labels if Y is None else self._labels(Y)
        K = self.variance * (labels[:, np.newaxis] == other[np.newaxis, :]).astype(float)
        if eval_gradient:
            if self.hyperparameter_variance.fixed:
                return K, np.empty((len(labels), len(labels), 0))
            # scikit-learn optimises log-hyperparameters, so d K / d log(variance) = K
            return K, K[:, :, np.newaxis]
        return K

    def diag(self, X):
        return np.full(len(self._labels(X)), float(self.variance))

    def is_stationary(self):
        return True


#······································································#
# Contact-graph response
#······································································#

# A tuple, not an ndarray: scikit-learn's repr machinery compares every __init__ default
# against a sentinel, and an array default makes that comparison elementwise, raising
# instead of returning a bool. An estimator that cannot repr cannot be displayed in a
# notebook, nor nested in another estimator's repr.
DEFAULT_ALPHAS = tuple(np.logspace(-2, 5, 22))

class GraphResponse(BaseEstimator, TransformerMixin):
    """Linear response of an ``eps_go(dG)`` curve to edits of the contact graph.

    Fitted on reweighted contact-graph variants: for a given system, the same trajectory
    reweighted to a subset of its Go edges (see
    :func:`src.analysis.mbar.reweight_subgraph_system`). Sequence, box and structure are
    fixed within a family, so only the graph moves, which identifies the response free of
    the confounding that afflicts a regression across different systems.

    Identification is by the fixed-effects (within) transform: features and curves are
    centred on their family means before the regression, absorbing every system-level
    effect. This holds only for a linear response, since under a non-linear map the centred
    target is not the map of the centred input, so the response is kept linear.

    Parameters
    ----------
    alphas : array-like, default=:data:`DEFAULT_ALPHAS`
        Ridge penalties searched by :class:`~sklearn.linear_model.RidgeCV`.

    Attributes
    ----------
    imputer_ : SimpleImputer
        Median imputer fitted on the centred features.
    scaler_ : StandardScaler
        Standardiser fitted on the centred, imputed features.
    ridge_ : RidgeCV
        The fitted response, mapping standardised feature deltas to curve deltas. Penalised
        per grid point (``alpha_per_target``), so ``ridge_.alpha_`` is one alpha per dG rather
        than a scalar.
    n_families_ : int
        Number of variant families the response was identified on.
    """

    def __init__(self, alphas=DEFAULT_ALPHAS) -> None:
        self.alphas = alphas

    def fit(self, X, Y, families):
        """Identify the response on family-centred variants.

        Parameters
        ----------
        X : array-like of shape (n_variants, n_features)
            Contact-graph features of each variant.
        Y : array-like of shape (n_variants, n_grid_points)
            Each variant's curve, evaluated on a shared grid.
        families : array-like of shape (n_variants,)
            Family label of each variant, i.e. the system it was reweighted from.
            Centring is performed within these groups.
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        families = np.asarray(families)
        if not (len(X) == len(Y) == len(families)):
            raise ValueError(
                f"X ({len(X)}), Y ({len(Y)}) and families ({len(families)}) "
                "must have the same length."
            )
        if len(X) < 2:
            raise ValueError('At least two variants are required to identify a response.')

        X_centred = (pd.DataFrame(X) - pd.DataFrame(X).groupby(families).transform('mean')).to_numpy()
        Y_centred = (pd.DataFrame(Y) - pd.DataFrame(Y).groupby(families).transform('mean')).to_numpy()

        self.imputer_ = SimpleImputer(strategy='median').fit(X_centred)
        self.scaler_ = StandardScaler().fit(self.imputer_.transform(X_centred))
        # No intercept, since the centred data has none. One penalty per grid point rather
        # than one shared across the curve: coverage thins at the extremes of every sweep,
        # so the grid points differ in how much shrinkage they need.
        self.ridge_ = RidgeCV(alphas=self.alphas, fit_intercept=False,
                              alpha_per_target=True).fit(
            self.scaler_.transform(self.imputer_.transform(X_centred)), Y_centred
        )
        self.n_families_ = len(np.unique(families))

        return self

    def transform(self, X):
        """Predict the curve offset implied by a set of contact graphs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Contact-graph features.

        Returns
        -------
        offsets : ndarray of shape (n_samples, n_grid_points)
            The graph-attributable part of each curve, on the fitting grid.
        """
        check_is_fitted(self, attributes=['ridge_'])
        X = np.asarray(X, dtype=float)
        return self.ridge_.predict(self.scaler_.transform(self.imputer_.transform(X)))


class HybridCurveRegressor(BaseEstimator):
    """Curve regressor that predicts a graph response and a residual separately.

    ``eps_go(dG) = GraphResponse(x_graph) + regressor(x)``

    The response is estimated on contact-graph variants, which are numerous and
    interventional but ill-suited to a kernel: a system and its own variant sit very close
    in feature space while their curves differ, and no single RBF length scale accommodates
    that alongside genuine system-to-system distances. The residual is left to the usual
    small-data regressor on real systems only.

    Parameters
    ----------
    score_pipeline : sklearn estimator
        Fitted on features and fPCA scores of the residual curves. Cloned on fit.
    target_encoder : FPCATargetEncoder
        Encodes residual curves as scores and decodes predictions back. Cloned on fit.
    graph_response : GraphResponse, optional
        The response estimator. A default :class:`GraphResponse` is used if omitted.

    Attributes
    ----------
    score_pipeline_, target_encoder_, graph_response_ : estimators
        The fitted copies.
    """

    def __init__(self, score_pipeline, target_encoder, graph_response=None) -> None:
        self.score_pipeline = score_pipeline
        self.target_encoder = target_encoder
        self.graph_response = graph_response

    def fit(self, X, Y, X_graph, X_variants, Y_variants, variant_families):
        """Fit the response on variants, then the regressor on what it leaves behind.

        Parameters
        ----------
        X : array-like of shape (n_systems, n_features)
            Model features of the real systems.
        Y : array-like of shape (n_systems,) of callables, or (n_systems, n_grid_points)
            Each system's ``eps_go(dG)`` curve, as callables or grid values.
        X_graph : array-like of shape (n_systems, n_graph_features)
            Contact-graph features of the real systems, in the same column order as
            ``X_variants``.
        X_variants, Y_variants, variant_families
            Passed through to :meth:`GraphResponse.fit`.
        """
        self.graph_response_ = clone(
            self.graph_response if self.graph_response is not None else GraphResponse()
        ).fit(X_variants, Y_variants, variant_families)
        self.target_encoder_ = clone(self.target_encoder)
        self.score_pipeline_ = clone(self.score_pipeline)

        grid = np.asarray(self.target_encoder_.grid, dtype=float)
        Y_grid = _curves_on_grid(Y, grid)
        offsets = self.graph_response_.transform(X_graph)
        if offsets.shape != Y_grid.shape:
            raise ValueError(
                f"The graph response returns curves of width {offsets.shape[1]}, but the "
                f"target encoder's grid has {Y_grid.shape[1]} points; they must be "
                "fitted on the same grid."
            )

        residuals = [Curve(dGs=grid, eps_gos=row).eps_go for row in Y_grid - offsets]
        self.score_pipeline_.fit(X, self.target_encoder_.fit_transform(residuals))

        return self

    def predict(self, X, X_graph):
        """Predict ``eps_go(dG)`` curves.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Model features.
        X_graph : array-like of shape (n_samples, n_graph_features)
            Contact-graph features, in the column order used at fit time.

        Returns
        -------
        Y_callables : ndarray of shape (n_samples,), dtype=object
            Array of callable interpolators, matching
            :meth:`FPCATargetEncoder.inverse_transform`.
        """
        check_is_fitted(self, attributes=['score_pipeline_'])

        grid = self.target_encoder_.grid_
        predicted = self.target_encoder_.inverse_transform(self.score_pipeline_.predict(X))
        residuals = np.vstack([curve(grid) for curve in predicted])
        total = residuals + self.graph_response_.transform(X_graph)

        return np.array([Curve(dGs=grid, eps_gos=row).eps_go for row in total], dtype=object)


class CurveRegressor(BaseEstimator):
    """Curve regressor: features to an ``eps_go(dG)`` curve, through an encoded target.

    ``eps_go(dG) = target_encoder.inverse_transform(regressor.predict(X))``

    The single-step counterpart to :class:`HybridCurveRegressor`: no graph response, no
    variants, real systems only.

    Parameters
    ----------
    regressor : sklearn estimator
        Fitted on features and the encoded scores of the curves. Cloned on fit.
    target_encoder : FPCATargetEncoder or ParametricTargetEncoder
        Encodes curves as scores and decodes predictions back. Cloned on fit.

    Attributes
    ----------
    regressor_, target_encoder_ : estimators
        The fitted copies.
    """

    def __init__(self, regressor, target_encoder) -> None:
        self.regressor = regressor
        self.target_encoder = target_encoder

    def fit(self, X, Y):
        """Fit the encoder on ``Y``, then the regressor on ``X`` and the encoded scores.

        Parameters
        ----------
        X : array-like of shape (n_systems, n_features)
            Model features.
        Y : array-like of shape (n_systems,) of callables, or (n_systems, n_grid_points)
            Each system's ``eps_go(dG)`` curve, as callables or grid values.
        """
        self.target_encoder_ = clone(self.target_encoder)
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, self.target_encoder_.fit_transform(Y))
        return self

    def predict(self, X):
        """Predict ``eps_go(dG)`` curves.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Model features, in the column order used at fit time.

        Returns
        -------
        Y_callables : ndarray of shape (n_samples,), dtype=object
            Array of callable interpolators, matching
            :meth:`FPCATargetEncoder.inverse_transform`.
        """
        check_is_fitted(self, attributes=['regressor_'])
        return self.target_encoder_.inverse_transform(self.regressor_.predict(X))


def _curves_on_grid(Y, grid):
    """Evaluate an array of curve callables on ``grid``, passing arrays through."""
    if len(Y) > 0 and callable(Y[0]):
        return np.array([curve(grid) for curve in Y], dtype=float)
    return np.asarray(Y, dtype=float)


#······································································#
# Cross-validation
#······································································#

def fit_predict_fold(pipeline, X_train, y_train, X_test, y_test):
    """Fit a pipeline on one cross-validation fold and predict its held-out fold.

    Defined at module level so it can be pickled and dispatched to worker
    processes, e.g. to parallelise a Leave-One-System-Out loop across a
    :class:`ProcessPoolExecutor`.

    Parameters
    ----------
    pipeline : sklearn.base.BaseEstimator
        An unfitted estimator or pipeline, already ``clone``-d by the caller.
    X_train, y_train : ndarray
        Training features and targets for this fold.
    X_test, y_test : ndarray
        Held-out features and targets for this fold.

    Returns
    -------
    y_test : ndarray
        The held-out targets, passed through unchanged.
    y_pred : ndarray
        Predictions for the held-out fold.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return y_test, y_pred

