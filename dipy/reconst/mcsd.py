import warnings

import numpy as np
import numbers
from dipy.core import geometry as geo
from dipy.core.gradients import (GradientTable, gradient_table,
                                 unique_bvals_tol, get_bval_indices)
from dipy.data import default_sphere
from dipy.reconst import shm
from dipy.reconst.csdeconv import response_from_mask_ssst
from dipy.reconst.dti import (TensorModel, fractional_anisotropy,
                              mean_diffusivity)
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst.utils import _roi_in_volume, _mask_from_roi
from dipy.sims.voxel import single_tensor

from dipy.utils.optpkg import optional_package
cvx, have_cvxpy, _ = optional_package("cvxpy")

SH_CONST = .5 / np.sqrt(np.pi)


def multi_tissue_basis(gtab, sh_order, iso_comp):
    """
    Builds a basis for multi-shell multi-tissue CSD model.

    Parameters
    ----------
    gtab : GradientTable
    sh_order : int
    iso_comp: int
        Number of tissue compartments for running the MSMT-CSD. Minimum
        number of compartments required is 2.

    Returns
    -------
    B : ndarray
        Matrix of the spherical harmonics model used to fit the data
    m : int ``|m| <= n``
        The order of the harmonic.
    n : int ``>= 0``
        The degree of the harmonic.
    """
    if iso_comp < 2:
        msg = ("Multi-tissue CSD requires at least 2 tissue compartments")
        raise ValueError(msg)
    r, theta, phi = geo.cart2sphere(*gtab.gradients.T)
    m, n = shm.sph_harm_ind_list(sh_order)
    B = shm.real_sph_harm(m, n, theta[:, None], phi[:, None])
    B[np.ix_(gtab.b0s_mask, n > 0)] = 0.

    iso = np.empty([B.shape[0], iso_comp])
    iso[:] = SH_CONST

    B = np.concatenate([iso, B], axis=1)
    return B, m, n


class MultiShellResponse(object):

    def __init__(self, response, sh_order, shells):
        """ Estimate Multi Shell response function for multiple tissues and
        multiple shells.

        Parameters
        ----------
        response : tuple or AxSymShResponse object
            A tuple with two elements. The first is the eigen-values as an (3,)
            ndarray and the second is the signal value for the response
            function without diffusion weighting.  This is to be able to
            generate a single fiber synthetic signal.
        sh_order : int
        shells : int
            Number of shells in the data
        """
        self.response = response
        self.sh_order = sh_order
        self.n = np.arange(0, sh_order + 1, 2)
        self.m = np.zeros_like(self.n)
        self.shells = shells
        if self.iso < 1:
            raise ValueError("sh_order and shape of response do not agree")

    @property
    def iso(self):
        return self.response.shape[1] - (self.sh_order // 2) - 1


def _inflate_response(response, gtab, n, delta):
    """Used to inflate the response for the `multiplier_matrix` in the
    `MultiShellDeconvModel`.
    Parameters
    ----------
    response : tuple or AxSymShResponse object
        A tuple with two elements. The first is the eigen-values as an (3,)
        ndarray and the second is the signal value for the response
        function without diffusion weighting.  This is to be able to
        generate a single fiber synthetic signal. The response function
        will be used as deconvolution kernel ([1]_)
    gtab : GradientTable
    n : int ``>= 0``
        The degree of the harmonic.
    delta : Delta generated from `_basic_delta`
    """
    if any((n % 2) != 0) or (n.max() // 2) >= response.sh_order:
        raise ValueError("Response and n do not match")

    iso = response.iso
    n_idx = np.empty(len(n) + iso, dtype=int)
    n_idx[:iso] = np.arange(0, iso)
    n_idx[iso:] = n // 2 + iso
    diff = abs(response.shells[:, None] - gtab.bvals)
    b_idx = np.argmin(diff, axis=0)
    kernal = response.response / delta

    return kernal[np.ix_(b_idx, n_idx)]


def _basic_delta(iso, m, n, theta, phi):
    """Simple delta function
    Parameters
    ----------
    iso: int
        Number of tissue compartments for running the MSMT-CSD. Minimum
        number of compartments required is 2.
        Default: 2
    m : int ``|m| <= n``
        The order of the harmonic.
    n : int ``>= 0``
        The degree of the harmonic.
    theta : array_like
       inclination or polar angle
    phi : array_like
       azimuth angle
    """
    wm_d = shm.gen_dirac(m, n, theta, phi)
    iso_d = [SH_CONST] * iso
    return np.concatenate([iso_d, wm_d])


class MultiShellDeconvModel(shm.SphHarmModel):
    def __init__(self, gtab, response, reg_sphere=default_sphere, iso=2):
        r"""
        Multi-Shell Multi-Tissue Constrained Spherical Deconvolution
        (MSMT-CSD) [1]_. This method extends the CSD model proposed in [2]_ by
        the estimation of multiple response functions as a function of multiple
        b-values and multiple tissue types.

        Spherical deconvolution computes a fiber orientation distribution
        (FOD), also called fiber ODF (fODF) [2]_. The fODF is derived from
        different tissue types and thus overcomes the overestimation of WM in
        GM and CSF areas.

        The response function is based on the different tissue types
        and is provided as input to the MultiShellDeconvModel.
        It will be used as deconvolution kernel, as described in [2]_.

        Parameters
        ----------
        gtab : GradientTable
        response : tuple or AxSymShResponse object
            A tuple with two elements. The first is the eigen-values as an (3,)
            ndarray and the second is the signal value for the response
            function without diffusion weighting.  This is to be able to
            generate a single fiber synthetic signal. The response function
            will be used as deconvolution kernel ([1]_)
        reg_sphere : Sphere (optional)
            sphere used to build the regularization B matrix.
            Default: 'symmetric362'.
        iso: int (optional)
            Number of tissue compartments for running the MSMT-CSD. Minimum
            number of compartments required is 2.
            Default: 2

        References
        ----------
        .. [1] Jeurissen, B., et al. NeuroImage 2014. Multi-tissue constrained
               spherical deconvolution for improved analysis of multi-shell
               diffusion MRI data
        .. [2] Tournier, J.D., et al. NeuroImage 2007. Robust determination of
               the fibre orientation distribution in diffusion MRI:
               Non-negativity constrained super-resolved spherical
               deconvolution
        .. [3] Tournier, J.D, et al. Imaging Systems and Technology
               2012. MRtrix: Diffusion Tractography in Crossing Fiber Regions
        """
        if not iso >= 2:
            msg = ("Multi-tissue CSD requires at least 2 tissue compartments")
            raise ValueError(msg)

        sh_order = response.sh_order
        super(MultiShellDeconvModel, self).__init__(gtab)
        B, m, n = multi_tissue_basis(gtab, sh_order, iso)

        delta = _basic_delta(response.iso, response.m, response.n, 0., 0.)
        self.delta = delta
        multiplier_matrix = _inflate_response(response, gtab, n, delta)

        r, theta, phi = geo.cart2sphere(*reg_sphere.vertices.T)
        odf_reg, _, _ = shm.real_sym_sh_basis(sh_order, theta, phi)
        reg = np.zeros([i + iso for i in odf_reg.shape])
        reg[:iso, :iso] = np.eye(iso)
        reg[iso:, iso:] = odf_reg

        X = B * multiplier_matrix

        self.fitter = QpFitter(X, reg)
        self.sh_order = sh_order
        self._X = X
        self.sphere = reg_sphere
        self.response = response
        self.B_dwi = B
        self.m = m
        self.n = n

    def predict(self, params, gtab=None, S0=None):
        """Compute a signal prediction given spherical harmonic coefficients
        for the provided GradientTable class instance.

        Parameters
        ----------
        params : ndarray
            The spherical harmonic representation of the FOD from which to make
            the signal prediction.
        gtab : GradientTable
            The gradients for which the signal will be predicted. Use the
            model's gradient table by default.
        S0 : ndarray or float
            The non diffusion-weighted signal value.
            Default : None
        """
        if gtab is None:
            X = self._X
        else:
            iso = self.response.iso
            B, m, n = multi_tissue_basis(gtab, self.sh_order, iso)
            multiplier_matrix = _inflate_response(self.response, gtab, n,
                                                  self.delta)
            X = B * multiplier_matrix
        return np.dot(params, X.T)

    @multi_voxel_fit
    def fit(self, data):
        coeff = self.fitter(data)
        return MSDeconvFit(self, coeff, None)


class MSDeconvFit(shm.SphHarmFit):

    def __init__(self, model, coeff, mask):
        """
        Abstract class which holds the fit result of MultiShellDeconvModel.
        Inherits the SphHarmFit which fits the diffusion data to a spherical
        harmonic model.

        Parameters
        ----------
        model: object
            MultiShellDeconvModel
        coeff : array
            Spherical harmonic coefficients for the ODF.
        mask: ndarray
            Mask for fitting
        """
        self._shm_coef = coeff
        self.mask = mask
        self.model = model

    @property
    def shm_coeff(self):
        return self._shm_coef[..., self.model.response.iso:]

    @property
    def volume_fractions(self):
        tissue_classes = self.model.response.iso + 1
        return self._shm_coef[..., :tissue_classes] / SH_CONST


def solve_qp(P, Q, G, H):
    r"""
    Helper function to set up and solve the Quadratic Program (QP) in CVXPY.
    A QP problem has the following form:
    minimize      1/2 x' P x + Q' x
    subject to    G x <= H

    Here the QP solver is based on CVXPY and uses OSQP.

    Parameters
    ----------
    P : ndarray
        n x n matrix for the primal QP objective function.
    Q : ndarray
        n x 1 matrix for the primal QP objective function.
    G : ndarray
        m x n matrix for the inequality constraint.
    H : ndarray
        m x 1 matrix for the inequality constraint.

    Returns
    -------
    x : array
        Optimal solution to the QP problem.
    """
    x = cvx.Variable(Q.shape[0])
    P = cvx.Constant(P)
    objective = cvx.Minimize(0.5 * cvx.quad_form(x, P) + Q * x)
    constraints = [G*x <= H]

    # setting up the problem
    prob = cvx.Problem(objective, constraints)
    prob.solve()
    opt = np.array(x.value).reshape((Q.shape[0],))
    return opt


class QpFitter(object):

    def __init__(self, X, reg):
        r"""
        Makes use of the quadratic programming solver `solve_qp` to fit the
        model. The initialization for the model is done using the warm-start by
        default in `CVXPY`.

        Parameters
        ----------
        X : ndarray
            Matrix to be fit by the QP solver calculated in
            `MultiShellDeconvModel`
        reg : ndarray
            the regularization B matrix calculated in `MultiShellDeconvModel`
        """
        self._P = P = np.dot(X.T, X)
        self._X = X

        self._reg = reg
        self._P_mat = np.array(P)
        self._reg_mat = np.array(-reg)
        self._h_mat = np.array([0])

    def __call__(self, signal):
        Q = np.dot(self._X.T, signal)
        Q_mat = np.array(-Q)
        fodf_sh = solve_qp(self._P_mat, Q_mat, self._reg_mat, self._h_mat)
        return fodf_sh


def multi_shell_fiber_response(sh_order, bvals, evals, csf_md, gm_md,
                               sphere=None):
    """Fiber response function estimation for multi-shell data.

    Parameters
    ----------
    sh_order : int
         Maximum spherical harmonics order.
    bvals : ndarray
        Array containing the b-values.
    evals : (3,) ndarray
        Eigenvalues of the diffusion tensor.
    csf_md : float
        CSF tissue mean diffusivity value.
    gm_md : float
        GM tissue mean diffusivity value.
    sphere : `dipy.core.Sphere` instance, optional
        Sphere where the signal will be evaluated.

    Returns
    -------
    MultiShellResponse
        MultiShellResponse object.
    """

    bvals = np.array(bvals, copy=True)
    evecs = np.zeros((3, 3))
    z = np.array([0, 0, 1.])
    evecs[:, 0] = z
    evecs[:2, 1:] = np.eye(2)

    n = np.arange(0, sh_order + 1, 2)
    m = np.zeros_like(n)

    if sphere is None:
        sphere = default_sphere

    big_sphere = sphere.subdivide()
    theta, phi = big_sphere.theta, big_sphere.phi

    B = shm.real_sph_harm(m, n, theta[:, None], phi[:, None])
    A = shm.real_sph_harm(0, 0, 0, 0)

    response = np.empty([len(bvals), len(n) + 2])
    for i, bvalue in enumerate(bvals):
        gtab = GradientTable(big_sphere.vertices * bvalue)
        wm_response = single_tensor(gtab, 1., evals, evecs, snr=None)
        response[i, 2:] = np.linalg.lstsq(B, wm_response)[0]

        response[i, 0] = np.exp(-bvalue * csf_md) / A
        response[i, 1] = np.exp(-bvalue * gm_md) / A

    return MultiShellResponse(response, sh_order, bvals)


def mask_for_response_msmt(gtab, data, roi_center=None, roi_radii=10,
                           fa_data=None, wm_fa_thr=0.7, gm_fa_thr=0.3,
                           csf_fa_thr=0.15, md_data=None,
                           gm_md_thr=0.001, csf_md_thr=0.003):
    """ Computation of masks for msmt response function using FA and MD.

    Parameters
    ----------
    gtab : GradientTable
    data : ndarray
        diffusion data (4D)
    roi_center : array-like, (3,)
        Center of ROI in data. If center is None, it is assumed that it is
        the center of the volume with shape `data.shape[:3]`.
    roi_radii : int or array-like, (3,)
        radii of cuboid ROI
    fa_data : ndarray
        FA data, optionnal.
    wm_fa_thr : float
        FA threshold for WM.
    gm_fa_thr : float
        FA threshold for GM.
    csf_fa_thr : float
        FA threshold for CSF.
    md_data : ndarray
        MD data, optionnal.
    gm_md_thr : float
        MD threshold for GM.
    csf_md_thr : float
        MD threshold for CSF.

    Returns
    -------
    mask_wm : ndarray
        Mask of voxels within the ROI and with FA above the FA threshold
        for WM.
    mask_gm : ndarray
        Mask of voxels within the ROI and with FA below the FA threshold
        for GM and with MD above the MD threshold for GM.
    mask_csf : ndarray
        Mask of voxels within the ROI and with FA below the FA threshold
        for CSF and with MD above the MD threshold for CSF.

    Notes
    -----
    In msmt-CSD there is an important pre-processing step: the estimation of
    every tissue's response function. In order to do this, we look for voxels
    corresponding to WM, GM and CSF. This function aims to accomplish that by
    returning a mask of voxels within a ROI and who respect some threshold
    constraints, for each tissue. More precisely, the WM mask must have a FA
    value above a given threshold. The GM mask and CSF mask must have a FA
    below given thresholds and a MD above other thresholds. Of course, if we
    haven't precalculated FA and MD, we need to fit a Tensor model to the
    datasets. The option is given to the user with this function. Note that
    the user has to give either the FA and MD data, or none of them.
    """

    if len(data.shape) < 4:
        msg = """Data must be 4D (3D image + directions). To use a 2D image,
        please reshape it into a (N, N, 1, ndirs) array."""
        raise ValueError(msg)

    if isinstance(roi_radii, numbers.Number):
        roi_radii = (roi_radii, roi_radii, roi_radii)

    if roi_center is None:
        roi_center = np.array(data.shape[:3]) // 2

    roi_radii = _roi_in_volume(data.shape, np.asarray(roi_center),
                               np.asarray(roi_radii))

    roi_mask = _mask_from_roi(data.shape[:3], roi_center, roi_radii)

    if fa_data is None and md_data is None:
        ten = TensorModel(gtab)
        tenfit = ten.fit(data, mask=roi_mask)
        fa = fractional_anisotropy(tenfit.evals)
        fa[np.isnan(fa)] = 0
        md = mean_diffusivity(tenfit.evals)
        md[np.isnan(md)] = 0
    elif fa_data is not None and md_data is None:
        msg = "Missing MD data."
        raise ValueError(msg)
    elif fa_data is None and md_data is not None:
        msg = "Missing FA data."
        raise ValueError(msg)
    else:
        fa = fa_data * roi_mask
        md = md_data * roi_mask

    mask_wm = np.zeros(fa.shape)
    mask_wm[fa > wm_fa_thr] = 1

    md_mask_gm = np.ones(md.shape)
    md_mask_gm[(md > gm_md_thr)] = 0

    fa_mask_gm = np.zeros(fa.shape)
    fa_mask_gm[(fa < gm_fa_thr) & (fa >= 0)] = 1

    mask_gm = md_mask_gm * fa_mask_gm

    md_mask_csf = np.ones(md.shape)
    md_mask_csf[(md > csf_md_thr)] = 0

    fa_mask_csf = np.zeros(fa.shape)
    fa_mask_csf[(fa < csf_fa_thr) & (fa >= 0)] = 1

    mask_csf = md_mask_csf * fa_mask_csf

    msg = """No voxel with a {0} than {1} were found.
    Try a larger roi or a {2} threshold for {3}."""

    if np.sum(mask_wm) == 0:
        msg_fa = msg.format('FA higher', str(wm_fa_thr), 'lower FA', 'WM')
        warnings.warn(msg_fa, UserWarning)

    if np.sum(mask_gm) == 0:
        msg_fa = msg.format('FA lower', str(gm_fa_thr), 'higher FA', 'GM')
        msg_md = msg.format('MD lower', str(gm_md_thr), 'higher MD', 'GM')
        warnings.warn(msg_fa, UserWarning)
        warnings.warn(msg_md, UserWarning)

    if np.sum(mask_csf) == 0:
        msg_fa = msg.format('FA lower', str(csf_fa_thr), 'higher FA', 'CSF')
        msg_md = msg.format('MD lower', str(csf_md_thr), 'higher MD', 'CSF')
        warnings.warn(msg_fa, UserWarning)
        warnings.warn(msg_md, UserWarning)

    return mask_wm, mask_gm, mask_csf


def response_from_mask_msmt(gtab, data, mask_wm, mask_gm, mask_csf, tol=20):
    """ Computation of msmt response functions from given tissues masks.

    Parameters
    ----------
    gtab : GradientTable
    data : ndarray
        diffusion data
    mask_wm : ndarray
        mask from where to compute the WM response function.
    mask_gm : ndarray
        mask from where to compute the GM response function.
    mask_csf : ndarray
        mask from where to compute the CSF response function.
    tol : int
        tolerance gap for b-values clustering. (Default = 20)

    Returns
    -------
    response_wm : tuple, (2,)
        (`evals`, `S0`) for WM.
    response_gm : tuple, (2,)
        (`evals`, `S0`) for GM.
    response_csf : tuple, (2,)
        (`evals`, `S0`) for CSF.

    Notes
    -----
    In msmt-CSD there is an important pre-processing step: the estimation of
    every tissue's response function. In order to do this, we look for voxels
    corresponding to WM, GM and CSF. This information can be obtained by using
    mcsd.mask_for_response_msmt() through masks of selected voxels. The present
    function uses such masks to compute the msmt response functions.

    For the responses, we base our approach on the function
    csdeconv.response_from_mask_ssst(), with the added layers of multishell and
    multi-tissue (see the ssst function for more information about the
    computation of the ssst response function). This means that for each tissue
    we use the previously found masks and loop on them. For each mask, we loop
    on the b-values (clustered using the tolerance gap) to get many responses
    and then average them to get one response per tissue.
    """

    bvals = gtab.bvals
    bvecs = gtab.bvecs

    list_bvals = unique_bvals_tol(bvals, tol)

    b0_indices = get_bval_indices(bvals, list_bvals[0], tol)
    b0_map = np.mean(data[..., b0_indices], axis=-1)[..., np.newaxis]

    masks = [mask_wm, mask_gm, mask_csf]
    tissue_responses = []
    for mask in masks:
        responses = []
        for bval in list_bvals[1:]:
            indices = get_bval_indices(bvals, bval, tol)

            bvecs_sub = np.concatenate([[bvecs[b0_indices[0]]],
                                       bvecs[indices]])
            bvals_sub = np.concatenate([[0], bvals[indices]])

            data_conc = np.concatenate([b0_map, data[..., indices]], axis=3)

            gtab = gradient_table(bvals_sub, bvecs_sub)
            response, _ = response_from_mask_ssst(gtab, data_conc, mask)

            responses.append(list(response))
        response_mean = np.mean(responses, axis=0)
        tissue_responses.append(list(np.concatenate([response_mean[0],
                                                    [response_mean[1]]])))

    return tissue_responses[0], tissue_responses[1], tissue_responses[2]


def auto_response_msmt(gtab, data, tol=20, roi_center=None, roi_radii=10,
                       fa_data=None, wm_fa_thr=0.7, gm_fa_thr=0.3,
                       csf_fa_thr=0.15, md_data=None,
                       gm_md_thr=0.001, csf_md_thr=0.003):
    """ Automatic estimation of msmt response functions using FA and MD.

    Parameters
    ----------
    gtab : GradientTable
    data : ndarray
        diffusion data
    roi_center : array-like, (3,)
        Center of ROI in data. If center is None, it is assumed that it is
        the center of the volume with shape `data.shape[:3]`.
    roi_radii : int or array-like, (3,)
        radii of cuboid ROI
    fa_data : ndarray
        FA data, optionnal.
    wm_fa_thr : float
        FA threshold for WM.
    gm_fa_thr : float
        FA threshold for GM.
    csf_fa_thr : float
        FA threshold for CSF.
    md_data : ndarray
        MD data, optionnal.
    gm_md_thr : float
        MD threshold for GM.
    csf_md_thr : float
        MD threshold for CSF.

    Returns
    -------
    response_wm : tuple, (2,)
        (`evals`, `S0`) for WM.
    response_gm : tuple, (2,)
        (`evals`, `S0`) for GM.
    response_csf : tuple, (2,)
        (`evals`, `S0`) for CSF.

    Notes
    -----
    In msmt-CSD there is an important pre-processing step: the estimation of
    every tissue's response function. In order to do this, we look for voxels
    corresponding to WM, GM and CSF. We get this information from
    mcsd.mask_for_response_msmt(), which returns masks of selected voxels
    (more details are available in the description of the function).

    With the masks, we compute the response functions by using
    mcsd.response_from_mask_msmt(), which returns the `response` for each
    tissue (more details are available in the description of the function).
    """

    mask_wm, mask_gm, mask_csf = mask_for_response_msmt(gtab, data,
                                                        roi_center,
                                                        roi_radii,
                                                        fa_data, wm_fa_thr,
                                                        gm_fa_thr, csf_fa_thr,
                                                        md_data, gm_md_thr,
                                                        csf_md_thr)
    response_wm, response_gm, response_csf = response_from_mask_msmt(
                                                        gtab, data,
                                                        mask_wm, mask_gm,
                                                        mask_csf, tol)

    return response_wm, response_gm, response_csf
