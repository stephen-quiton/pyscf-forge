#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regularized restricted MP2.

The implementations follow the kappa and sigma regularizers of J. Lee and
M. Head-Gordon, J. Chem. Theory Comput. 14, 5203 (2018),
doi:10.1021/acs.jctc.8b00731, and the MP2 parameterization of J. Shee et al.,
J. Phys. Chem. Lett. 12, 12084 (2021), doi:10.1021/acs.jpclett.1c03468.

For a positive excitation gap ``Delta``, the energy damping factors are
``(1-exp(-kappa*Delta))**2``, ``1-exp(-sigma*Delta)``, and
``1-exp(-sigma*Delta**2)``.  The stored first-order amplitudes contain one
factor; kappa applies the second factor only during the energy contraction.
"""

import numbers

import numpy

from pyscf import lib
from pyscf import scf
from pyscf import __config__
from pyscf.lib import logger
from pyscf.mp import mp2


WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)
_DEFAULT_REG_PARAMS = {
    'kappa': 1.1,
    'sigma': 0.7,
    'sigma2': 0.4,
}
_REGULARIZER_ALIASES = {
    'sigma^2': 'sigma2',
}
_LOG_MAX = numpy.log(numpy.finfo(float).max)


def _normalize_regularizer(regularizer):
    if not isinstance(regularizer, str):
        raise ValueError("regularizer must be one of 'kappa', 'sigma', or 'sigma2'")
    regularizer = regularizer.strip().lower()
    regularizer = _REGULARIZER_ALIASES.get(regularizer, regularizer)
    if regularizer not in _DEFAULT_REG_PARAMS:
        raise ValueError(
            "Unknown regularizer %r; choose 'kappa', 'sigma', or 'sigma2'"
            % regularizer)
    return regularizer


def _resolve_regularizer(regularizer, reg_param):
    regularizer = _normalize_regularizer(regularizer)
    if reg_param is None:
        reg_param = _DEFAULT_REG_PARAMS[regularizer]
    elif isinstance(reg_param, (bool, numpy.bool_)) or not isinstance(reg_param, numbers.Real):
        raise ValueError('reg_param must be a finite nonnegative real number')
    else:
        reg_param = float(reg_param)

    if not numpy.isfinite(reg_param) or reg_param < 0:
        raise ValueError('reg_param must be a finite nonnegative real number')
    return regularizer, reg_param


def _one_minus_exp_neg(x):
    """Evaluate ``1-exp(-x)`` without overflow for finite real ``x``."""
    x = numpy.asarray(x, dtype=float)
    # Positive gaps only require the upper clip.  The lower clip also keeps
    # diagnostics on an accidentally negative gap finite and warning-free.
    x = numpy.clip(x, -_LOG_MAX, _LOG_MAX)
    return -numpy.expm1(-x)


def _regularization_factors(delta, regularizer, reg_param):
    """Return ``(damping, damping / delta)`` safely.

    The second value uses the analytic zero-gap limits.  Kappa applies the
    first value a second time in the energy contraction.  The zero-gap limits
    of the second value are ``kappa``, ``sigma``, and zero, respectively.
    """
    regularizer, reg_param = _resolve_regularizer(regularizer, reg_param)
    delta = numpy.asarray(delta, dtype=float)
    damping = numpy.zeros(delta.shape, dtype=float)
    damping_over_gap = numpy.zeros(delta.shape, dtype=float)

    # This branch also avoids 0*inf producing NaN for the HF limit.
    if reg_param == 0:
        return damping, damping_over_gap

    with numpy.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        if regularizer == 'sigma2':
            gap_sq = delta * delta
            x = reg_param * gap_sq
            damping = _one_minus_exp_neg(x)
            zero_gap_limit = 0.0
            small_x_limit = reg_param * delta
        else:
            x = reg_param * delta
            damping = _one_minus_exp_neg(x)
            zero_gap_limit = reg_param
            small_x_limit = reg_param

        nonzero = delta != 0
        numpy.divide(damping, delta, out=damping_over_gap, where=nonzero)

        # The series is useful below the resolution at which a quotient of
        # two nearly equal floating-point numbers loses digits.  It also
        # handles exact zero without ever forming 0/0.
        small = numpy.isfinite(x) & (numpy.abs(x) < 1e-4)
        series = small_x_limit * (1 - x / 2 + x * x / 6 - x * x * x / 24)
        damping_over_gap = numpy.where(small, series, damping_over_gap)
        damping_over_gap = numpy.where(nonzero, damping_over_gap, zero_gap_limit)

    return damping, damping_over_gap


def _ovov_block(ovov, i, nocc, nvir):
    if isinstance(ovov, numpy.ndarray) and ovov.ndim == 4:
        gi = ovov[i]
    else:
        gi = numpy.asarray(ovov[i*nvir:(i+1)*nvir])
    return gi.reshape(nvir, nocc, nvir).transpose(1, 0, 2)


def _is_rhf_reference(mf):
    return (isinstance(mf, scf.hf.RHF) and
            not mf.istype('ROHF') and not mf.istype('KohnShamDFT'))


def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2, verbose=None):
    """Initialize regularized amplitudes and compute their correlation energy."""
    if mo_energy is not None or mo_coeff is not None:
        # Match the compatibility guard in pyscf.mp.mp2.kernel.
        assert mp.frozen is None

    if eris is None:
        eris = mp.ao2mo(mo_coeff)
    if mo_energy is None:
        mo_energy = eris.mo_energy

    regularizer, reg_param = _resolve_regularizer(mp.regularizer, mp.reg_param)
    mp.regularizer, mp.reg_param = regularizer, reg_param

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    mo_energy = numpy.asarray(mo_energy)
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
    if with_t2:
        t2 = numpy.empty((nocc, nocc, nvir, nvir), dtype=eris.ovov.dtype)
    else:
        t2 = None

    emp2_ss = emp2_os = 0
    for i in range(nocc):
        gi = _ovov_block(eris.ovov, i, nocc, nvir)
        delta = -(eia[:, None, :] + eia[i][None, :, None])
        damping, damping_over_gap = _regularization_factors(
            delta, regularizer, reg_param)
        t2i = -gi.conj() * damping_over_gap

        if regularizer == 'kappa':
            t2_energy = t2i * damping
        else:
            t2_energy = t2i
        edi = numpy.einsum('jab,jab', t2_energy, gi) * 2
        exi = -numpy.einsum('jab,jba', t2_energy, gi)
        emp2_ss += edi * 0.5 + exi
        emp2_os += edi * 0.5
        if with_t2:
            t2[i] = t2i

    emp2_ss = emp2_ss.real
    emp2_os = emp2_os.real
    emp2 = lib.tag_array(emp2_ss + emp2_os,
                         e_corr_ss=emp2_ss, e_corr_os=emp2_os)
    return emp2.real, t2


class RegMP2(mp2.MP2):
    """Regularized MP2 for a converged canonical restricted HF reference.

    Args:
        mf: A converged :class:`pyscf.scf.hf.RHF` object, including a
            density-fitted RHF object.
        frozen, mo_coeff, mo_occ: The usual PySCF MP2 inputs.
        regularizer: ``'kappa'``, ``'sigma'``, or ``'sigma2'``.  ``'sigma^2'``
            is accepted as an alias.
        reg_param: A finite nonnegative parameter.  If omitted, the defaults
            are 1.1, 0.7, and 0.4 for kappa, sigma, and sigma2.
    """

    _keys = mp2.MP2._keys | {'regularizer', 'reg_param'}

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None,
                 regularizer='kappa', reg_param=None):
        if not _is_rhf_reference(mf):
            raise NotImplementedError('RegMP2 supports canonical restricted RHF references only')
        regularizer, reg_param = _resolve_regularizer(regularizer, reg_param)
        mp2.MP2.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.regularizer = regularizer
        self.reg_param = reg_param

    def _check_supported_reference(self, mo_coeff=None, mo_occ=None, mo_energy=None):
        if not _is_rhf_reference(self._scf):
            raise NotImplementedError('RegMP2 supports canonical restricted RHF references only')
        if not getattr(self._scf, 'converged', False):
            raise NotImplementedError(
                'RegMP2 does not support noncanonical or iterative MP2 references')

        mo_coeff = self.mo_coeff if mo_coeff is None else mo_coeff
        mo_occ = self.mo_occ if mo_occ is None else mo_occ
        if mo_coeff is None or mo_occ is None:
            raise NotImplementedError('RegMP2 requires converged canonical RHF orbitals')
        if numpy.asarray(mo_coeff).ndim != 2 or numpy.asarray(mo_occ).ndim != 1:
            raise NotImplementedError(
                'RegMP2 supports restricted (not spin-resolved) orbitals only')
        if not numpy.all((numpy.isclose(mo_occ, 0) | numpy.isclose(mo_occ, 2))):
            raise NotImplementedError('RegMP2 requires integer restricted-RHF occupations')

        # A converged RHF calculation already provides the canonical reference.
        # Only the same orbitals and occupations are supported; rebuilding a
        # Fock matrix here is both expensive and unnecessary.
        same_coeff = (mo_coeff is self._scf.mo_coeff or
                      (numpy.shape(mo_coeff) == numpy.shape(self._scf.mo_coeff) and
                       numpy.allclose(mo_coeff, self._scf.mo_coeff)))
        same_occ = (mo_occ is self._scf.mo_occ or
                    (numpy.shape(mo_occ) == numpy.shape(self._scf.mo_occ) and
                     numpy.allclose(mo_occ, self._scf.mo_occ)))
        if not (same_coeff and same_occ):
            raise NotImplementedError('RegMP2 supports canonical RHF orbitals only')
        if mo_energy is not None:
            reference_energy = numpy.asarray(self._scf.mo_energy)
            if (numpy.shape(mo_energy) != reference_energy.shape or
                    not numpy.allclose(mo_energy, reference_energy)):
                raise NotImplementedError('RegMP2 supports canonical RHF orbital energies only')

    def dump_flags(self, verbose=None):
        regularizer, reg_param = _resolve_regularizer(self.regularizer, self.reg_param)
        self.regularizer, self.reg_param = regularizer, reg_param
        mp2.MP2.dump_flags(self, verbose)
        log = logger.new_logger(self, verbose)
        log.info('regularizer = %s, reg_param = %g', regularizer, reg_param)
        return self

    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        self._check_supported_reference(mo_coeff, self.mo_occ, mo_energy)
        return mp2.MP2.kernel(self, mo_energy, mo_coeff, eris, with_t2)

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        self._check_supported_reference(mo_coeff, self.mo_occ, mo_energy)
        regularizer, reg_param = _resolve_regularizer(self.regularizer, self.reg_param)
        self.regularizer, self.reg_param = regularizer, reg_param
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

    def energy(self, t2, eris):
        """Contract stored regularized amplitudes with regularized energy weights."""
        nocc, nvir = t2.shape[1:3]
        regularizer, reg_param = _resolve_regularizer(self.regularizer, self.reg_param)
        eia = (numpy.asarray(eris.mo_energy[:nocc])[:, None] -
               numpy.asarray(eris.mo_energy[nocc:])[None, :])
        emp2_ss = emp2_os = 0
        for i in range(nocc):
            gi = _ovov_block(eris.ovov, i, nocc, nvir)
            t2i = t2[i]
            if regularizer == 'kappa':
                delta = -(eia[:, None, :] + eia[i][None, :, None])
                damping, _ = _regularization_factors(
                    delta, regularizer, reg_param)
                t2i = t2i * damping
            edi = numpy.einsum('jab,jab', t2i, gi) * 2
            exi = -numpy.einsum('jab,jba', t2i, gi)
            emp2_ss += edi * 0.5 + exi
            emp2_os += edi * 0.5
        emp2_ss = emp2_ss.real
        emp2_os = emp2_os.real
        return lib.tag_array(emp2_ss + emp2_os,
                             e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    def make_rdm1(self, t2=None, eris=None, ao_repr=False, with_frozen=True):
        if t2 is None and self.t2 is None:
            raise NotImplementedError('Run RegMP2 with_t2=True before building an RDM')
        return mp2.make_rdm1(self, t2, eris, ao_repr, with_frozen)

    def make_rdm2(self, t2=None, eris=None, ao_repr=False):
        if t2 is None and self.t2 is None:
            raise NotImplementedError('Run RegMP2 with_t2=True before building an RDM')
        return mp2.make_rdm2(self, t2, eris, ao_repr)

    def update_amps(self, t2, eris):
        raise NotImplementedError('RegMP2 does not support iterative/noncanonical amplitudes')

    def density_fit(self, auxbasis=None, with_df=None):
        raise NotImplementedError('density_fit() would discard RegMP2 regularization')

    def nuc_grad_method(self):
        raise NotImplementedError('Analytic gradients are not implemented for RegMP2')

    def to_gpu(self):
        raise NotImplementedError('GPU conversion is not implemented for RegMP2')
