#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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

"""Regularized restricted k-point MP2.

The kappa, sigma, and sigma-squared damping factors are those of J. Lee and
M. Head-Gordon, J. Chem. Theory Comput. 14, 5203 (2018),
doi:10.1021/acs.jctc.8b00731, and J. Shee et al., J. Phys. Chem. Lett. 12,
12084 (2021), doi:10.1021/acs.jpclett.1c03468.  The traversal retains the
periodic KMP2 GDF and generic AO2MO paths and regularizes only active
occupied/virtual blocks.
"""

import numpy

from pyscf import lib
from pyscf import __config__
from pyscf.lib import logger
from pyscf.mp.regmp2 import _regularization_factors, _resolve_regularizer
from pyscf.pbc import scf
from pyscf.pbc.df import df
from pyscf.pbc.lib import kpts as libkpts
from pyscf.pbc.mp import kmp2


WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)


def _same_kpoint_data(x, y):
    if x is y:
        return True
    if isinstance(x, (list, tuple)) or isinstance(y, (list, tuple)):
        if not isinstance(x, (list, tuple)) or not isinstance(y, (list, tuple)):
            return False
        return len(x) == len(y) and all(_same_kpoint_data(a, b) for a, b in zip(x, y))
    try:
        return numpy.shape(x) == numpy.shape(y) and numpy.allclose(x, y)
    except (TypeError, ValueError):
        return False


def _copy_kpoint_data(x):
    if isinstance(x, (list, tuple)):
        return [numpy.array(value, copy=True) for value in x]
    return numpy.array(x, copy=True)


def _active_block(array, idx):
    return array[numpy.ix_(*idx)]


def _is_krhf_reference(mf):
    return (isinstance(mf, scf.khf.KRHF) and
            not mf.istype('KROHF') and not mf.istype('KohnShamDFT'))


def kernel(mp, mo_energy, mo_coeff, verbose=logger.NOTE, with_t2=WITH_T2):
    """Compute regularized k-point MP2 energy and first-order amplitudes."""
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mp, verbose)

    mp.dump_flags()
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = mp.nkpts

    with_df_ints = mp.with_df_ints and isinstance(mp._scf.with_df, df.GDF)

    mem_avail = mp.max_memory - lib.current_memory()[0]
    mem_usage = (nkpts * (nocc * nvir)**2) * 16 / 1e6
    if with_df_ints:
        mydf = mp._scf.with_df
        if mydf.auxcell is None:
            naux = mydf.get_naoaux()
        else:
            naux = mydf.auxcell.nao_nr()
        mem_usage += (nkpts**2 * naux * nocc * nvir) * 16 / 1e6
    if with_t2:
        mem_usage += (nkpts**3 * (nocc * nvir)**2) * 16 / 1e6
    if mem_usage > mem_avail:
        raise MemoryError('Insufficient memory! MP2 memory usage %d MB (currently available %d MB)'
                          % (mem_usage, mem_avail))

    fao2mo = mp._scf.with_df.ao2mo
    kconserv = mp.khelper.kconserv
    oovv_ij = numpy.zeros((nkpts, nocc, nocc, nvir, nvir),
                          dtype=numpy.result_type(mo_coeff[0].dtype, complex))

    mo_e_o = [numpy.asarray(mo_energy[k][:nocc]) for k in range(nkpts)]
    mo_e_v = [numpy.asarray(mo_energy[k][nocc:]) for k in range(nkpts)]
    nonzero_opadding, nonzero_vpadding = kmp2.padding_k_idx(mp, kind='split')

    if with_t2:
        t2 = numpy.zeros((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir),
                         dtype=complex)
    else:
        t2 = None

    if with_df_ints:
        Lov = kmp2._init_mp_df_eris(mp)

    regularizer, reg_param = _resolve_regularizer(mp.regularizer, mp.reg_param)
    mp.regularizer, mp.reg_param = regularizer, reg_param

    emp2_ss = emp2_os = 0.0
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]
                if with_df_ints:
                    oovv_ij[ka] = (1. / nkpts) * lib.einsum(
                        'Lia,Ljb->iajb', Lov[ki, ka], Lov[kj, kb]
                    ).transpose(0, 2, 1, 3)
                else:
                    orbo_i = mo_coeff[ki][:, :nocc]
                    orbo_j = mo_coeff[kj][:, :nocc]
                    orbv_a = mo_coeff[ka][:, nocc:]
                    orbv_b = mo_coeff[kb][:, nocc:]
                    oovv_ij[ka] = fao2mo(
                        (orbo_i, orbv_a, orbo_j, orbv_b),
                        (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]),
                        compact=False,
                    ).reshape(nocc, nvir, nocc, nvir).transpose(0, 2, 1, 3) / nkpts

            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]
                oi = nonzero_opadding[ki]
                oj = nonzero_opadding[kj]
                va = nonzero_vpadding[ka]
                vb = nonzero_vpadding[kb]
                active = (oi, oj, va, vb)

                # Work only on physical orbitals.  In particular, LARGE_DENOM
                # padding is never sent through a regularizer exponential.
                delta_ia = mo_e_v[ka][va][None, :] - mo_e_o[ki][oi][:, None]
                delta_jb = mo_e_v[kb][vb][None, :] - mo_e_o[kj][oj][:, None]
                delta = (delta_ia[:, None, :, None] +
                         delta_jb[None, :, None, :])
                damping, damping_over_gap = _regularization_factors(
                    delta, regularizer, reg_param)

                eri_active = _active_block(oovv_ij[ka], active)
                t2_active = -eri_active.conj() * damping_over_gap
                t2_ijab = numpy.zeros((nocc, nocc, nvir, nvir), dtype=complex)
                t2_ijab[numpy.ix_(*active)] = t2_active

                if regularizer == 'kappa':
                    t2_energy = numpy.zeros_like(t2_ijab)
                    t2_energy[numpy.ix_(*active)] = t2_active * damping
                else:
                    t2_energy = t2_ijab

                if with_t2:
                    t2[ki, kj, ka] = t2_ijab
                edi = lib.einsum('ijab,ijab', t2_energy, oovv_ij[ka]).real * 2
                exi = -lib.einsum('ijab,ijba', t2_energy, oovv_ij[kb]).real
                emp2_ss += edi * 0.5 + exi
                emp2_os += edi * 0.5

    log.timer('KRegMP2', *cput0)
    emp2_ss /= nkpts
    emp2_os /= nkpts
    emp2 = lib.tag_array(emp2_ss + emp2_os,
                         e_corr_ss=emp2_ss, e_corr_os=emp2_os)
    return emp2, t2


class KRegMP2(kmp2.KMP2):
    """Regularized MP2 for a converged canonical periodic KRHF reference.

    Args:
        mf: A converged :class:`pyscf.pbc.scf.khf.KRHF` object.  Both GDF and
            generic periodic AO2MO backends are supported.
        frozen, mo_coeff, mo_occ: The usual PySCF KMP2 inputs.
        regularizer: ``'kappa'``, ``'sigma'``, or ``'sigma2'``.  ``'sigma^2'``
            is accepted as an alias.
        reg_param: A finite nonnegative parameter.  If omitted, the defaults
            are 1.1, 0.7, and 0.4 for kappa, sigma, and sigma2.
    """

    _keys = kmp2.KMP2._keys | {'regularizer', 'reg_param'}

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None,
                 regularizer='kappa', reg_param=None):
        if not _is_krhf_reference(mf):
            raise NotImplementedError('KRegMP2 supports canonical periodic KRHF references only')
        regularizer, reg_param = _resolve_regularizer(regularizer, reg_param)

        canonical_coeff = mf.mo_coeff
        canonical_occ = mf.mo_occ
        if isinstance(mf.kpts, libkpts.KPoints):
            canonical_coeff = mf.kpts.transform_mo_coeff(canonical_coeff)
            canonical_occ = mf.kpts.transform_mo_occ(canonical_occ)

        kmp2.KMP2.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self._canonical_mo_coeff = _copy_kpoint_data(canonical_coeff)
        self._canonical_mo_occ = _copy_kpoint_data(canonical_occ)
        self._canonical_mo_energy = _copy_kpoint_data(self.mo_energy)
        self.regularizer = regularizer
        self.reg_param = reg_param

    def _check_supported_reference(self, mo_energy=None, mo_coeff=None):
        if not _is_krhf_reference(self._scf):
            raise NotImplementedError('KRegMP2 supports canonical periodic KRHF references only')
        if not getattr(self._scf, 'converged', False):
            raise NotImplementedError(
                'KRegMP2 does not support noncanonical or iterative MP2 references')

        if not _same_kpoint_data(self.mo_coeff if mo_coeff is None else mo_coeff,
                                 self._canonical_mo_coeff):
            raise NotImplementedError('KRegMP2 supports canonical KRHF orbitals only')
        if not _same_kpoint_data(self.mo_occ, self._canonical_mo_occ):
            raise NotImplementedError('KRegMP2 supports canonical KRHF occupations only')
        if not numpy.all((numpy.isclose(numpy.asarray(self.mo_occ), 0) |
                          numpy.isclose(numpy.asarray(self.mo_occ), 2))):
            raise NotImplementedError('KRegMP2 requires integer restricted-KRHF occupations')
        if mo_energy is not None and not _same_kpoint_data(mo_energy, self._canonical_mo_energy):
            raise NotImplementedError('KRegMP2 supports canonical KRHF orbital energies only')

    def dump_flags(self):
        regularizer, reg_param = _resolve_regularizer(self.regularizer, self.reg_param)
        self.regularizer, self.reg_param = regularizer, reg_param
        kmp2.KMP2.dump_flags(self)
        logger.info(self, 'regularizer = %s, reg_param = %g',
                    regularizer, reg_param)
        return self

    def kernel(self, mo_energy=None, mo_coeff=None, with_t2=WITH_T2):
        self._check_supported_reference(mo_energy, mo_coeff)
        if mo_energy is None:
            mo_energy = self.mo_energy
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_energy is None or mo_coeff is None:
            log = logger.Logger(self.stdout, self.verbose)
            log.warn('mo_coeff, mo_energy are not given.\n'
                     'You may need to call mf.kernel() to generate them.')
            raise RuntimeError

        self.e_hf = self.get_e_hf(mo_coeff=mo_coeff)
        mo_coeff, mo_energy = kmp2._add_padding(self, mo_coeff, mo_energy)
        self.e_corr, self.t2 = kernel(self, mo_energy, mo_coeff,
                                      verbose=self.verbose, with_t2=with_t2)

        self.e_corr_ss = getattr(self.e_corr, 'e_corr_ss', 0)
        self.e_corr_os = getattr(self.e_corr, 'e_corr_os', 0)
        self.e_corr = float(self.e_corr)
        self._finalize()
        return self.e_corr, self.t2

    def update_amps(self, t2, eris):
        raise NotImplementedError('KRegMP2 does not support iterative/noncanonical amplitudes')

    def init_amps(self, *args, **kwargs):
        raise NotImplementedError('KRegMP2 amplitudes are initialized by its k-point traversal')

    def energy(self, t2, eris):
        raise NotImplementedError(
            'KRegMP2 energy() is only defined by its regularized k-point traversal')

    def density_fit(self, *args, **kwargs):
        raise NotImplementedError('density_fit() would discard KRegMP2 regularization')

    def nuc_grad_method(self):
        raise NotImplementedError('Analytic gradients are not implemented for KRegMP2')

    def to_gpu(self):
        raise NotImplementedError('GPU conversion is not implemented for KRegMP2')
