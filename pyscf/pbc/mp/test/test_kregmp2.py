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

import numpy
import pytest

from pyscf.pbc import dft, gto, scf
from pyscf.pbc.mp import kmp2
from pyscf.pbc.mp.kregmp2 import KRegMP2


@pytest.fixture(scope='module')
def krhf():
    cell = gto.Cell(
        atom='H 0 0 0; H 0 0 1.4',
        a=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        basis='gth-szv', pseudo='gth-pade', unit='B', verbose=0,
    ).build()
    kpts = cell.make_kpts([1, 1, 2])
    return scf.KRHF(cell, kpts, exxdiv=None).run()


@pytest.fixture(scope='module')
def krhf_df(krhf):
    return scf.KRHF(krhf.cell, krhf.kpts, exxdiv=None).density_fit().run()


@pytest.fixture(scope='module')
def co_krhf_df():
    # A large-vacuum representation of the molecular CO benchmark.  This is
    # deliberately a molecular gas-limit check of the periodic code, not a
    # claim to reproduce a published crystalline-solid calculation.
    cell = gto.Cell(
        atom='C 0 0 0; O 0 0 1.1282',
        a=numpy.eye(3) * 16,
        basis='def2-tzvpp', unit='A', verbose=0,
    ).build()
    kpts = cell.make_kpts([1, 1, 1])
    return scf.KRHF(cell, kpts).density_fit().run()


def _direct_regularization_factors(delta, regularizer, reg_param):
    delta = numpy.asarray(delta, dtype=float)
    with numpy.errstate(over='ignore', under='ignore', divide='ignore'):
        if regularizer == 'kappa':
            x = reg_param * delta
            one = -numpy.expm1(-x)
            contraction_factor = one
            zero_gap_limit = reg_param
        elif regularizer == 'sigma':
            x = reg_param * delta
            one = -numpy.expm1(-x)
            contraction_factor = one
            zero_gap_limit = reg_param
        else:
            x = reg_param * delta * delta
            one = -numpy.expm1(-x)
            contraction_factor = one
            zero_gap_limit = 0.
        amplitude_factor = numpy.divide(
            one, delta, out=numpy.zeros_like(delta), where=delta != 0)
        amplitude_factor = numpy.where(delta == 0, zero_gap_limit,
                                       amplitude_factor)
    return contraction_factor, amplitude_factor


def _expected_periodic_result(pt):
    nocc = pt.nocc
    nvir = pt.nmo - nocc
    nkpts = pt.nkpts
    kconserv = pt.khelper.kconserv
    mo_coeff, mo_energy = kmp2._add_padding(
        pt, pt.mo_coeff, pt.mo_energy)
    occ_idx, vir_idx = kmp2.padding_k_idx(pt, kind='split')
    dtype = numpy.result_type(mo_coeff[0].dtype, complex)
    eri = numpy.zeros((nkpts, nocc, nocc, nvir, nvir), dtype=dtype)
    expected_t2 = numpy.zeros(
        (nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=complex)
    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]
    fao2mo = pt._scf.with_df.ao2mo
    emp2_ss = emp2_os = 0.

    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]
                eri[ka] = fao2mo(
                    (mo_coeff[ki][:, :nocc], mo_coeff[ka][:, nocc:],
                     mo_coeff[kj][:, :nocc], mo_coeff[kb][:, nocc:]),
                    (pt.kpts[ki], pt.kpts[ka], pt.kpts[kj], pt.kpts[kb]),
                    compact=False,
                ).reshape(nocc, nvir, nocc, nvir).transpose(0, 2, 1, 3) / nkpts

            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]
                active = (occ_idx[ki], occ_idx[kj],
                          vir_idx[ka], vir_idx[kb])
                delta_ia = (mo_e_v[ka][vir_idx[ka]][None, :] -
                            mo_e_o[ki][occ_idx[ki]][:, None])
                delta_jb = (mo_e_v[kb][vir_idx[kb]][None, :] -
                            mo_e_o[kj][occ_idx[kj]][:, None])
                delta = (delta_ia[:, None, :, None] +
                         delta_jb[None, :, None, :])
                damping, amplitude_factor = _direct_regularization_factors(
                    delta, pt.regularizer, pt.reg_param)
                t2_active = -eri[ka][numpy.ix_(*active)].conj() * amplitude_factor
                t2_ijab = numpy.zeros((nocc, nocc, nvir, nvir), dtype=complex)
                t2_ijab[numpy.ix_(*active)] = t2_active
                expected_t2[ki, kj, ka] = t2_ijab

                if pt.regularizer == 'kappa':
                    t2_energy = numpy.zeros_like(t2_ijab)
                    t2_energy[numpy.ix_(*active)] = t2_active * damping
                else:
                    t2_energy = t2_ijab
                ed = numpy.einsum('ijab,ijab', t2_energy, eri[ka]).real * 2
                ex = -numpy.einsum('ijab,ijba', t2_energy, eri[kb]).real
                emp2_ss += ed * .5 + ex
                emp2_os += ed * .5

    return (emp2_ss / nkpts + emp2_os / nkpts,
            emp2_ss / nkpts, emp2_os / nkpts, expected_t2)


@pytest.mark.parametrize('regularizer, default', [
    ('kappa', 1.1), ('sigma', .7), ('sigma2', .4), ('sigma^2', .4),
])
def test_periodic_variants_and_direct_formula(krhf_df, regularizer, default):
    pt = KRegMP2(krhf_df, regularizer=regularizer)
    pt.kernel(with_t2=True)
    assert pt.reg_param == default
    expected = _expected_periodic_result(pt)
    numpy.testing.assert_allclose(pt.t2, expected[3], atol=2e-11)
    numpy.testing.assert_allclose(pt.e_corr, expected[0], atol=2e-11)
    numpy.testing.assert_allclose(pt.e_corr_ss, expected[1], atol=2e-11)
    numpy.testing.assert_allclose(pt.e_corr_os, expected[2], atol=2e-11)
    numpy.testing.assert_allclose(pt.e_corr, pt.e_corr_ss + pt.e_corr_os, atol=2e-12)
    assert numpy.iscomplexobj(pt.t2)


@pytest.mark.parametrize('reference', ['krhf', 'krhf_df'])
def test_periodic_with_t2_false_and_mp2_limit(request, reference):
    mf = request.getfixturevalue(reference)
    plain_obj = kmp2.KMP2(mf)
    plain_obj.kernel(with_t2=False)
    plain = plain_obj
    for regularizer in ('kappa', 'sigma', 'sigma2'):
        with_t2 = KRegMP2(mf, regularizer=regularizer, reg_param=.23)
        with_t2.kernel(with_t2=True)
        without_t2 = KRegMP2(mf, regularizer=regularizer, reg_param=.23)
        without_t2.kernel(with_t2=False)
        assert without_t2.t2 is None
        numpy.testing.assert_allclose(without_t2.e_corr, with_t2.e_corr, atol=2e-12)

        large = KRegMP2(mf, regularizer=regularizer, reg_param=1e6)
        large.kernel(with_t2=False)
        numpy.testing.assert_allclose(large.e_corr, plain.e_corr, atol=2e-11)


def test_periodic_generic_direct_formula(krhf):
    pt = KRegMP2(krhf, regularizer='sigma2', reg_param=.4)
    expected = _expected_periodic_result(pt)
    pt.kernel(with_t2=True)
    numpy.testing.assert_allclose(pt.t2, expected[3], atol=2e-11)
    numpy.testing.assert_allclose(pt.e_corr, expected[0], atol=2e-11)
    numpy.testing.assert_allclose(pt.e_corr_ss, expected[1], atol=2e-11)
    numpy.testing.assert_allclose(pt.e_corr_os, expected[2], atol=2e-11)


def test_periodic_gdf_path(krhf_df):
    pt = KRegMP2(krhf_df, regularizer='sigma2', reg_param=.4)
    pt.kernel(with_t2=False)
    assert krhf_df.with_df.__class__.__name__ == 'GDF'
    assert numpy.isfinite(pt.e_corr)
    assert pt.e_corr == pytest.approx(pt.e_corr_ss + pt.e_corr_os)


def test_large_vacuum_co_gas_limit_against_literature(co_krhf_df):
    # Correlation energies derived from Table S8 of Shee et al., JPCL 12,
    # 12084 (2021), doi:10.1021/acs.jpclett.1c03468; see the molecular test.
    # The 1 mEh tolerance additionally covers the finite 16 Angstrom cell.
    literature = {
        ('kappa', 1.1): -0.3259899226,
        ('sigma2', .4): -0.3294597819,
    }
    for (regularizer, reg_param), expected in literature.items():
        pt = KRegMP2(
            co_krhf_df, frozen=2, regularizer=regularizer,
            reg_param=reg_param,
        ).run(with_t2=False)
        numpy.testing.assert_allclose(pt.e_corr, expected, atol=1e-3, rtol=0)


def test_periodic_frozen_padding_is_exactly_zero(krhf_df):
    # Freeze a virtual only at one k point.  KMP2 pads that k point to the
    # dense virtual dimension; no exponential should see the padding value.
    frozen = [[1], []]
    pt = KRegMP2(krhf_df, frozen=frozen, regularizer='kappa', reg_param=1.1).run()
    assert pt.t2 is not None
    occ_idx, vir_idx = kmp2.padding_k_idx(pt, kind='split')
    padded_blocks = 0
    for ki in range(pt.nkpts):
        for kj in range(pt.nkpts):
            for ka in range(pt.nkpts):
                kb = pt.khelper.kconserv[ki, ka, kj]
                active = (occ_idx[ki], occ_idx[kj], vir_idx[ka], vir_idx[kb])
                if any(len(index) != size for index, size in
                       zip(active, (pt.nocc, pt.nocc, pt.nmo - pt.nocc,
                                   pt.nmo - pt.nocc))):
                    padded_blocks += 1
                    block = pt.t2[ki, kj, ka]
                    active_mask = numpy.zeros(block.shape, dtype=bool)
                    active_mask[numpy.ix_(*active)] = True
                    assert numpy.all(block[~active_mask] == 0)
    assert padded_blocks
    assert numpy.isfinite(pt.t2).all()


def test_periodic_canonical_reference_check_does_not_rebuild_fock(krhf, monkeypatch):
    def fail_get_fock(*args, **kwargs):
        raise AssertionError('canonical reference check rebuilt the Fock matrix')

    monkeypatch.setattr(krhf, 'get_fock', fail_get_fock)
    KRegMP2(krhf).kernel(with_t2=False)


def test_periodic_reject_invalid_inputs(krhf):
    with pytest.raises(ValueError, match='Unknown regularizer'):
        KRegMP2(krhf, regularizer='bad')
    for value in (-1, numpy.inf, numpy.nan, True, '0.5'):
        with pytest.raises(ValueError, match='reg_param'):
            KRegMP2(krhf, reg_param=value)

    changed = KRegMP2(krhf)
    changed.reg_param = -1
    with pytest.raises(ValueError, match='reg_param'):
        changed.kernel(with_t2=False)

    for mf in (scf.KUHF(krhf.cell, krhf.kpts),
               scf.KROHF(krhf.cell, krhf.kpts),
               dft.KRKS(krhf.cell, krhf.kpts)):
        with pytest.raises(NotImplementedError, match='KRHF'):
            KRegMP2(mf)
    with pytest.raises(NotImplementedError, match='noncanonical'):
        KRegMP2(scf.KRHF(krhf.cell, krhf.kpts, exxdiv=None)).kernel(with_t2=False)
    with pytest.raises(NotImplementedError, match='gradients'):
        KRegMP2(krhf).nuc_grad_method()
