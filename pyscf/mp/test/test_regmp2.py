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

from types import SimpleNamespace

import numpy
import pytest

from pyscf import dft, gto, mp, scf
from pyscf.mp.regmp2 import RegMP2, _regularization_factors, kernel


@pytest.fixture(scope='module')
def rhf():
    mol = gto.M(atom='Be 0 0 0', basis='sto-3g', verbose=0)
    return scf.RHF(mol).run()


@pytest.fixture(scope='module')
def co_rhf():
    # Geometry from Hait and Head-Gordon, JCTC 14, 1969 (2018),
    # doi:10.1021/acs.jctc.7b01252.  It is the CO geometry used for the
    # regularized-MP2 energies in Table S8 of doi:10.1021/acs.jpclett.1c03468.
    mol = gto.M(
        atom='C 0 0 0; O 0 0 1.1282',
        basis='def2-tzvpp', unit='A', verbose=0,
    )
    return scf.RHF(mol).run()


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


def _direct_molecular_result(pt, eris, regularizer, reg_param):
    nocc = pt.nocc
    nvir = pt.nmo - nocc
    eri = numpy.asarray(eris.ovov).reshape(nocc, nvir, nocc, nvir)
    mo_energy = numpy.asarray(eris.mo_energy)
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
    delta = -(eia[:, None, :, None] + eia[None, :, None, :])
    damping, damping_over_gap = _direct_regularization_factors(
        delta, regularizer, reg_param)
    t2 = -eri.transpose(0, 2, 1, 3).conj() * damping_over_gap
    t2_energy = t2 * damping if regularizer == 'kappa' else t2
    ed = numpy.einsum('ijab,iajb', t2_energy, eri) * 2
    ex = -numpy.einsum('ijab,ibja', t2_energy, eri)
    ss = (ed * .5 + ex).real
    os = ed.real * .5
    return ss + os, ss, os, t2


@pytest.mark.parametrize('regularizer, default', [
    ('kappa', 1.1), ('sigma', .7), ('sigma2', .4), ('sigma^2', .4),
])
def test_selectors_defaults_and_formula(rhf, regularizer, default):
    pt = RegMP2(rhf, regularizer=regularizer)
    assert pt.reg_param == default
    eris = pt.ao2mo()
    expected = _direct_molecular_result(pt, eris, pt.regularizer, default)
    pt.kernel(with_t2=True)
    numpy.testing.assert_allclose(pt.e_corr, expected[0], atol=1e-12)
    numpy.testing.assert_allclose(pt.e_corr_ss, expected[1], atol=1e-12)
    numpy.testing.assert_allclose(pt.e_corr_os, expected[2], atol=1e-12)
    numpy.testing.assert_allclose(pt.t2, expected[3], atol=1e-12)
    numpy.testing.assert_allclose(pt.energy(pt.t2, eris), expected[0], atol=1e-12)
    numpy.testing.assert_allclose(pt.e_corr, pt.e_corr_ss + pt.e_corr_os, atol=1e-12)
    numpy.testing.assert_allclose(pt.e_tot, pt.e_hf + pt.e_corr, atol=1e-12)


def test_override_zero_and_large_parameter(rhf):
    ordinary = mp.MP2(rhf).run(with_t2=True)
    for regularizer in ('kappa', 'sigma', 'sigma2'):
        zero = RegMP2(rhf, regularizer=regularizer, reg_param=0).run(with_t2=True)
        assert zero.e_corr == 0
        assert zero.t2 is not None
        numpy.testing.assert_allclose(zero.t2, 0)
        numpy.testing.assert_allclose(zero.e_tot, zero.e_hf)

        large = RegMP2(rhf, regularizer=regularizer, reg_param=1e6).run(with_t2=True)
        numpy.testing.assert_allclose(large.e_corr, ordinary.e_corr, atol=1e-12)
        numpy.testing.assert_allclose(large.t2, ordinary.t2, atol=1e-12)


def test_co_correlation_energy_against_literature(co_rhf):
    # Table S8 of Shee et al., J. Phys. Chem. Lett. 12, 12084 (2021),
    # doi:10.1021/acs.jpclett.1c03468, reports Q-Chem RI/def2-TZVPP total
    # energies with two frozen orbitals.  Subtracting its HF energy
    # (-112.7846172412 Eh) gives these correlation energies.  The tolerance
    # accounts for the different programs and RI versus four-center integrals.
    literature = {
        ('kappa', 1.1): -0.3259899226,
        ('sigma2', .4): -0.3294597819,
    }
    for (regularizer, reg_param), expected in literature.items():
        pt = RegMP2(
            co_rhf, frozen=2, regularizer=regularizer, reg_param=reg_param,
        ).run(with_t2=False)
        numpy.testing.assert_allclose(pt.e_corr, expected, atol=1e-3, rtol=0)


def test_with_t2_and_frozen_consistency(rhf):
    for with_t2 in (True, False):
        pt = RegMP2(rhf, frozen=1, regularizer='sigma', reg_param=.31)
        pt.kernel(with_t2=with_t2)
        assert (pt.t2 is not None) is with_t2
        assert pt.e_corr_ss + pt.e_corr_os == pytest.approx(pt.e_corr)

    frozen_plain = mp.MP2(rhf, frozen=1).run(with_t2=False)
    frozen_reg = RegMP2(rhf, frozen=1, reg_param=1e6).run(with_t2=False)
    assert frozen_reg.e_corr == pytest.approx(frozen_plain.e_corr, abs=1e-12)


def test_rdm_uses_stored_regularized_amplitudes(rhf):
    pt = RegMP2(rhf, regularizer='sigma2').run(with_t2=True)
    assert pt.make_rdm1().shape == (pt.nmo, pt.nmo)
    assert pt.make_rdm2().shape == (pt.nmo, pt.nmo, pt.nmo, pt.nmo)
    no_t2 = RegMP2(rhf, regularizer='sigma2')
    no_t2.kernel(with_t2=False)
    with pytest.raises(NotImplementedError, match='with_t2=True'):
        no_t2.make_rdm1()


def test_already_density_fitted_rhf_uses_generic_ao2mo(rhf):
    mf_df = scf.RHF(rhf.mol).density_fit().run()
    pt = RegMP2(mf_df, regularizer='kappa', reg_param=.25).run(with_t2=False)
    assert numpy.isfinite(pt.e_corr)
    with pytest.raises(NotImplementedError, match='discard'):
        RegMP2(mf_df).density_fit()


def test_canonical_reference_check_does_not_rebuild_fock(rhf, monkeypatch):
    def fail_get_fock(*args, **kwargs):
        raise AssertionError('canonical reference check rebuilt the Fock matrix')

    monkeypatch.setattr(rhf, 'get_fock', fail_get_fock)
    RegMP2(rhf).kernel(with_t2=False)


def test_rejects_canonical_permutation(rhf):
    with pytest.raises(NotImplementedError, match='canonical RHF orbitals'):
        RegMP2(rhf, mo_coeff=rhf.mo_coeff[:, ::-1]).kernel(with_t2=False)


def test_zero_gap_limits_and_energy_behavior():
    eris = SimpleNamespace(
        mo_energy=numpy.zeros(2),
        ovov=numpy.ones((1, 1, 1, 1)),
    )
    fake = SimpleNamespace(nocc=1, nmo=2, frozen=None)
    for regularizer, param in [('kappa', 1.1), ('sigma', .7), ('sigma2', .4)]:
        fake.regularizer = regularizer
        fake.reg_param = param
        e, t2 = kernel(fake, eris=eris, with_t2=True)
        damping, damping_over_gap = _regularization_factors(
            numpy.zeros(1), regularizer, param)
        assert damping[0] == 0
        assert numpy.isfinite(damping_over_gap[0])
        assert numpy.isfinite(t2).all()
        if regularizer == 'sigma2':
            assert t2[0, 0, 0, 0] == 0
            assert e == 0
        elif regularizer == 'kappa':
            assert t2[0, 0, 0, 0] == pytest.approx(-param)
            assert e == 0
        else:
            assert t2[0, 0, 0, 0] == pytest.approx(-param)
            assert numpy.isfinite(e) and e != 0

    for regularizer in ('kappa', 'sigma', 'sigma2'):
        damping, ratio = _regularization_factors(
            numpy.array([0., 1e-300, 1., 1e300]), regularizer, 1.0)
        assert numpy.isfinite(damping).all()
        assert numpy.isfinite(ratio).all()


def test_reject_invalid_selector_parameter_reference_and_gradients(rhf):
    with pytest.raises(ValueError, match='Unknown regularizer'):
        RegMP2(rhf, regularizer='ordinary')
    for value in (-1, numpy.inf, numpy.nan, True, '0.5'):
        with pytest.raises(ValueError, match='reg_param'):
            RegMP2(rhf, reg_param=value)

    changed = RegMP2(rhf)
    changed.reg_param = -1
    with pytest.raises(ValueError, match='reg_param'):
        changed.kernel(with_t2=False)

    mol = rhf.mol
    for mf in (scf.UHF(mol), scf.ROHF(mol), dft.RKS(mol)):
        with pytest.raises(NotImplementedError, match='restricted'):
            RegMP2(mf)

    noncanonical = scf.RHF(mol)
    noncanonical.mo_coeff = rhf.mo_coeff
    noncanonical.mo_occ = rhf.mo_occ
    noncanonical.mo_energy = rhf.mo_energy
    noncanonical.converged = False
    with pytest.raises(NotImplementedError, match='noncanonical'):
        RegMP2(noncanonical).kernel(with_t2=False)
    with pytest.raises(NotImplementedError, match='gradients'):
        RegMP2(rhf).nuc_grad_method()
