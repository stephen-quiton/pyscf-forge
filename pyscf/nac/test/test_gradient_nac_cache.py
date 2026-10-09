#!/usr/bin/env python

import numpy as np
import pytest

from pyscf import dft
from pyscf import gto
from pyscf import scf
from pyscf.nac.gradient_nac_cache import GradientNACCacheManager, get_cache_manager
from pyscf.nac import tduks_sf


class FakeMolecule:
    def __init__(self):
        self.coords = np.zeros((1, 3))

    def atom_coords(self):
        return self.coords


class FakeMeanField:
    def __init__(self):
        self.mol = FakeMolecule()
        self.xc = 'HF'


class FakeNumInt:
    def __init__(self):
        self.block_calls = 0
        self.rho_calls = 0

    def _xc_type(self, xc_code):
        return 'LDA'

    def block_loop(self, mol, grids, nao, ao_deriv, max_memory):
        self.block_calls += 1
        yield np.ones((1, 1)), np.ones(1, dtype=bool), np.ones(1), np.zeros((1, 3))

    def eval_rho2(self, mol, ao, mo_coeff, mo_occ, mask, xctype, with_lapl):
        self.rho_calls += 1
        return np.ones(1)


class FakeTDGradient:
    def __init__(self):
        self.mol = FakeMolecule()
        self.base = type('FakeTD', (), {})()
        self.base._scf = FakeMeanField()
        self.base._scf.mol = self.mol
        self.base._scf.xc = 'LDA'
        self.base._scf.mo_coeff = (np.eye(1), np.eye(1))
        self.base._scf.mo_occ = (np.ones(1), np.ones(1))
        self.base._scf.grids = object()
        self.base._scf._numint = FakeNumInt()
        self.base.extype = 1
        self.base.collinear_samples = 0


def test_context_caches_jk_calls_and_returns_copies():
    target = FakeMeanField()
    dm = np.eye(2)
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        return np.array([1.0, 2.0])

    with GradientNACCacheManager(target) as cache:
        assert get_cache_manager(target) is cache
        first = cache.cached_jk_call(target, 'get_jk', dm, {'hermi': 1}, compute)
        first[0] = 99.0
        second = cache.cached_jk_call(target, 'get_jk', dm, {'hermi': 1}, compute)

        assert calls == 1
        np.testing.assert_allclose(second, [1.0, 2.0])
        assert cache.get_stats()['jk_misses'] == 1
        assert cache.get_stats()['jk_hits'] == 1

    assert not hasattr(target, GradientNACCacheManager._cache_attribute)
    assert cache.get_stats()['jk_cache_size'] == 0


def test_cached_jk_matches_direct_pyscf_jk():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    dm = np.array([[1.0, 0.2], [0.2, 0.8]])
    direct_j, direct_k = mf.get_jk(mol, dm, hermi=1)

    with GradientNACCacheManager(mf) as cache:
        cached_j, cached_k = cache.cached_jk_call(
            mf,
            'get_jk',
            dm,
            {'hermi': 1},
            lambda: mf.get_jk(mol, dm, hermi=1),
        )
        reused_j, reused_k = cache.cached_jk_call(
            mf,
            'get_jk',
            dm,
            {'hermi': 1},
            lambda: pytest.fail('the cached JK result was not reused'),
        )

        np.testing.assert_allclose(cached_j, direct_j)
        np.testing.assert_allclose(cached_k, direct_k)
        np.testing.assert_allclose(reused_j, direct_j)
        np.testing.assert_allclose(reused_k, direct_k)
        assert cache.get_stats()['jk_misses'] == 1
        assert cache.get_stats()['jk_hits'] == 1


def test_cache_keys_include_geometry():
    target = FakeMeanField()
    dm = np.eye(2)
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        return calls

    with GradientNACCacheManager(target) as cache:
        assert cache.cached_jk_call(target, 'get_j', dm, {}, compute) == 1
        target.mol.coords[0, 0] = 1.0
        assert cache.cached_jk_call(target, 'get_j', dm, {}, compute) == 2
        assert cache.get_stats()['jk_misses'] == 2
        assert cache.get_stats()['jk_hits'] == 0


def test_xc_blocks_are_reused_within_context():
    target = FakeTDGradient()
    ni = target.base._scf._numint

    with GradientNACCacheManager(target) as cache:
        first = cache.get_xc_blocks(target, 'LDA', 1, 2, 100, need_sc=False)
        second = cache.get_xc_blocks(target, 'LDA', 1, 2, 100, need_sc=False)

        assert first is second
        assert ni.block_calls == 1
        assert ni.rho_calls == 2
        assert cache.get_stats()['xc_block_misses'] == 1
        assert cache.get_stats()['xc_block_hits'] == 1


def test_nested_contexts_are_isolated_and_restored_after_exception():
    target = FakeMeanField()
    outer = GradientNACCacheManager(target)
    inner = GradientNACCacheManager(target)

    with outer:
        assert get_cache_manager(target) is outer
        with pytest.raises(ValueError):
            with inner:
                assert get_cache_manager(target) is inner
                inner.cached_jk_call(target, 'get_k', np.eye(2), {}, lambda: np.eye(2))
                raise ValueError('test cleanup')
        assert get_cache_manager(target) is outer
        assert inner.get_stats()['jk_cache_size'] == 0

    assert not hasattr(target, GradientNACCacheManager._cache_attribute)


def test_end_to_end_nac_is_unchanged_by_cache():
    mol = gto.M(
        atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
        basis='sto-3g',
        spin=2,
        verbose=0,
    )
    mf = dft.UKS(mol)
    mf.xc = 'B3LYP'
    mf.grids.level = 0
    mf.kernel()
    td = mf.TDDFT_SF().set(extype=1, collinear_samples=20, nstates=3).run()

    uncached = tduks_sf.NAC(td).kernel(
        state_I=1,
        state_J=2,
        use_etfs=False,
        ediff=False,
        use_cache=False,
    )
    cached = tduks_sf.NAC(td).kernel(
        state_I=1,
        state_J=2,
        use_etfs=False,
        ediff=False,
        use_cache=True,
    )

    np.testing.assert_allclose(cached, uncached, rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__])
