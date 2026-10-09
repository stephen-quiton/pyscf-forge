#!/usr/bin/env python

import copy
import hashlib
from typing import Any, Dict, Tuple

import numpy as np

from pyscf.dft import numint2c
from pyscf.sftda.numint2c_sftd import mcfun_eval_xc_adapter_sf

_MISSING = object()


def _clone_value(value):
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_value(val) for key, val in value.items()}
    return copy.deepcopy(value)


def _hash_payload(payload):
    sha1 = hashlib.sha1()

    def _update(obj):
        if isinstance(obj, np.ndarray):
            arr = np.ascontiguousarray(obj)
            sha1.update(arr.dtype.str.encode())
            sha1.update(str(arr.shape).encode())
            sha1.update(arr.tobytes())
        elif isinstance(obj, (list, tuple)):
            sha1.update(f"{type(obj).__name__}:{len(obj)}".encode())
            for item in obj:
                _update(item)
        elif isinstance(obj, dict):
            sha1.update(b"dict")
            for key, val in sorted(obj.items(), key=lambda kv: repr(kv[0])):
                sha1.update(repr(key).encode())
                _update(val)
        elif obj is None:
            sha1.update(b"None")
        else:
            sha1.update(repr(obj).encode())

    _update(payload)
    return sha1.hexdigest()


class GradientNACCacheManager:
    """Cache expensive intermediates within one gradient/NAC batch.

    Target objects are bound only for the lifetime of the context manager. All
    cached arrays and target bindings are released on exit, including when the
    calculation raises an exception.
    """

    _cache_attribute = '_gradient_nac_cache'

    def __init__(self, *targets):
        self._xc_block_cache: Dict[Tuple[Any, ...], Tuple[Dict[str, Any], ...]] = {}
        self._jk_cache: Dict[Tuple[Any, ...], Any] = {}
        self._targets = targets
        self._previous_managers = []
        self._active = False
        self._stats = {
            'xc_block_hits': 0,
            'xc_block_misses': 0,
            'jk_hits': 0,
            'jk_misses': 0,
        }

    def __enter__(self):
        if self._active:
            raise RuntimeError('GradientNACCacheManager cannot be re-entered')
        self.clear()
        self._active = True
        self._previous_managers = []
        for target in self._targets:
            previous = getattr(target, self._cache_attribute, _MISSING)
            self._previous_managers.append((target, previous))
            setattr(target, self._cache_attribute, self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for target, previous in reversed(self._previous_managers):
            if previous is _MISSING:
                delattr(target, self._cache_attribute)
            else:
                setattr(target, self._cache_attribute, previous)
        self._previous_managers = []
        self.clear()
        self._active = False

    def _geometry_key(self, mf, mol, td_obj=None):
        return (
            id(mf),
            mol.atom_coords().tobytes(),
            getattr(mf, 'xc', None),
            getattr(td_obj, 'extype', None),
            getattr(td_obj, 'collinear_samples', None),
        )

    def _call_scope_key(self, target):
        if hasattr(target, 'base') and hasattr(target.base, '_scf'):
            mf = target.base._scf
            mol = target.mol
            td_obj = target.base
        else:
            mf = target
            mol = target.mol
            td_obj = None
        return self._geometry_key(mf, mol, td_obj)

    def get_xc_blocks(self, td_grad, xc_code, ao_deriv, deriv, max_memory, need_sc):
        mf = td_grad.base._scf
        mol = td_grad.mol
        ni = mf._numint
        xctype = ni._xc_type(xc_code)
        need_sf = td_grad.base.collinear_samples > 0 and xctype != 'HF'
        key = self._geometry_key(mf, mol, td_grad.base) + (
            xc_code,
            ao_deriv,
            deriv,
            int(max_memory),
            bool(need_sc),
            bool(need_sf),
        )
        if key in self._xc_block_cache:
            self._stats['xc_block_hits'] += 1
            return self._xc_block_cache[key]

        self._stats['xc_block_misses'] += 1
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        nao = mo_coeff[0].shape[0]
        blocks = []

        eval_xc_eff = None
        if need_sf:
            nimc = numint2c.NumInt2C()
            nimc.collinear = 'mcol'
            nimc.collinear_samples = td_grad.base.collinear_samples
            eval_xc_eff = mcfun_eval_xc_adapter_sf(nimc, xc_code)

        for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
            ao0 = ao[0] if xctype == 'LDA' else ao
            rho = (
                ni.eval_rho2(mol, ao0, mo_coeff[0], mo_occ[0], mask, xctype, with_lapl=False),
                ni.eval_rho2(mol, ao0, mo_coeff[1], mo_occ[1], mask, xctype, with_lapl=False),
            )
            data = {'rho': rho}

            if need_sf:
                rho_z = np.array([rho[0] + rho[1], rho[0] - rho[1]])
                sf_eval = eval_xc_eff(xc_code, rho_z, deriv, xctype=xctype)
                data['fxc_sf'] = sf_eval[2]
                if deriv >= 3 and sf_eval[3] is not None:
                    data['kxc_sf'] = np.stack(
                        (sf_eval[3][:, :, 0] + sf_eval[3][:, :, 1], sf_eval[3][:, :, 0] - sf_eval[3][:, :, 1]),
                        axis=2,
                    )
                else:
                    data['kxc_sf'] = None
            else:
                data['fxc_sf'] = None
                data['kxc_sf'] = None

            if need_sc:
                vxc, fxc, _ = ni.eval_xc_eff(xc_code, rho, deriv=2, spin=1)[1:]
                data['vxc'] = vxc
                data['fxc'] = fxc
            else:
                data['vxc'] = None
                data['fxc'] = None

            blocks.append(data)

        self._xc_block_cache[key] = tuple(blocks)
        return self._xc_block_cache[key]

    def cached_jk_call(self, target, method_name, dm, kwargs, compute_fn):
        key = self._call_scope_key(target) + (
            method_name,
            _hash_payload(dm),
            _hash_payload(kwargs),
        )
        if key in self._jk_cache:
            self._stats['jk_hits'] += 1
            return _clone_value(self._jk_cache[key])

        self._stats['jk_misses'] += 1
        result = compute_fn()
        self._jk_cache[key] = _clone_value(result)
        return _clone_value(result)

    def clear(self):
        self._xc_block_cache.clear()
        self._jk_cache.clear()
        for key in self._stats:
            self._stats[key] = 0

    def get_stats(self):
        stats = dict(self._stats)
        stats['xc_block_cache_size'] = len(self._xc_block_cache)
        stats['jk_cache_size'] = len(self._jk_cache)
        return stats

    def print_stats(self):
        stats = self.get_stats()
        print('Gradient-NAC cache stats:')
        print(
            f"  xc blocks hits={stats['xc_block_hits']} misses={stats['xc_block_misses']} "
            f"size={stats['xc_block_cache_size']}"
        )
        print(f"  jk calls  hits={stats['jk_hits']} misses={stats['jk_misses']} size={stats['jk_cache_size']}")


def get_cache_manager(target, create=True):
    """Return a bound cache, optionally creating an unbound local cache."""
    manager = getattr(target, GradientNACCacheManager._cache_attribute, None)
    if manager is None and create:
        manager = GradientNACCacheManager()
    return manager
