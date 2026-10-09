#!/usr/bin/env python

import time
from typing import Dict, List, Optional, Tuple

from pyscf.lib import logger
from pyscf.grad import tduks_sf as tduks_sf_grad
from pyscf.nac import tduks_sf as tduks_sf_nac
from pyscf.nac.gradient_nac_cache import GradientNACCacheManager


def _apply_cphf_options(obj, cphf_options):
    if not cphf_options:
        return
    if 'max_cycle' in cphf_options and hasattr(obj, 'cphf_max_cycle'):
        obj.cphf_max_cycle = cphf_options['max_cycle']
    if 'conv_tol' in cphf_options and hasattr(obj, 'cphf_conv_tol'):
        obj.cphf_conv_tol = cphf_options['conv_tol']


def _resolve_state_id(state_id, state_map):
    if state_map is None:
        return state_id
    return state_map.get(state_id, state_id)


def _default_nac_pairs(states):
    return [(states[i], states[j]) for i in range(len(states) - 1) for j in range(i + 1, len(states))]


def compute_gradients_and_nacs(
    td_obj,
    states: List[int],
    nac_pairs: Optional[List[Tuple[int, int]]] = None,
    state_map: Optional[Dict[int, int]] = None,
    atmlst: Optional[List[int]] = None,
    ediff: bool = True,
    use_etfs: bool = True,
    max_memory: int = 6000,
    verbose: int = logger.INFO,
    cphf_options: Optional[Dict[str, float]] = None,
):
    if not states:
        raise ValueError('states cannot be empty')

    nac_pairs = _default_nac_pairs(states) if nac_pairs is None else list(nac_pairs)
    gradients = {}
    nacs = {}
    timings = {'gradient_times': {}, 'nac_pair_times': {}, 'cache_stats': None}

    grad_obj = tduks_sf_grad.Gradients(td_obj)
    grad_obj.max_memory = max_memory
    grad_obj.verbose = verbose
    _apply_cphf_options(grad_obj, cphf_options)

    nac_objects = {}
    for pair in nac_pairs:
        nac_obj = tduks_sf_nac.NAC(td_obj)
        nac_obj.max_memory = max_memory
        nac_obj.verbose = verbose
        _apply_cphf_options(nac_obj, cphf_options)
        nac_objects[pair] = nac_obj

    with GradientNACCacheManager(grad_obj, *nac_objects.values()) as cache_mgr:
        for state_id in states:
            pyscf_state = _resolve_state_id(state_id, state_map)
            t0 = time.perf_counter()
            gradients[state_id] = grad_obj.kernel(state=pyscf_state, atmlst=atmlst)
            timings['gradient_times'][state_id] = time.perf_counter() - t0

        for (state_i, state_j), nac_obj in nac_objects.items():
            t0 = time.perf_counter()
            nacs[(state_i, state_j)] = nac_obj.kernel(
                state_I=_resolve_state_id(state_i, state_map),
                state_J=_resolve_state_id(state_j, state_map),
                atmlst=atmlst,
                ediff=ediff,
                use_etfs=use_etfs,
            )
            timings['nac_pair_times'][(state_i, state_j)] = time.perf_counter() - t0

        timings['cache_stats'] = cache_mgr.get_stats()
    return gradients, nacs, timings


def compute_mecp_data(
    td_obj,
    state_I: int,
    state_J: int,
    state_map: Optional[Dict[int, int]] = None,
    atmlst: Optional[List[int]] = None,
    ediff: bool = False,
    use_etfs: bool = True,
    max_memory: int = 6000,
    verbose: int = logger.INFO,
    cphf_options: Optional[Dict[str, float]] = None,
):
    gradients, nacs, timings = compute_gradients_and_nacs(
        td_obj,
        states=[state_I, state_J],
        nac_pairs=[(state_I, state_J)],
        state_map=state_map,
        atmlst=atmlst,
        ediff=ediff,
        use_etfs=use_etfs,
        max_memory=max_memory,
        verbose=verbose,
        cphf_options=cphf_options,
    )
    return gradients[state_I], gradients[state_J], nacs[(state_I, state_J)], timings


def compute_fssh_data(
    td_obj,
    states: List[int],
    active_state: int,
    state_map: Optional[Dict[int, int]] = None,
    nac_pairs: Optional[List[Tuple[int, int]]] = None,
    atmlst: Optional[List[int]] = None,
    ediff: bool = True,
    use_etfs: bool = True,
    max_memory: int = 6000,
    verbose: int = logger.INFO,
    cphf_options: Optional[Dict[str, float]] = None,
):
    if active_state not in states:
        raise ValueError('active_state must be contained in states')

    nac_pairs = _default_nac_pairs(states) if nac_pairs is None else list(nac_pairs)
    timings = {'gradient_s': 0.0, 'nac_total_s': 0.0, 'nac_pair_times': {}, 'cache_stats': None}

    grad_obj = tduks_sf_grad.Gradients(td_obj)
    grad_obj.max_memory = max_memory
    grad_obj.verbose = verbose
    _apply_cphf_options(grad_obj, cphf_options)

    nac_objects = {}
    for pair in nac_pairs:
        nac_obj = tduks_sf_nac.NAC(td_obj)
        nac_obj.max_memory = max_memory
        nac_obj.verbose = verbose
        _apply_cphf_options(nac_obj, cphf_options)
        nac_objects[pair] = nac_obj

    with GradientNACCacheManager(grad_obj, *nac_objects.values()) as cache_mgr:
        grad_start = time.perf_counter()
        active_grad = grad_obj.kernel(state=_resolve_state_id(active_state, state_map), atmlst=atmlst)
        timings['gradient_s'] = time.perf_counter() - grad_start

        nacs = {}
        nac_start = time.perf_counter()
        for (state_i, state_j), nac_obj in nac_objects.items():
            pair_start = time.perf_counter()
            nacs[(state_i, state_j)] = nac_obj.kernel(
                state_I=_resolve_state_id(state_i, state_map),
                state_J=_resolve_state_id(state_j, state_map),
                atmlst=atmlst,
                ediff=ediff,
                use_etfs=use_etfs,
            )
            timings['nac_pair_times'][(state_i, state_j)] = time.perf_counter() - pair_start
        timings['nac_total_s'] = time.perf_counter() - nac_start
        timings['cache_stats'] = cache_mgr.get_stats()

    return -active_grad, nacs, timings
