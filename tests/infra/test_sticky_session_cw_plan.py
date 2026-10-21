# Copyright (c) ModelScope Contributors. All rights reserved.
"""Session affinity remains stable across load and worker-topology changes."""
from collections import OrderedDict

from twinkle.infra.continuous_work import _CW_SESSION_IDLE_TIMEOUT, _sticky_session_cw_plan


def test_sessions_stay_bound_when_load_changes():
    bindings = OrderedDict()
    load = [0, 0]
    assert _sticky_session_cw_plan(('a', 'b'), load, ('s1', 's2', 's1'), bindings, now=0) == [[0, 2], [1]]
    assert load == [2, 1]

    # An existing session stays on its worker even when the other worker is idle.
    load = [100, 0]
    assert _sticky_session_cw_plan(('a', 'b'), load, ('s1', 's2'), bindings, now=1) == [[0], [1]]
    assert load == [101, 1]


def test_worker_reordering_and_addition_preserve_existing_sessions():
    bindings = OrderedDict()
    _sticky_session_cw_plan(('a', 'b'), [0, 0], ('s1', 's2'), bindings, now=0)

    assert _sticky_session_cw_plan(
        ('c', 'b', 'a'), [0, 0, 100], ('s1', 's2'), bindings, now=1) == [[], [1], [0]]


def test_missing_worker_rebinds_and_does_not_move_back_when_it_returns():
    bindings = OrderedDict()
    _sticky_session_cw_plan(('a', 'b'), [0, 0], ('s1',), bindings, now=0)
    assert _sticky_session_cw_plan(('b',), [0], ('s1',), bindings, now=1) == [[0]]
    assert _sticky_session_cw_plan(('a', 'b'), [0, 100], ('s1',), bindings, now=2) == [[], [0]]


def test_none_sessions_use_load_balancing_without_creating_bindings():
    bindings = OrderedDict()
    assert _sticky_session_cw_plan(
        ('a', 'b'), [0, 0], (None, 's1', None, 's1'), bindings, now=0) == [[0, 2], [1, 3]]
    assert list(bindings) == ['s1']


def test_idle_session_expires_and_can_choose_a_new_worker():
    bindings = OrderedDict()
    _sticky_session_cw_plan(('a', 'b'), [0, 0], ('s1',), bindings, now=0)
    assert _sticky_session_cw_plan(
        ('a', 'b'), [100, 0], ('s1',), bindings, now=_CW_SESSION_IDLE_TIMEOUT) == [[], [0]]


def test_recently_used_sessions_survive_idle_cleanup():
    bindings = OrderedDict()
    _sticky_session_cw_plan(('a', 'b'), [0, 0], ('active', 'idle'), bindings, now=0)
    _sticky_session_cw_plan(('a', 'b'), [0, 0], ('active',), bindings, now=_CW_SESSION_IDLE_TIMEOUT - 1)
    _sticky_session_cw_plan(('a', 'b'), [0, 0], (), bindings, now=_CW_SESSION_IDLE_TIMEOUT)
    assert list(bindings) == ['active']
