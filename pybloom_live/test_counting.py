"""Tests for CountingBloomFilter: credential-based revocation semantics."""
import io
import pickle
from struct import pack

import pytest

from pybloom_live.counting import (
    TOKEN_SIZE,
    CountingBloomFilter,
    CounterOverflowError,
    CredentialTableFullError,
    InvalidCredentialError,
)


def find_colliding_keys(filt, n, start=0):
    """Return n distinct keys that hash to identical counter positions."""
    seen = {}
    i = start
    while True:
        key = 'collision-probe-%d' % i
        pos = filt._positions(key)
        if pos in seen:
            group = [seen[pos], key]
            while len(group) < n:
                i += 1
                key2 = 'collision-probe-%d' % i
                if filt._positions(key2) == pos:
                    group.append(key2)
            return group
        seen[pos] = key
        i += 1


# ---------------------------------------------------------------------
# basic insertion / membership / revocation
# ---------------------------------------------------------------------

def test_add_returns_opaque_credential_and_membership():
    bf = CountingBloomFilter(capacity=100, error_rate=0.01)
    token = bf.add('apple')
    assert isinstance(token, bytes)
    assert len(token) == TOKEN_SIZE
    assert 'apple' in bf
    assert 'banana' not in bf
    assert len(bf) == 1


def test_revoke_removes_exact_insertion():
    bf = CountingBloomFilter(capacity=100, error_rate=0.01)
    token = bf.add('apple')
    bf.revoke(token)
    assert 'apple' not in bf
    assert len(bf) == 0


def test_two_insertions_same_key_give_independent_credentials():
    bf = CountingBloomFilter(capacity=100, error_rate=0.01)
    t1 = bf.add('apple')
    t2 = bf.add('apple')
    assert t1 != t2

    bf.revoke(t1)
    # the second insertion's contribution must survive
    assert 'apple' in bf
    assert len(bf) == 1

    bf.revoke(t2)
    assert 'apple' not in bf
    assert len(bf) == 0


def test_revocation_of_one_key_preserves_colliding_key():
    # tiny filter: force distinct keys to share counter positions
    bf = CountingBloomFilter(capacity=4, error_rate=0.5)
    k1, k2 = find_colliding_keys(bf, 2)
    assert bf._positions(k1) == bf._positions(k2)

    t1 = bf.add(k1)
    bf.add(k2)
    bf.revoke(t1)
    # no false negative for the key whose insertion was not revoked
    assert k2 in bf
    assert k1 in bf  # k1 still "present" only via k2's counters (false pos ok)
    assert len(bf) == 1


def test_no_false_negatives_under_adversarial_collisions():
    # small filter, many colliding keys, interleaved inserts and revokes
    bf = CountingBloomFilter(capacity=4, error_rate=0.5, counter_bits=16,
                             max_credentials=None)
    keys = find_colliding_keys(bf, 6)
    live = {}
    for k in keys:
        live[k] = bf.add(k)
    # revoke every other insertion
    for k in keys[::2]:
        bf.revoke(live.pop(k))
    for k in keys[1::2]:
        assert k in bf, 'false negative for live key %r' % k
    # revoke the rest; counters must drain back to zero
    for k, tok in live.items():
        bf.revoke(tok)
    assert all(c == 0 for c in bf._counters)
    assert len(bf) == 0


# ---------------------------------------------------------------------
# credential misuse: forgery, cross-filter, replay
# ---------------------------------------------------------------------

def snapshot(bf):
    return (bytes(bf._counters), bf.count, dict(bf._credentials),
            bf._next_seq)


def test_forged_credentials_do_not_change_state():
    bf = CountingBloomFilter(capacity=100, error_rate=0.01)
    real = bf.add('apple')
    before = snapshot(bf)

    forgeries = [
        b'',                                  # empty
        b'\x00' * TOKEN_SIZE,                 # zero blob
        real[:-1] + bytes([real[-1] ^ 1]),    # tampered MAC
        real[:8] + bytes([real[8] ^ 1]) + real[9:],  # tampered seq
        os_urandom_token(),                   # random bytes
    ]
    for bad in forgeries:
        with pytest.raises(InvalidCredentialError):
            bf.revoke(bad)
        assert snapshot(bf) == before
        assert not bf.is_valid(bad)
    assert bf.is_valid(real)


def os_urandom_token():
    import os
    return os.urandom(TOKEN_SIZE)


def test_cross_filter_credential_rejected():
    bf1 = CountingBloomFilter(capacity=100, error_rate=0.01)
    bf2 = CountingBloomFilter(capacity=100, error_rate=0.01)
    token = bf1.add('apple')
    before = snapshot(bf2)
    with pytest.raises(InvalidCredentialError):
        bf2.revoke(token)
    assert snapshot(bf2) == before
    assert not bf2.is_valid(token)
    # still valid on its own filter
    assert bf1.is_valid(token)


def test_double_revoke_rejected_and_state_unchanged():
    bf = CountingBloomFilter(capacity=100, error_rate=0.01)
    t1 = bf.add('apple')
    t2 = bf.add('apple')
    bf.revoke(t1)
    before = snapshot(bf)
    with pytest.raises(InvalidCredentialError):
        bf.revoke(t1)
    assert snapshot(bf) == before
    assert 'apple' in bf  # t2's contribution intact
    bf.revoke(t2)


def test_credential_not_valid_after_restore_of_other_filter():
    bf1 = CountingBloomFilter(capacity=100, error_rate=0.01)
    token = bf1.add('apple')
    buf = io.BytesIO()
    bf1.tofile(buf)
    buf.seek(0)
    bf2 = CountingBloomFilter.fromfile(buf)  # same epoch/secret: valid
    assert bf2.is_valid(token)
    # but a fresh filter with same parameters rejects it
    bf3 = CountingBloomFilter(capacity=100, error_rate=0.01)
    assert not bf3.is_valid(token)


# ---------------------------------------------------------------------
# counter overflow
# ---------------------------------------------------------------------

def test_overflow_rejects_whole_insertion_atomically():
    # 2 slices x 2 counters, 8-bit counters -> max 255
    bf = CountingBloomFilter(capacity=2, error_rate=0.25, counter_bits=8,
                             max_credentials=None)
    assert bf.num_slices >= 2

    # find a key whose first-slice counter we can saturate
    target_pos = bf._positions('victim')[0]
    fillers = []
    i = 0
    while len(fillers) < 255:
        k = 'filler-%d' % i
        i += 1
        if bf._positions(k)[0] == target_pos:
            fillers.append(k)
    tokens = [bf.add(k) for k in fillers]
    assert bf._counters[target_pos] == 255

    # craft a key touching the saturated counter plus another, unsaturated one
    j = 0
    while True:
        victim = 'victim-%d' % j
        j += 1
        pos = bf._positions(victim)
        if pos[0] == target_pos and bf._counters[pos[1]] < 255:
            break
    before = snapshot(bf)
    with pytest.raises(CounterOverflowError):
        bf.add(victim)
    # whole insertion rejected: no counter changed, no credential issued
    assert snapshot(bf) == before
    assert len(bf) == 255

    # all 255 credentials still revoke cleanly down to zero
    for t in tokens:
        bf.revoke(t)
    assert all(c == 0 for c in bf._counters)
    assert len(bf) == 0


def test_no_saturate_then_decrement_to_wrong_zero():
    # If insertion saturated instead of rejecting, a revoke could drive a
    # shared counter to 0 while another live insertion still needs it.
    bf = CountingBloomFilter(capacity=2, error_rate=0.25, counter_bits=8,
                             max_credentials=None)
    keys = find_colliding_keys(bf, 256)  # all share every counter position
    pos = bf._positions(keys[0])
    tokens = [bf.add(k) for k in keys[:255]]
    assert all(bf._counters[p] == 255 for p in pos)

    # the 256th identical contribution must be refused, not saturated
    with pytest.raises(CounterOverflowError):
        bf.add(keys[255])
    assert all(bf._counters[p] == 255 for p in pos)
    assert len(bf) == 255

    # every live insertion still reports present; revokes drain exactly to 0
    for k in keys[:255]:
        assert k in bf
    for t in tokens:
        bf.revoke(t)
    assert all(c == 0 for c in bf._counters)
    assert keys[0] not in bf


# ---------------------------------------------------------------------
# bounded credential table
# ---------------------------------------------------------------------

def test_credential_table_bound_enforced():
    bf = CountingBloomFilter(capacity=100, error_rate=0.01, max_credentials=3)
    tokens = [bf.add('k%d' % i) for i in range(3)]
    with pytest.raises(CredentialTableFullError):
        bf.add('overflow')
    assert len(bf) == 3
    bf.revoke(tokens[0])
    bf.add('k3')  # space freed
    assert len(bf) == 3


def test_capacity_and_cost_introspection():
    bf = CountingBloomFilter(capacity=100, error_rate=0.01, max_credentials=10)
    assert bf.max_credentials == 10
    assert bf.active_credentials == 0
    assert bf.counter_bytes == bf.num_counters
    for i in range(4):
        bf.add('k%d' % i)
    assert bf.active_credentials == 4
    assert bf.credential_entry_bytes == 8 + 8 * bf.num_slices
    assert bf.credential_table_bytes > 0
    # persistent_bytes must match the real serialized size exactly
    buf = io.BytesIO()
    bf.tofile(buf)
    assert buf.tell() == bf.persistent_bytes()


# ---------------------------------------------------------------------
# serialization / pickle
# ---------------------------------------------------------------------

def test_serialization_preserves_counts_and_credentials():
    bf = CountingBloomFilter(capacity=100, error_rate=0.01)
    t1 = bf.add('apple')
    t2 = bf.add('banana')
    bf.add('cherry')
    bf.revoke(t2)  # revoked before save: must stay revoked

    buf = io.BytesIO()
    bf.tofile(buf)
    buf.seek(0)
    bf2 = CountingBloomFilter.fromfile(buf)

    assert len(bf2) == 2
    assert 'apple' in bf2 and 'cherry' in bf2
    assert bytes(bf2._counters) == bytes(bf._counters)

    # credential issued before save revokes fine after restore
    bf2.revoke(t1)
    assert 'apple' not in bf2
    # already-revoked credential stays revoked
    with pytest.raises(InvalidCredentialError):
        bf2.revoke(t2)
    # forgery still rejected
    with pytest.raises(InvalidCredentialError):
        bf2.revoke(t1[:-1] + bytes([t1[-1] ^ 1]))


def test_serialization_to_real_file(tmp_path):
    bf = CountingBloomFilter(capacity=50, error_rate=0.01)
    token = bf.add('persisted')
    path = tmp_path / 'filter.cbf'
    with open(path, 'wb') as f:
        bf.tofile(f)
    assert path.stat().st_size == bf.persistent_bytes()
    with open(path, 'rb') as f:
        bf2 = CountingBloomFilter.fromfile(f)
    assert 'persisted' in bf2
    bf2.revoke(token)
    assert 'persisted' not in bf2


def test_pickle_roundtrip():
    bf = CountingBloomFilter(capacity=100, error_rate=0.01)
    token = bf.add('apple')
    bf2 = pickle.loads(pickle.dumps(bf))
    assert 'apple' in bf2
    bf2.revoke(token)
    assert 'apple' not in bf2
    with pytest.raises(InvalidCredentialError):
        bf2.revoke(token)


def test_fromfile_rejects_truncated_and_foreign_data():
    bf = CountingBloomFilter(capacity=10, error_rate=0.01)
    bf.add('x')
    buf = io.BytesIO()
    bf.tofile(buf)
    data = buf.getvalue()
    for cut in (10, len(data) // 2, len(data) - 1):
        with pytest.raises(Exception):
            CountingBloomFilter.fromfile(io.BytesIO(data[:cut]))
    with pytest.raises(ValueError):
        CountingBloomFilter.fromfile(io.BytesIO(b'XXXX' + data[4:]))
