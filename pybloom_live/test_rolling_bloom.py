"""Tests for RollingBloomFilter.

Every test drives a manually controlled FakeClock, so the whole suite runs
without waiting a real second (let alone a real hour).

Coverage:
- Same key replayed across two slices stays queryable after the old slice
  rotates out (insert always lands in the current slice).
- Minimum retention and maximum extra retention bounds.
- Jumps across several slices and past the whole ring.
- Exact boundary instants.
- Clock rollback (clamping, no resurrection of expired slices).
- Persistence round trip (binary file and pickle).
- Rotation never clears data freshly written into the reused slot.
- Thread-safety under concurrent inserts (single internal lock).
- Union false-positive budget split across the valid slices.
"""
import io
import pickle
import random
import threading

import pytest

from pybloom_live.pybloom import BloomFilter, FakeClock, RollingBloomFilter


WINDOW = 3600.0
NUM_SLICES = 4
# slice_seconds = window / (num_slices - 1) = 1200s
SLICE = WINDOW / (NUM_SLICES - 1)


def make_filter(**kwargs):
    clock = kwargs.pop('clock', FakeClock())
    rbf = RollingBloomFilter(
        window=WINDOW, num_slices=NUM_SLICES,
        slice_capacity=kwargs.pop('slice_capacity', 5000),
        error_rate=kwargs.pop('error_rate', 0.001),
        time_func=clock, **kwargs)
    return rbf, clock


def test_retention_contract():
    rbf, clock = make_filter()
    assert rbf.min_retention == WINDOW
    assert rbf.max_extra_retention == pytest.approx(SLICE)
    assert rbf.num_slots == NUM_SLICES + 1


def test_same_key_spanning_two_slices_survives_old_slice_exit():
    """Headline scenario from the spec, replayed with a fake clock."""
    rbf, clock = make_filter()

    # t = 0: key first lands in bucket 0.
    assert rbf.add('key') is False

    # Move to the boundary between bucket 0 and bucket 1. Boundary instants
    # belong to the new bucket.
    clock.set(SLICE)
    assert rbf.add('other') is False  # ensure rotation happens
    # The key is already matched by the *old* slice, but it must nonetheless
    # be written into the current slice.
    assert rbf.add('key') is True
    assert 'key' in rbf

    # bucket 0 is valid for queries in buckets 0..3; at bucket 4 it is gone.
    clock.set(WINDOW + SLICE)  # bucket 4
    # 'other' was only written into bucket 1, still covered by buckets 1..4.
    assert 'other' in rbf
    # 'key' survives because it was re-inserted at t=1200.
    assert 'key' in rbf

    # Both keys were last written in bucket 1 (t=1200): guaranteed present
    # through 1200+W=4800 and gone by 1200+W+d=6000.
    clock.set(WINDOW + 2 * SLICE - 1e-9)  # still bucket 4
    assert 'other' in rbf
    assert 'key' in rbf
    clock.set(WINDOW + 2 * SLICE)  # bucket 5 boundary -> both expired
    assert 'other' not in rbf
    assert 'key' not in rbf

    # Contrast: a key inserted only once at t=0 is guaranteed gone by W+d.
    rbf2, clock2 = make_filter()
    rbf2.add('once')
    clock2.set(SLICE)
    rbf2.add('dummy')
    clock2.set(WINDOW + SLICE)
    assert 'once' not in rbf2


def test_minimum_retention_and_max_extra_bounds():
    rbf, clock = make_filter()
    rbf.add('x')
    # Present for the whole window.
    clock.set(WINDOW)
    assert 'x' in rbf
    # Forgotten no later than window + one slice.
    clock.set(WINDOW + SLICE)
    assert 'x' not in rbf

    # Sweep offsets inside one slice: every key obeys the bounds.
    for offset in [0.0, 0.25 * SLICE, 0.99 * SLICE, SLICE - 1e-9]:
        f, c = make_filter()
        c.set(offset)
        f.add('y')
        c.set(offset + WINDOW)
        assert 'y' in f
        c.set(offset + WINDOW + SLICE)
        assert 'y' not in f


def test_exact_boundary_belong_to_new_slice():
    rbf, clock = make_filter()
    rbf.add('a')                      # bucket 0 at t=0
    clock.set(SLICE)                  # exact boundary -> bucket 1
    rbf.add('b')
    # 'a' must still be queryable immediately after the rotation.
    assert 'a' in rbf
    assert 'b' in rbf


def test_jump_across_several_slices_clears_only_crossed_slots():
    rbf, clock = make_filter()
    rbf.add('keep0')
    clock.set(SLICE)
    rbf.add('keep1')
    # Jump forward three slices to bucket 4: buckets 1..4 remain valid,
    # bucket 0 expires.
    clock.set(4 * SLICE)
    assert 'keep1' in rbf
    assert 'keep0' not in rbf


def test_jump_past_whole_ring_resets_everything():
    rbf, clock = make_filter()
    rbf.add('a')
    clock.set(SLICE)
    rbf.add('b')
    # Jump beyond all physical slots; stale bits must not resurrect.
    clock.set(100 * SLICE)
    assert 'a' not in rbf
    assert 'b' not in rbf
    # Filter is still usable after the big jump.
    rbf.add('c')
    assert 'c' in rbf
    clock.set(100 * SLICE + 1)
    assert 'c' in rbf


def test_clock_rollback_is_clamped():
    rbf, clock = make_filter()
    clock.set(10 * SLICE)
    rbf.add('late')
    # Roll the clock back to bucket 0; time is clamped to the last seen value
    # so expired slices can never be written to or resurrected.
    clock.set(0)
    assert 'late' in rbf
    rbf.add('late')  # re-insert under clamped time, into the current slice
    assert 'late' in rbf
    clock.set(5 * SLICE)  # still behind 10*d -> remains clamped
    assert 'late' in rbf
    clock.set(10 * SLICE + WINDOW + SLICE)  # advance past expiry
    assert 'late' not in rbf


def test_persistence_roundtrip_file():
    rbf, clock = make_filter()
    rbf.add('persist-me')
    clock.set(SLICE)
    rbf.add('second-key')
    clock.set(2 * SLICE + 5)

    buf = io.BytesIO()
    rbf.tofile(buf)
    buf.seek(0)

    restored_clock = FakeClock(clock())
    restored = RollingBloomFilter.fromfile(buf, time_func=restored_clock)
    assert 'persist-me' in restored
    assert 'second-key' in restored
    assert restored.slice_seconds == pytest.approx(SLICE)
    assert restored.min_retention == WINDOW

    # Continuing from restored state expires data on schedule.
    restored_clock.set(WINDOW + SLICE)
    assert 'persist-me' not in restored
    assert 'second-key' in restored


def test_persistence_roundtrip_pickle():
    rbf, clock = make_filter()
    rbf.add('pickled')
    clock.set(SLICE)
    data = pickle.dumps(rbf)
    restored = pickle.loads(data)
    restored._time_func = FakeClock(clock())
    assert 'pickled' in restored


def test_rotation_does_not_clear_fresh_data_in_reused_slot():
    """White-box check: the reused slot is always the spare."""
    rbf, clock = make_filter()
    rbf.add('old')                              # bucket 0
    clock.set(SLICE - 1)
    rbf.add('just-before')                      # still bucket 0
    clock.set(SLICE)                            # rotate to bucket 1
    rbf.add('just-after')                       # bucket 1
    assert 'old' in rbf
    assert 'just-before' in rbf
    assert 'just-after' in rbf

    # Walk the ring forward many rotations; keys inserted within the window
    # must always survive regardless of physical slot reuse.
    f, c = make_filter()
    keys = {}
    for bucket in range(20):
        c.set(bucket * SLICE + 1)
        key = f'k{bucket}'
        keys[bucket] = key
        f.add(key)
        for past in range(bucket - (NUM_SLICES - 1), bucket + 1):
            if past >= 0:
                assert keys[past] in f
        if bucket >= NUM_SLICES:
            assert keys[bucket - NUM_SLICES] not in f


def test_concurrent_inserts_with_single_lock():
    rbf, _ = make_filter(slice_capacity=100000)
    n_threads = 8
    per_thread = 500

    def worker(tid):
        for i in range(per_thread):
            rbf.add(f't{tid}-{i}')

    threads = [threading.Thread(target=worker, args=(t,))
               for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(rbf) == n_threads * per_thread
    for t in range(n_threads):
        assert f't{t}-{per_thread // 2}' in rbf


def test_union_false_positive_budget_split():
    error_rate = 0.02
    rbf, _ = make_filter(error_rate=error_rate)
    # (1 - q) ** K == 1 - error_rate
    q = 1.0 - (1.0 - error_rate) ** (1.0 / NUM_SLICES)
    assert rbf.slice_error_rate == pytest.approx(q)
    assert 1.0 - (1.0 - q) ** NUM_SLICES == pytest.approx(error_rate)

    # Empirical sanity check with generous margins: fill every valid slice,
    # confirm no false negatives, then probe keys never inserted.
    rbf2, clock = make_filter(slice_capacity=20000, error_rate=error_rate)
    inserted = [f'inserted-{i}' for i in range(8000)]
    for bucket in range(NUM_SLICES):
        clock.set(bucket * SLICE + 1)
        for key in inserted[bucket * 2000:(bucket + 1) * 2000]:
            rbf2.add(key)
    assert all(k in rbf2 for k in inserted)

    rng = random.Random(1234)
    hits = 0
    trials = 20000
    for _ in range(trials):
        if f'never-{rng.random()}' in rbf2:
            hits += 1
    observed = hits / trials
    # Observed union FPP should stay within a loose 3x band of the budget.
    assert observed < error_rate * 3


def test_invalid_construction_arguments():
    with pytest.raises(ValueError):
        RollingBloomFilter(window=0)
    with pytest.raises(ValueError):
        RollingBloomFilter(num_slices=1)
    with pytest.raises(ValueError):
        RollingBloomFilter(slice_capacity=0)
    with pytest.raises(ValueError):
        RollingBloomFilter(error_rate=0)
    with pytest.raises(ValueError):
        RollingBloomFilter(error_rate=1)


def test_touch_advances_without_insert():
    rbf, clock = make_filter()
    rbf.add('x')
    clock.set(WINDOW + SLICE)
    rbf.touch()
    assert 'x' not in rbf


def test_fromfile_rejects_foreign_stream():
    plain = BloomFilter(capacity=10, error_rate=0.01)
    buf = io.BytesIO()
    plain.tofile(buf)
    buf.seek(0)
    with pytest.raises(ValueError):
        RollingBloomFilter.fromfile(buf)
