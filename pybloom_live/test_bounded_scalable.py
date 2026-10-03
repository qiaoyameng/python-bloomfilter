"""Tests for BoundedScalableBloomFilter's total-budget contract.

The focus is on the deterministic guarantees of the data structure:

- the per-layer budget allocation formula (geometric series bounded by
  the configured total error rate),
- per-layer parameters derived from each layer's own budget,
- growth driven by actual bit occupancy rather than the inexact count,
- absence of false negatives across many growths and across
  serialization round-trips,
- the union contract (count re-estimation and refusal of combinations
  that can no longer guarantee the total budget).

Empirical false-positive rates are only checked as a loose supplement,
never as the sole assertion of correctness.
"""
import io
import math
import pickle
import random

import pytest

from pybloom_live.pybloom import (
    BloomFilter,
    BoundedScalableBloomFilter,
    ScalableBloomFilter,
)


def make_sbf(**kwargs):
    """Create a small BoundedScalableBloomFilter for deterministic tests."""
    params = dict(initial_capacity=4, error_rate=0.1,
                  mode=BoundedScalableBloomFilter.SMALL_SET_GROWTH,
                  ratio=0.5, saturation_threshold=0.5)
    params.update(kwargs)
    return BoundedScalableBloomFilter(**params)


def grow_to_layers(sbf, num_layers, prefix="key"):
    """Add distinct keys until the filter has at least num_layers layers."""
    i = 0
    while len(sbf.filters) < num_layers:
        sbf.add("%s-%d" % (prefix, i))
        i += 1
    return i


class TestParameterValidation:
    """Constructor rejects parameters outside their valid ranges."""

    @pytest.mark.parametrize("error_rate", [0, -0.1, 1, 1.5])
    def test_invalid_error_rate(self, error_rate):
        with pytest.raises(ValueError):
            make_sbf(error_rate=error_rate)

    @pytest.mark.parametrize("ratio", [0, -0.5, 1, 2])
    def test_invalid_ratio(self, ratio):
        with pytest.raises(ValueError):
            make_sbf(ratio=ratio)

    @pytest.mark.parametrize("mode", [0, 1, -2])
    def test_invalid_mode(self, mode):
        with pytest.raises(ValueError):
            make_sbf(mode=mode)

    def test_invalid_initial_capacity(self):
        with pytest.raises(ValueError):
            make_sbf(initial_capacity=0)

    @pytest.mark.parametrize("threshold", [0, 1, -0.1, 1.1])
    def test_invalid_saturation_threshold(self, threshold):
        with pytest.raises(ValueError):
            make_sbf(saturation_threshold=threshold)


class TestBudgetAllocation:
    """The per-layer budgets follow the geometric allocation formula."""

    def test_layer_budget_formula(self):
        sbf = make_sbf(error_rate=0.2, ratio=0.5)
        for i in range(6):
            assert sbf.layer_budget(i) == pytest.approx(
                0.2 * 0.5 * 0.5 ** i)

    def test_budget_series_sums_to_total(self):
        sbf = make_sbf(error_rate=0.13, ratio=0.9)
        partial = sum(sbf.layer_budget(i) for i in range(2000))
        assert partial == pytest.approx(0.13, rel=1e-9)

    def test_layer_capacity_formula(self):
        sbf = make_sbf(initial_capacity=5,
                       mode=BoundedScalableBloomFilter.LARGE_SET_GROWTH)
        for i in range(4):
            assert sbf.layer_capacity(i) == 5 * 4 ** i

    def test_layer_parameters_derived_from_own_budget(self):
        """Each layer's k/m/capacity match a BloomFilter built from p_i."""
        sbf = make_sbf()
        grow_to_layers(sbf, 4)
        assert len(sbf.filters) >= 4
        for i, layer in enumerate(sbf.filters):
            reference = BloomFilter(capacity=sbf.layer_capacity(i),
                                    error_rate=sbf.layer_budget(i))
            assert layer.error_rate == pytest.approx(sbf.layer_budget(i))
            assert layer.capacity == sbf.layer_capacity(i)
            assert layer.num_slices == reference.num_slices
            assert layer.bits_per_slice == reference.bits_per_slice
            assert layer.num_bits == reference.num_bits

    def test_layers_do_not_share_a_single_error_rate(self):
        sbf = make_sbf()
        grow_to_layers(sbf, 3)
        rates = [f.error_rate for f in sbf.filters]
        assert len(set(rates)) == len(rates)
        assert rates == sorted(rates, reverse=True)
        assert all(r < sbf.error_rate for r in rates)

    def test_error_bound_contract(self):
        sbf = make_sbf(error_rate=0.2, ratio=0.5)
        assert sbf.error_bound == 0
        grow_to_layers(sbf, 4)
        previous = 0.0
        for i in range(1, len(sbf.filters) + 1):
            bound = sum(sbf.layer_budget(j) for j in range(i))
            assert bound > previous
            previous = bound
        assert sbf.error_bound == pytest.approx(
            sum(sbf.layer_budget(i) for i in range(len(sbf.filters))))
        assert 0 < sbf.error_bound < sbf.error_rate
        assert sbf.remaining_budget == pytest.approx(
            sbf.error_rate - sbf.error_bound)
        assert sbf.remaining_budget > 0


class TestGrowthAndSaturation:
    """Growth is driven by bit occupancy, with count only as a backstop."""

    def test_tiny_capacity_triggers_many_layers(self):
        sbf = make_sbf()
        for i in range(300):
            sbf.add("key-%d" % i)
        assert len(sbf.filters) >= 5

    def test_non_last_layers_are_saturated(self):
        """Every layer that was grown past satisfied the saturation rule."""
        sbf = make_sbf()
        for i in range(200):
            sbf.add("key-%d" % i)
        for layer in sbf.filters[:-1]:
            fill = layer.bitarray.count() / layer.num_bits
            assert fill >= sbf.saturation_threshold or \
                layer.count >= layer.capacity

    def test_no_false_negatives_across_growth(self):
        sbf = make_sbf(error_rate=1e-6)
        keys = ["key-%d" % i for i in range(300)]
        for key in keys:
            assert sbf.add(key) is False
        for key in keys:
            assert key in sbf
        assert len(sbf) == 300
        assert sbf.count == 300

    def test_duplicates_do_not_inflate_count_or_trigger_growth(self):
        """count is not an exact dedup count, but re-adding known keys
        must neither grow the filter nor increment the count."""
        sbf = make_sbf()
        for _ in range(50):
            for key in ("a", "b", "c"):
                sbf.add(key)
        assert len(sbf.filters) == 1
        assert len(sbf) == 3
        assert sbf.add("a") is True

    def test_estimated_count_tracks_insertions(self):
        sbf = make_sbf(initial_capacity=16, error_rate=0.01)
        for i in range(100):
            sbf.add("key-%d" % i)
        assert 60 <= sbf.estimated_count <= 160

    def test_capacity_property(self):
        sbf = make_sbf()
        grow_to_layers(sbf, 3)
        assert sbf.capacity == sum(
            sbf.layer_capacity(i) for i in range(len(sbf.filters)))


class TestUnion:
    """Union re-estimates occupancy and refuses unguaranteed merges."""

    def test_union_rejects_other_types(self):
        sbf = make_sbf()
        sbf.add("a")
        with pytest.raises(ValueError):
            sbf.union(BloomFilter(capacity=10, error_rate=0.1))
        with pytest.raises(ValueError):
            sbf.union(ScalableBloomFilter(initial_capacity=10))

    def test_union_rejects_mismatched_strategy(self):
        sbf = make_sbf()
        sbf.add("a")
        for kwargs in ({"ratio": 0.6}, {"error_rate": 0.2},
                       {"initial_capacity": 8},
                       {"mode": BoundedScalableBloomFilter.LARGE_SET_GROWTH},
                       {"saturation_threshold": 0.6}):
            other = make_sbf(**kwargs)
            other.add("b")
            with pytest.raises(ValueError):
                sbf.union(other)

    def test_union_merges_and_reestimates_count(self):
        """Merged layer counts come from bit occupancy, not pre-merge
        counts, and no members are lost."""
        sbf1 = make_sbf(initial_capacity=8, saturation_threshold=0.95)
        sbf2 = make_sbf(initial_capacity=8, saturation_threshold=0.95)
        keys1 = ["a-%d" % i for i in range(3)]
        keys2 = ["b-%d" % i for i in range(3)]
        for key in keys1:
            sbf1.add(key)
        for key in keys2:
            sbf2.add(key)

        merged = sbf1 | sbf2
        assert len(merged.filters) == 1
        layer = merged.filters[0]
        set_bits = layer.bitarray.count()
        expected = -(layer.num_bits / layer.num_slices) * math.log(
            1.0 - set_bits / layer.num_bits)
        # Count is re-estimated from the bits (rounded up), not the sum
        # of the pre-merge counts.
        assert layer.count == int(math.ceil(expected))
        for key in keys1 + keys2:
            assert key in merged

    def test_union_with_unequal_layer_counts(self):
        sbf1 = make_sbf(saturation_threshold=0.95)
        sbf2 = make_sbf(saturation_threshold=0.95)
        n1 = grow_to_layers(sbf1, 3, prefix="one")
        n2 = grow_to_layers(sbf2, 1, prefix="two")
        merged = sbf1.union(sbf2)
        assert len(merged.filters) == len(sbf1.filters) == 3
        for i in range(n1):
            assert "one-%d" % i in merged
        for i in range(n2):
            assert "two-%d" % i in merged

    def test_union_refuses_oversaturated_merge(self):
        """A merge that would saturate a layer is rejected outright."""
        sbf1 = make_sbf()
        sbf2 = make_sbf()
        sbf1.add("a")
        sbf2.add("b")
        # Simulate two heavily loaded filters whose combined occupancy
        # exceeds the saturation threshold of layer 0.
        sbf1.filters[0].bitarray.setall(True)
        sbf2.filters[0].bitarray.setall(True)
        with pytest.raises(ValueError, match="budget"):
            sbf1.union(sbf2)

    def test_union_does_not_mutate_operands(self):
        sbf1 = make_sbf(initial_capacity=8, saturation_threshold=0.95)
        sbf2 = make_sbf(initial_capacity=8, saturation_threshold=0.95)
        sbf1.add("a")
        sbf2.add("b")
        bits1 = sbf1.filters[0].bitarray.copy()
        bits2 = sbf2.filters[0].bitarray.copy()
        sbf1.union(sbf2)
        assert sbf1.filters[0].bitarray == bits1
        assert sbf2.filters[0].bitarray == bits2


class TestPersistence:
    """Serialization records the strategy and every layer's parameters."""

    def test_roundtrip_preserves_strategy_and_layers(self):
        sbf = make_sbf()
        n = grow_to_layers(sbf, 3)
        keys = ["key-%d" % i for i in range(n)]

        bio = io.BytesIO()
        sbf.tofile(bio)
        bio.seek(0)
        restored = BoundedScalableBloomFilter.fromfile(bio)

        assert restored.scale == sbf.scale
        assert restored.ratio == sbf.ratio
        assert restored.initial_capacity == sbf.initial_capacity
        assert restored.error_rate == sbf.error_rate
        assert restored.saturation_threshold == sbf.saturation_threshold
        assert len(restored.filters) == len(sbf.filters)
        for original, loaded in zip(sbf.filters, restored.filters):
            assert loaded.error_rate == original.error_rate
            assert loaded.num_slices == original.num_slices
            assert loaded.bits_per_slice == original.bits_per_slice
            assert loaded.capacity == original.capacity
            assert loaded.count == original.count
            # Deserialized bit arrays may include byte-alignment padding;
            # the significant bits must be identical.
            assert loaded.bitarray[:original.num_bits] == original.bitarray
        for key in keys:
            assert key in restored
        assert restored.error_bound == pytest.approx(sbf.error_bound)

    def test_growth_after_restore_keeps_hashes_and_budget_series(self):
        """Restoring and growing further must not rehash existing keys and
        must continue the same geometric budget series."""
        sbf = make_sbf()
        n = grow_to_layers(sbf, 2)
        keys = ["key-%d" % i for i in range(n)]

        bio = io.BytesIO()
        sbf.tofile(bio)
        bio.seek(0)
        restored = BoundedScalableBloomFilter.fromfile(bio)

        layers_before = len(restored.filters)
        m = grow_to_layers(restored, layers_before + 2, prefix="new")
        assert len(restored.filters) >= layers_before + 2
        # New layers follow the same allocation formula.
        for i in range(layers_before, len(restored.filters)):
            assert restored.filters[i].error_rate == pytest.approx(
                restored.layer_budget(i))
            assert restored.filters[i].capacity == restored.layer_capacity(i)
        # Old keys are still found (their hashes never changed) and new
        # keys exhibit no false negatives either.
        for key in keys:
            assert key in restored
        for i in range(m):
            assert "new-%d" % i in restored

    def test_fromfile_rejects_foreign_data(self):
        other = ScalableBloomFilter(initial_capacity=10)
        other.add("x")
        bio = io.BytesIO()
        other.tofile(bio)
        bio.seek(0)
        with pytest.raises(ValueError):
            BoundedScalableBloomFilter.fromfile(bio)

    def test_empty_filter_roundtrip(self):
        sbf = make_sbf()
        bio = io.BytesIO()
        sbf.tofile(bio)
        bio.seek(0)
        restored = BoundedScalableBloomFilter.fromfile(bio)
        assert restored.filters == []
        restored.add("a")
        assert "a" in restored
        assert restored.filters[0].error_rate == pytest.approx(
            restored.layer_budget(0))

    def test_pickle_roundtrip(self):
        sbf = make_sbf()
        n = grow_to_layers(sbf, 3)
        restored = pickle.loads(pickle.dumps(sbf))
        assert restored.error_rate == sbf.error_rate
        assert restored.ratio == sbf.ratio
        assert len(restored.filters) == len(sbf.filters)
        for i in range(n):
            assert "key-%d" % i in restored


class TestEmpiricalFalsePositiveRate:
    """Supplementary statistical check (never the sole assertion)."""

    def test_empirical_fp_rate_within_total_budget(self):
        sbf = BoundedScalableBloomFilter(
            initial_capacity=50, error_rate=0.1,
            mode=BoundedScalableBloomFilter.SMALL_SET_GROWTH, ratio=0.9)
        for i in range(400):
            sbf.add("member-%d" % i)
        assert len(sbf.filters) >= 3

        rng = random.Random(20261004)
        queries = ["nonmember-%d" % i for i in range(5000)]
        rng.shuffle(queries)
        false_positives = sum(1 for q in queries if q in sbf)
        # Loose supplementary bound: the measured rate must stay under
        # the configured total budget (expected value is error_bound,
        # well below 0.1 for these parameters).
        assert false_positives / len(queries) <= sbf.error_rate
