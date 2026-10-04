[![Build Status](https://travis-ci.org/joseph-fox/python-bloomfilter.svg?branch=master)](https://travis-ci.org/joseph-fox/python-bloomfilter)

# pybloom_live

A fast, scalable Python implementation of Bloom filters - space-efficient probabilistic data structures for set membership testing.

## What is a Bloom Filter?

A Bloom filter is a space-efficient probabilistic data structure that is used to test whether an element is a member of a set. It can tell you with certainty that an element is **not** in the set, or that it **might be** in the set (with a controllable false positive probability). Bloom filters never produce false negatives.

**Use cases:**
- Web crawlers: Track visited URLs to avoid re-crawling
- Caching systems: Quick membership checks before expensive lookups
- Database optimization: Pre-filter queries to avoid unnecessary disk reads
- Distributed systems: Efficient set reconciliation
- Network routers: Fast packet filtering
- Spell checkers: Quick dictionary lookups

## Features

- **Two implementations:**
  - `BloomFilter`: Fixed-capacity filter for known dataset sizes
  - `ScalableBloomFilter`: Automatically grows to accommodate more elements

- **High performance:** Uses xxHash for 128-bit hashing - one of the fastest non-cryptographic hash functions available

- **Set operations:** Union and intersection support for combining filters

- **Serialization:** Efficient binary file format for saving/loading filters

- **Python 3.6+:** Modern Python 3 implementation without legacy Python 2 baggage

- **Optimized space usage:** Tightening ratio of 0.9 provides excellent space efficiency across a wide range of growth patterns

## Installation

Install from PyPI:

```bash
pip install pybloom-live
```

Or install from source:

```bash
git clone https://github.com/joseph-fox/python-bloomfilter.git
cd python-bloomfilter
pip install .
```

Verify installation:

```bash
pip show pybloom-live
# Should show version 4.0.0 or higher
```

## Quick Start

### Basic BloomFilter

```python
import pybloom_live

# Create a Bloom Filter with capacity for 1000 elements
# and a 0.1% false positive rate
f = pybloom_live.BloomFilter(capacity=1000, error_rate=0.001)

# Add elements
f.add("apple")
f.add("banana")
f.add("orange")

# Check membership
print("apple" in f)      # True
print("grape" in f)      # False

# The add() method returns True if the element might already exist
already_exists = f.add("apple")  # Returns True
already_exists = f.add("mango")  # Returns False
```

### ScalableBloomFilter

The `ScalableBloomFilter` automatically grows as you add more elements, maintaining the target error rate:

```python
import pybloom_live

# Create a scalable filter with initial capacity of 100
sbf = pybloom_live.ScalableBloomFilter(
    initial_capacity=100,
    error_rate=0.001,
    mode=pybloom_live.ScalableBloomFilter.LARGE_SET_GROWTH
)

# Add many elements - it scales automatically
for i in range(10000):
    sbf.add(i)

print(f"Elements: {len(sbf)}")           # ~10000
print(f"Capacity: {sbf.capacity}")       # Automatically expanded
print(f"Filters: {len(sbf.filters)}")    # Multiple internal filters

# Check membership
print(5000 in sbf)   # True
print(10000 in sbf)  # False
```

**Growth modes:**
- `SMALL_SET_GROWTH = 2`: Slower growth, uses less memory
- `LARGE_SET_GROWTH = 4`: Faster growth (default), better for large datasets

## Examples

### Example 1: Web Crawler - Tracking Visited URLs

```python
import pybloom_live

# Track visited URLs to avoid re-crawling
visited = pybloom_live.ScalableBloomFilter(
    initial_capacity=10000,
    error_rate=0.001
)

urls_to_crawl = [
    "https://example.com/page1",
    "https://example.com/page2",
    "https://example.com/page1",  # duplicate
    "https://example.com/page3",
]

for url in urls_to_crawl:
    if url in visited:
        print(f"SKIP (already crawled): {url}")
    else:
        visited.add(url)
        print(f"CRAWL: {url}")
        # crawl_page(url)
```

### Example 2: Set Operations

Combine filters using union and intersection:

```python
import pybloom_live

# Create two filters with same capacity and error rate
f1 = pybloom_live.BloomFilter(capacity=100, error_rate=0.01)
f2 = pybloom_live.BloomFilter(capacity=100, error_rate=0.01)

# Add elements
for i in range(20):
    f1.add(i)

for i in range(10, 30):
    f2.add(i)

# Union - elements in either filter
f_union = f1 | f2  # or f1.union(f2)
print(0 in f_union)   # True (in f1)
print(15 in f_union)  # True (in both)
print(25 in f_union)  # True (in f2)

# Intersection - elements likely in both filters
f_intersect = f1 & f2  # or f1.intersection(f2)
print(5 in f_intersect)   # False (only in f1)
print(15 in f_intersect)  # True (in both)
print(25 in f_intersect)  # False (only in f2)
```

### Example 3: Serialization

Save and load filters from disk:

```python
import pybloom_live

# Create and populate a filter
bf = pybloom_live.BloomFilter(capacity=10000, error_rate=0.001)
for i in range(1000):
    bf.add(f"user_{i}")

# Save to file
with open('bloom_filter.bin', 'wb') as f:
    bf.tofile(f)

# Load from file
with open('bloom_filter.bin', 'rb') as f:
    loaded_bf = pybloom_live.BloomFilter.fromfile(f)

# Old v4 files must be requested explicitly:
# pybloom_live.BloomFilter.fromfile(f, legacy=True)

# Verify
print("user_500" in loaded_bf)    # True
print("user_9999" in loaded_bf)   # False
```

### Portable File Format (v1)

`tofile()` writes a versioned, self-delimiting binary frame. The frame records
the key encoding (UTF-8 for strings, identity bytes for bytes-like keys),
little-endian integer unpacking, LSB-first bit order, selected hash algorithm
and version, deterministic salts, slice count, bits per slice, capacity,
count, effective bit count, payload length, and CRC32 checksum.

Each call to `fromfile()` consumes exactly one object, so multiple filters can
be concatenated in one stream. Truncated payloads, non-zero padding bits,
checksum failures, parameter/formula mismatches, and unsupported limits are
rejected. The old v4 binary layout is read only with `legacy=True`; the reader
never guesses an unknown file as legacy because doing so could restore a filter
that produces false negatives.

**Bloom frame layout (all integers are little-endian):**

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 4 | Magic `PBLF` |
| 4 | 1 | Format version (`1`) |
| 5 | 1 | Key codec (`1`: UTF-8 strings, identity bytes-like keys) |
| 6 | 1 | Integer byte order (`1`: little-endian) |
| 7 | 1 | Bit order (`1`: least-significant bit first) |
| 8 | 1 | Hash algorithm (`1` xxh128, `2` SHA-1, `3` SHA-256, `4` SHA-384, `5` SHA-512) |
| 9 | 2 | Hash algorithm version (`1`) |
| 11 | 1 | Checksum algorithm (`1`: IEEE CRC32) |
| 12 | 1 | Flags, must be zero |
| 13 | 1 | Reserved, must be zero |
| 14 | 2 | Header length (`92`) |
| 16 | 8 | Error rate, IEEE 754 double |
| 24 | 8 | Slice count (`k`) |
| 32 | 8 | Bits per slice |
| 40 | 8 | Capacity |
| 48 | 8 | Added-element count |
| 56 | 8 | Effective bit count |
| 64 | 4 | Salt payload length |
| 68 | 8 | Total payload length |
| 76 | 4 | CRC32 of the zeroed-CRC header followed by the payload |
| 80 | 12 | Reserved, must be zero |

The payload is the deterministic salt digests followed by
`ceil(effective_bit_count / 8)` bit bytes. Padding bits above the effective
bit count must be zero. For salt index `i`, the portable salt is
`H(H(pack_uint32_le(i)).digest()).digest()`, and a key digest is
`H(salt || encoded_key)`. Digest integers use little-endian unsigned 16-, 32-,
or 64-bit words according to the slice width.

**Scalable frame layout:** magic `PBSF`, version `1`, 64-byte little-endian
header, layer count, growth mode, tightening ratio, initial capacity, outer
error rate, payload length, CRC32, and a payload of complete portable Bloom
frames. v1 allows at most 256 layers and a 1 GiB payload; Bloom capacity uses
an unsigned 32-bit maximum and Bloom payloads also have a 1 GiB limit.

### Example 4: Understanding Error Rates

```python
import pybloom_live

# Create a filter with 0.1% error rate
f = pybloom_live.BloomFilter(capacity=1000, error_rate=0.001)

# Add exactly capacity elements
for i in range(1000):
    f.add(i)

# Test false positive rate
false_positives = 0
tests = 10000
for i in range(1000, 1000 + tests):
    if i in f:  # These were never added
        false_positives += 1

actual_error_rate = false_positives / tests
print(f"Requested error rate: {f.error_rate}")
print(f"Actual error rate: {actual_error_rate:.4f}")
# Actual rate should be close to 0.001
```

## API Reference

### BloomFilter

**Constructor:**
```python
BloomFilter(capacity, error_rate=0.001)
```
- `capacity`: Maximum number of elements
- `error_rate`: False positive probability (0 < error_rate < 1)

**Methods:**
- `add(key, skip_check=False)`: Add element; returns `True` if element might already exist
- `__contains__(key)`: Check membership using `in` operator
- `__len__()`: Return count of elements added
- `copy()`: Create a copy of the filter
- `union(other)` / `__or__`: Combine with another filter (bitwise OR)
- `intersection(other)` / `__and__`: Intersect with another filter (bitwise AND)
- `tofile(f)`: Serialize to file object
- `fromfile(f, n=-1, legacy=False)`: Read one portable v1 object, or one legacy v4 object when explicitly requested

**Properties:**
- `capacity`: Maximum capacity
- `error_rate`: Target false positive rate
- `count`: Number of elements added
- `num_bits`: Total bits in filter
- `num_slices`: Number of hash functions

### ScalableBloomFilter

**Constructor:**
```python
ScalableBloomFilter(initial_capacity=100, error_rate=0.001, mode=LARGE_SET_GROWTH)
```
- `initial_capacity`: Starting capacity
- `error_rate`: Target false positive probability
- `mode`: Growth mode (`SMALL_SET_GROWTH=2` or `LARGE_SET_GROWTH=4`)

**Methods:**
- `add(key)`: Add element to filter
- `__contains__(key)`: Check membership
- `__len__()`: Return total count across all internal filters
- `union(other)`: Combine with another scalable filter
- `tofile(f)` / `fromfile(f, legacy=False)`: Portable v1 serialization and explicit legacy v4 support

**Properties:**
- `capacity`: Total capacity (sum of all internal filters)
- `count`: Total elements added
- `error_rate`: Target error rate
- `filters`: List of internal `BloomFilter` instances

## Hash Functions

The library automatically selects the optimal hash function based on the number of bits required:

| Bits Required | Hash Function | Type |
|--------------|---------------|------|
| ≤ 128 | xxh3_128 | Non-cryptographic (fastest) |
| 129-160 | SHA-1 | Cryptographic |
| 161-256 | SHA-256 | Cryptographic |
| 257-384 | SHA-384 | Cryptographic |
| > 384 | SHA-512 | Cryptographic |

## Algorithm Details

This implementation follows the Scalable Bloom Filter algorithm from:

> P. Almeida, C.Baquero, N. Preguiça, D. Hutchison, **Scalable Bloom Filters**, (GLOBECOM 2007), IEEE, 2007.

**Key optimizations:**
- **Tightening ratio of 0.9**: Provides better average space usage across a wide range of growth patterns
- **Exponential growth**: New filters are 2x or 4x larger than the previous one
- **Error rate tightening**: Each new filter has a tighter error probability to maintain overall target rate
- **Optimal bit calculation**: Uses `M ≈ n × |ln(P)| / (ln(2)²)` where n is capacity and P is error rate

## Breaking Changes in Version 4.x

Version 4.0.0 introduces significant performance improvements but is **not backward compatible** with earlier versions:

- **Hash function change**: Replaced MD5 with xxHash (xxh3_128) for 128-bit hashes
- **Performance**: Significantly faster hash computation (see [benchmarks](https://github.com/joseph-fox/python-bloomfilter/pull/38))
- **Incompatibility**: Files saved with versions < 4.0 cannot be loaded with version 4.x

**Migration:** Filters from versions earlier than 4.0 cannot be read reliably and should be regenerated. The 4.0 raw layout can be read with `legacy=True`; new files use the checked, self-describing v1 format.

## Performance

Typical performance characteristics (results may vary by system):

- **Add operation**: ~500k-1M elements/second
- **Lookup operation**: ~700k-2M checks/second
- **Memory usage**: ~1.2 bits per element (with default 0.1% error rate)

## Python Version Support

- Python 3.6+
- Python 3.7+
- Python 3.8+
- Python 3.9+
- Python 3.10+
- Python 3.11+
- Python 3.12+

**Note:** Python 2 support has been dropped in version 4.0+. For Python 2 compatibility, use an earlier version.

## Development

We follow the [git branching model](http://nvie.com/posts/a-successful-git-branching-model/) described by Vincent Driessen.

**Running tests:**

```bash
pip install pytest
pytest pybloom_live/test_pybloom.py
```

## License

MIT License - see LICENSE.txt for details.

## Credits

- Original `pybloom` by Jay Baird
- Maintained and enhanced by the `pybloom_live` contributors
- Uses [xxHash](https://github.com/Cyan4973/xxHash) for high-performance hashing
