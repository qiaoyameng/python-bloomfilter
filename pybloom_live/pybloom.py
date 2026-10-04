"""Bloom Filter and Scalable Bloom Filter implementations.

This module implements two probabilistic data structures for space-efficient
set membership testing:

1. BloomFilter: Fixed-capacity filter for known dataset sizes
2. ScalableBloomFilter: Dynamically growing filter that scales automatically

Both implementations maintain a configurable false positive rate while never
producing false negatives. The module uses optimized hash functions (xxHash
for speed) and efficient bit array storage.

Mathematical Foundation:
    - Optimal hash functions: k = log2(1/P) where P is error rate
    - Optimal bit count: M ≈ n × |ln(P)| / (ln(2)²) where n is capacity
    - False positive probability: P ≈ (1 - e^(-kn/m))^k

Requirements:
    - Python 3.6+
    - bitarray >= 0.3.4: Efficient bit array operations
    - xxhash >= 3.0.0: Fast non-cryptographic hashing
"""
import copy
import hashlib
import math
from struct import Struct, calcsize, pack, unpack

import xxhash

try:
    import bitarray
except ImportError:
    raise ImportError('pybloom_live requires bitarray >= 0.3.4')


PORTABLE_MAGIC = b'PBLF'
PORTABLE_VERSION = 1
SCALABLE_PORTABLE_MAGIC = b'PSBF'
SCALABLE_PORTABLE_VERSION = 1

MAX_BLOOM_CAPACITY = 1 << 40
MAX_BLOOM_PAYLOAD_BYTES = 1 << 30
MAX_NUM_SLICES = 4096
MAX_SCALABLE_FILTERS = 1024
MAX_SCALABLE_PAYLOAD_BYTES = 1 << 40

KEY_ENCODING_UTF8_OR_BYTES = 1
UINT_ENCODING_LITTLE_ENDIAN = 1
BIT_ORDER_LSB_FIRST = 1
SALT_SCHEME_U32LE_DOUBLE_DIGEST = 1

HASH_ALGORITHM_XXH3_128 = 1
HASH_ALGORITHM_SHA1 = 2
HASH_ALGORITHM_SHA256 = 3
HASH_ALGORITHM_SHA384 = 4
HASH_ALGORITHM_SHA512 = 5

_HASH_ALGORITHMS = {
    HASH_ALGORITHM_XXH3_128: xxhash.xxh128,
    HASH_ALGORITHM_SHA1: hashlib.sha1,
    HASH_ALGORITHM_SHA256: hashlib.sha256,
    HASH_ALGORITHM_SHA384: hashlib.sha384,
    HASH_ALGORITHM_SHA512: hashlib.sha512,
}

_HASH_VERSIONS = {
    HASH_ALGORITHM_XXH3_128: b'XXH3_128:' + xxhash.XXHASH_VERSION.encode('ascii'),
    HASH_ALGORITHM_SHA1: b'SHA1:HASHLIB:1',
    HASH_ALGORITHM_SHA256: b'SHA256:HASHLIB:1',
    HASH_ALGORITHM_SHA384: b'SHA384:HASHLIB:1',
    HASH_ALGORITHM_SHA512: b'SHA512:HASHLIB:1',
}

_PORTABLE_HEADER = Struct(
    '<4sBHQdQQQIIBBBBBB16sI'
)
_SCALABLE_PORTABLE_HEADER = Struct('<4sBHQidQdI')


def _read_exact(f, size):
    """Read exactly size bytes, rejecting truncated streams."""
    chunks = []
    remaining = size
    while remaining:
        chunk = f.read(remaining)
        if not chunk:
            raise ValueError('Truncated Bloom filter data')
        if not isinstance(chunk, bytes):
            raise ValueError('Binary streams must return bytes')
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


class _BoundedReader:
    def __init__(self, stream, limit):
        self._stream = stream
        self._remaining = limit

    def read(self, size=-1):
        if size is None or size < 0:
            size = self._remaining
        size = min(size, self._remaining)
        if size == 0:
            return b''
        chunk = self._stream.read(size)
        if not chunk:
            return b''
        if len(chunk) > self._remaining:
            raise ValueError('Read exceeded Bloom filter object boundary')
        self._remaining -= len(chunk)
        return chunk


class _HashingReader:
    def __init__(self, stream, hasher):
        self._stream = stream
        self._hasher = hasher

    def read(self, size=-1):
        chunk = self._stream.read(size)
        if chunk:
            self._hasher.update(chunk)
        return chunk

    @property
    def remaining(self):
        return self._stream._remaining


def make_hashfuncs(num_slices, num_bits):
    """Create optimized hash functions for the Bloom filter.

    This function generates a hash function generator that produces the required
    number of hash values efficiently. It selects the appropriate hash algorithm
    based on the total bits needed, preferring faster non-cryptographic hashes
    when possible.

    Hash Function Selection Strategy:
        - xxHash (xxh128): For ≤128 bits (fastest, non-cryptographic)
        - SHA-1: For 129-160 bits
        - SHA-256: For 161-256 bits
        - SHA-384: For 257-384 bits
        - SHA-512: For >384 bits

    The function uses salting to generate multiple independent hash values from
    a single hash computation, significantly improving performance.

    Args:
        num_slices (int): Number of hash functions needed (k in literature).
            This equals the number of bits to set per element.
        num_bits (int): Number of bits per hash slice. Determines the format
            code for unpacking hash digests.

    Returns:
        tuple: A 2-tuple containing:
            - hash_maker (callable): Generator function that yields hash values
            - hashfn (callable): The underlying hash function used

    Performance Notes:
        - Salts are pre-computed and reused for all keys
        - Single hash computation generates multiple independent values
        - Format codes are optimized for the bit array size
    """
    # Select format code based on number of bits to optimize struct unpacking
    if num_bits >= (1 << 31):
        fmt_code, chunk_size = 'Q', 8  # 64-bit unsigned long long
    elif num_bits >= (1 << 15):
        fmt_code, chunk_size = 'I', 4  # 32-bit unsigned int
    else:
        fmt_code, chunk_size = 'H', 2  # 16-bit unsigned short

    # Calculate total bits needed and select appropriate hash function
    total_hash_bits = 8 * num_slices * chunk_size
    if total_hash_bits > 384:
        hash_id = HASH_ALGORITHM_SHA512
        hashfn = hashlib.sha512
    elif total_hash_bits > 256:
        hash_id = HASH_ALGORITHM_SHA384
        hashfn = hashlib.sha384
    elif total_hash_bits > 160:
        hash_id = HASH_ALGORITHM_SHA256
        hashfn = hashlib.sha256
    elif total_hash_bits > 128:
        hash_id = HASH_ALGORITHM_SHA1
        hashfn = hashlib.sha1
    else:
        hash_id = HASH_ALGORITHM_XXH3_128
        hashfn = xxhash.xxh128  # Fastest option for typical filters

    # Pre-compute format string for unpacking hash digests
    uint_count = hashfn().digest_size // chunk_size
    fmt = '<' + fmt_code * uint_count

    # Calculate number of salts needed to generate enough hash values
    num_salts, extra = divmod(num_slices, uint_count)
    if extra:
        num_salts += 1

    # Pre-compute salts for reuse (double-hashing for better distribution)
    salts = tuple(
        hashfn(hashfn(pack('<I', i)).digest()).digest()
        for i in range(num_salts)
    )

    def _hash_maker(key):
        """Generate hash values for a given key.

        This inner function handles key encoding and generates the required
        number of hash values using the pre-computed salts.

        Args:
            key: The element to hash (str, bytes, or any object with __str__)

        Yields:
            int: Hash values in range [0, num_bits)

        Performance Optimization:
            - Early termination after generating required number of hashes
            - Efficient modulo operation for range reduction
        """
        # Normalize key to bytes for consistent hashing (Python 3)
        if isinstance(key, str):
            key = key.encode('utf-8')
        elif isinstance(key, (bytes, bytearray, memoryview)):
            key = bytes(key)
        elif not isinstance(key, bytes):
            key = str(key).encode('utf-8')

        # Generate hash values using salted hashing
        i = 0
        for salt in salts:
            h = hashfn(salt)
            h.update(key)    # Hash the key
            # Unpack hash digest into multiple integers
            for uint in unpack(fmt, h.digest()):
                yield uint % num_bits  # Reduce to valid bit index
                i += 1
                if i >= num_slices:  # Early exit optimization
                    return

    _hash_maker.hash_id = hash_id
    _hash_maker.hashfn = hashfn
    _hash_maker.chunk_size = chunk_size
    _hash_maker.uint_count = uint_count
    _hash_maker.salts = salts
    _hash_maker.key_encoding = KEY_ENCODING_UTF8_OR_BYTES
    _hash_maker.uint_encoding = UINT_ENCODING_LITTLE_ENDIAN
    _hash_maker.bit_order = BIT_ORDER_LSB_FIRST
    _hash_maker.salt_scheme = SALT_SCHEME_U32LE_DOUBLE_DIGEST
    return _hash_maker, hashfn


class BloomFilter:
    FILE_FMT = b'<dQQQQ'

    def __init__(self, capacity, error_rate=0.001):
        """Initialize a Bloom filter with specified capacity and error rate.

        This constructor calculates the optimal number of hash functions and
        bit array size based on the desired capacity and false positive rate
        using standard Bloom filter formulas.

        Mathematical Formulas Used:
            - Number of hash functions: k = ceil(log2(1/P))
            - Bits per slice: m = ceil(n × |ln(P)| / (k × (ln(2))²))
            - Total bits: M = k × m

        Where:
            - n = capacity (number of elements)
            - P = error_rate (false positive probability)
            - k = num_slices (number of hash functions)
            - m = bits_per_slice

        Args:
            capacity (int): Maximum number of elements the filter should hold
                while maintaining the specified error rate. Must be > 0.
            error_rate (float, optional): Target false positive probability.
                Must be between 0 and 1 (exclusive). Default is 0.001 (0.1%).

        Raises:
            ValueError: If error_rate is not in range (0, 1).
            ValueError: If capacity is not positive.

        Example:
            >>> bf = BloomFilter(capacity=1000, error_rate=0.01)
            >>> bf.add("test")
            >>> "test" in bf
            True

        Performance Characteristics:
            - Memory usage: O(n × log(1/P)) bits
            - Time complexity for add/query: O(k) where k ≈ log2(1/P)

        Requires:
            Python 3.6+
        """
        if not (0 < error_rate < 1):
            raise ValueError("Error_Rate must be between 0 and 1.")
        if not capacity > 0:
            raise ValueError("Capacity must be > 0")
        if capacity > MAX_BLOOM_CAPACITY:
            raise ValueError("Capacity exceeds portable format limit")

        # Calculate optimal number of hash functions (k)
        # k = ceil(log2(1/P)) minimizes false positive rate
        num_slices = int(math.ceil(math.log(1.0 / error_rate, 2)))

        # Calculate optimal bits per slice (m)
        # Derived from: n ≈ M × (ln(2)²) / |ln(P)|
        # where M = k × m (total bits)
        bits_per_slice = int(math.ceil(
            (capacity * abs(math.log(error_rate))) /
            (num_slices * (math.log(2) ** 2))))

        num_bits = num_slices * bits_per_slice
        self._validate_parameters(
            error_rate, num_slices, bits_per_slice, num_bits, capacity, 0)
        self._setup(error_rate, num_slices, bits_per_slice, capacity, 0)
        self.bitarray = bitarray.bitarray(self.num_bits, endian='little')
        self.bitarray.setall(False)

    def _setup(self, error_rate, num_slices, bits_per_slice, capacity, count,
               hash_spec=None):
        """Internal setup method for initializing or restoring filter state.

        This method is used both during normal initialization and when
        deserializing a filter from a file. It configures all filter
        parameters and creates the hash function generator.

        Args:
            error_rate (float): Target false positive probability
            num_slices (int): Number of hash functions (k)
            bits_per_slice (int): Bits in each slice of the bit array
            capacity (int): Maximum number of elements
            count (int): Current number of elements stored
        """
        self.error_rate = error_rate
        self.num_slices = num_slices
        self.bits_per_slice = bits_per_slice
        self.capacity = capacity
        self.num_bits = num_slices * bits_per_slice
        self.count = count
        self._validate_parameters(
            error_rate, num_slices, bits_per_slice, self.num_bits,
            capacity, count)
        expected_hashes, self.hashfn = make_hashfuncs(
            self.num_slices, self.bits_per_slice)
        if hash_spec is None:
            self.make_hashes = expected_hashes
            return

        self.make_hashes = self._validate_hash_spec(
            expected_hashes, hash_spec, self.num_slices)

    @staticmethod
    def _validate_hash_spec(expected_hashes, hash_spec, num_slices):
        required_keys = {
            'hash_id', 'hash_version', 'chunk_size', 'uint_count',
            'salt_scheme', 'key_encoding', 'uint_encoding', 'bit_order',
            'salts'
        }
        if set(hash_spec) != required_keys:
            raise ValueError('Hash specification fields mismatch')
        if hash_spec['hash_id'] != expected_hashes.hash_id:
            raise ValueError('Hash algorithm mismatch')
        if hash_spec['hash_version'] != _HASH_VERSIONS[expected_hashes.hash_id]:
            raise ValueError('Hash algorithm version mismatch')
        if hash_spec['chunk_size'] != expected_hashes.chunk_size:
            raise ValueError('Hash integer width mismatch')
        if hash_spec['uint_count'] != expected_hashes.uint_count:
            raise ValueError('Hash integer count mismatch')
        if hash_spec['salt_scheme'] != expected_hashes.salt_scheme:
            raise ValueError('Salt construction mismatch')
        if hash_spec['key_encoding'] != expected_hashes.key_encoding:
            raise ValueError('Key encoding mismatch')
        if hash_spec['uint_encoding'] != expected_hashes.uint_encoding:
            raise ValueError('Integer byte order mismatch')
        if hash_spec['bit_order'] != expected_hashes.bit_order:
            raise ValueError('Bit order mismatch')

        salts = hash_spec['salts']
        if not isinstance(salts, tuple) or salts != expected_hashes.salts:
            raise ValueError('Salt values mismatch')
        required_salts = (
            (num_slices + expected_hashes.uint_count - 1) //
            expected_hashes.uint_count)
        if len(salts) != required_salts:
            raise ValueError('Salt count mismatch')
        return expected_hashes

    def __contains__(self, key):
        """Test whether an element is in the Bloom filter.

        This method implements the membership test operation, allowing use of
        the 'in' operator. It checks if all k hash positions are set in the
        bit array.

        Args:
            key: The element to test for membership (any hashable type)

        Returns:
            bool: True if the element might be in the set (with possible false
                positives), False if the element is definitely not in the set
                (no false negatives).

        Time Complexity:
            O(k) where k is the number of hash functions

        Example:
            >>> bf = BloomFilter(100, 0.01)
            >>> bf.add("apple")
            >>> "apple" in bf
            True
            >>> "banana" in bf
            False

        Performance Optimization:
            - Local variable caching for faster attribute access
            - Early exit on first unset bit (short-circuit evaluation)
        """
        # Cache attributes as local variables for faster access in loop
        bits_per_slice = self.bits_per_slice
        bitarray = self.bitarray
        hashes = self.make_hashes(key)

        # Check all hash positions; return False immediately if any is unset
        offset = 0
        for k in hashes:
            if not bitarray[offset + k]:
                return False  # Definitely not in set
            offset += bits_per_slice
        return True  # Probably in set

    def __len__(self):
        """Return the number of elements added to this Bloom filter.

        Note that this is a count of additions, not unique elements. If the
        same element is added multiple times, it contributes to the count
        each time (unless detected as duplicate).

        Returns:
            int: Number of elements that have been added to the filter

        Example:
            >>> bf = BloomFilter(100, 0.01)
            >>> bf.add("apple")
            >>> bf.add("banana")
            >>> len(bf)
            2
        """
        return self.count

    def add(self, key, skip_check=False):
        """Add an element to the Bloom filter.

        This method sets the appropriate bits in the bit array based on the
        hash values for the key. It can optionally detect if the element was
        likely already present before adding it.

        Args:
            key: The element to add (any hashable type)
            skip_check (bool, optional): If True, skips the duplicate detection
                check for better performance. Use when you know elements are
                unique or don't care about detecting duplicates. Default is False.

        Returns:
            bool: True if the element might have already been in the filter
                before this operation (possible false positive), False if this
                is definitely a new element.

        Raises:
            IndexError: If the filter is already at or over capacity. Adding
                elements beyond capacity significantly increases false positive
                rate.

        Time Complexity:
            O(k) where k is the number of hash functions

        Example:
            >>> bf = BloomFilter(100, 0.01)
            >>> bf.add("apple")  # Returns False (new element)
            False
            >>> bf.add("apple")  # Returns True (duplicate)
            True
            >>> bf.add("banana", skip_check=True)  # Returns False (skipped check)
            False

        Performance Note:
            Use skip_check=True when adding bulk data where you know there are
            no duplicates. This can improve performance by 30-40%.
        """
        # Check capacity before computing hashes (fail-fast optimization)
        if self.count >= self.capacity:
            raise IndexError("BloomFilter is at capacity")

        # Cache attributes as local variables for faster loop access
        bitarray = self.bitarray
        bits_per_slice = self.bits_per_slice
        hashes = self.make_hashes(key)

        # Optimized path for skip_check=True (bulk insertions)
        if skip_check:
            offset = 0
            for k in hashes:
                bitarray[offset + k] = True
                offset += bits_per_slice
            self.count += 1
            return False

        # Check for duplicate and set bits in single pass
        found_all_bits = True
        offset = 0
        for k in hashes:
            if found_all_bits and not bitarray[offset + k]:
                found_all_bits = False  # Found at least one unset bit
            bitarray[offset + k] = True
            offset += bits_per_slice

        # Only increment count if this is a new element
        if not found_all_bits:
            self.count += 1
            return False
        else:
            return True  # Element was likely already present

    def copy(self):
        """Create a deep copy of this Bloom filter.

        The copy is completely independent and can be modified without
        affecting the original filter.

        Returns:
            BloomFilter: A new BloomFilter instance with the same state

        Time Complexity:
            O(M) where M is the size of the bit array

        Example:
            >>> bf1 = BloomFilter(100, 0.01)
            >>> bf1.add("apple")
            >>> bf2 = bf1.copy()
            >>> bf2.add("banana")
            >>> "banana" in bf1
            False
            >>> "banana" in bf2
            True
        """
        new_filter = BloomFilter(self.capacity, self.error_rate)
        new_filter.bitarray = self.bitarray.copy()
        new_filter.count = self.count  # Preserve count
        return new_filter

    def union(self, other):
        """Calculate the union of two Bloom filters.

        Creates a new Bloom filter containing elements that are in either this
        filter or the other filter. This is equivalent to the bitwise OR
        operation on the underlying bit arrays.

        Args:
            other (BloomFilter): Another Bloom filter to union with this one.
                Must have the same capacity and error rate.

        Returns:
            BloomFilter: A new filter representing the union of both filters

        Raises:
            ValueError: If filters have different capacities or error rates

        Time Complexity:
            O(M) where M is the size of the bit array

        Example:
            >>> bf1 = BloomFilter(100, 0.01)
            >>> bf2 = BloomFilter(100, 0.01)
            >>> bf1.add("apple")
            >>> bf2.add("banana")
            >>> bf_union = bf1.union(bf2)  # or bf1 | bf2
            >>> "apple" in bf_union and "banana" in bf_union
            True

        Note:
            The count in the resulting filter may not be accurate as it cannot
            account for elements present in both filters.
        """
        if self.capacity != other.capacity or self.error_rate != other.error_rate:
            raise ValueError(
                "Unioning filters requires both filters to have both the same capacity and error rate")
        new_bloom = self.copy()
        new_bloom.bitarray = new_bloom.bitarray | other.bitarray
        # Note: Count is not accurately updated as we can't determine overlaps
        return new_bloom

    def __or__(self, other):
        """Implement the | operator for union operation.

        Example:
            >>> bf_union = bf1 | bf2
        """
        return self.union(other)

    def intersection(self, other):
        """Calculate the intersection of two Bloom filters.

        Creates a new Bloom filter containing elements that are likely in both
        this filter and the other filter. This is equivalent to the bitwise AND
        operation on the underlying bit arrays.

        Args:
            other (BloomFilter): Another Bloom filter to intersect with this one.
                Must have the same capacity and error rate.

        Returns:
            BloomFilter: A new filter representing the intersection of both filters

        Raises:
            ValueError: If filters have different capacities or error rates

        Time Complexity:
            O(M) where M is the size of the bit array

        Example:
            >>> bf1 = BloomFilter(100, 0.01)
            >>> bf2 = BloomFilter(100, 0.01)
            >>> bf1.add("apple")
            >>> bf1.add("banana")
            >>> bf2.add("banana")
            >>> bf_intersect = bf1.intersection(bf2)  # or bf1 & bf2
            >>> "banana" in bf_intersect  # Likely True
            True
            >>> "apple" in bf_intersect   # Likely False
            False

        Note:
            Due to false positives, the intersection may contain elements that
            were not actually in both original sets. The false positive rate of
            the intersection can be higher than the individual filters.
        """
        if self.capacity != other.capacity or self.error_rate != other.error_rate:
            raise ValueError(
                "Intersecting filters requires both filters to have equal capacity and error rate")
        new_bloom = self.copy()
        new_bloom.bitarray = new_bloom.bitarray & other.bitarray
        # Note: Count is not accurately updated for intersections
        return new_bloom

    def __and__(self, other):
        """Implement the & operator for intersection operation.

        Example:
            >>> bf_intersect = bf1 & bf2
        """
        return self.intersection(other)

    def tofile(self, f):
        """Serialize to the current portable, versioned Bloom format."""
        header, payload, checksum = self._portable_parts()
        f.write(header)
        f.write(payload)
        f.write(checksum)

    def _portable_parts(self):
        hashes = self.make_hashes
        hash_version = _HASH_VERSIONS[hashes.hash_id]
        if len(hash_version) > 16:
            raise ValueError('Hash version identifier is too long')
        salts = b''.join(hashes.salts)
        payload = self.bitarray.tobytes()
        self._validate_state(len(payload))
        header_len = _PORTABLE_HEADER.size + len(salts)

        fixed = _PORTABLE_HEADER.pack(
            PORTABLE_MAGIC, PORTABLE_VERSION, header_len, len(payload),
            self.error_rate, self.capacity, self.count, self.num_bits,
            self.num_slices, self.bits_per_slice,
            hashes.key_encoding, hashes.hash_id, hashes.uint_encoding,
            hashes.bit_order, hashes.chunk_size, hashes.salt_scheme,
            hash_version.ljust(16, b'\x00'), len(salts))
        header = fixed + salts
        checksum = hashlib.sha256(header + payload).digest()
        return header, payload, checksum

    @staticmethod
    def _validate_parameters(error_rate, num_slices, bits_per_slice,
                             num_bits, capacity, count):
        if not math.isfinite(error_rate) or not 0 < error_rate < 1:
            raise ValueError('Invalid error rate')
        if not 0 < num_slices <= MAX_NUM_SLICES:
            raise ValueError('Unsupported number of slices')
        if bits_per_slice <= 0:
            raise ValueError('Invalid slice width')
        if num_slices * bits_per_slice != num_bits or num_bits <= 0:
            raise ValueError('Bit length mismatch')
        if (num_bits + 7) // 8 > MAX_BLOOM_PAYLOAD_BYTES:
            raise ValueError('Payload length exceeds limit')
        if not 0 < capacity <= MAX_BLOOM_CAPACITY:
            raise ValueError('Unsupported capacity')
        if not 0 <= count <= capacity:
            raise ValueError('Invalid stored count')

    def _validate_state(self, payload_len):
        self._validate_parameters(
            self.error_rate, self.num_slices, self.bits_per_slice,
            self.num_bits, self.capacity, self.count)
        expected_payload_len = (self.num_bits + 7) // 8
        if payload_len != expected_payload_len:
            raise ValueError('Bit length mismatch')
        if payload_len > MAX_BLOOM_PAYLOAD_BYTES:
            raise ValueError('Payload length exceeds limit')

    @classmethod
    def fromfile(cls, f, n=-1, legacy=False):
        """Read one self-delimited Bloom object from a binary stream."""
        if n == 0 or 0 < n < 4:
            raise ValueError('n too small!')
        stream = _BoundedReader(f, n) if n > 0 else f
        magic = _read_exact(stream, 4)
        if magic == PORTABLE_MAGIC:
            return cls._from_portable(stream, magic)
        if legacy:
            return cls._from_legacy(stream, magic, n)
        raise ValueError('Unrecognized Bloom filter format; enable legacy reading explicitly')

    @classmethod
    def _from_portable(cls, f, magic):
        fixed = magic + _read_exact(f, _PORTABLE_HEADER.size - 4)
        values = _PORTABLE_HEADER.unpack(fixed)
        (_magic, version, header_len, payload_len, error_rate, capacity,
         count, num_bits, num_slices, bits_per_slice, key_encoding, hash_id,
         uint_encoding, bit_order, chunk_size, salt_scheme, hash_version,
         salt_len) = values

        if version != PORTABLE_VERSION:
            raise ValueError('Unsupported portable Bloom filter version')
        if header_len < _PORTABLE_HEADER.size:
            raise ValueError('Invalid portable header length')
        if header_len - _PORTABLE_HEADER.size != salt_len:
            raise ValueError('Salt length mismatch')
        if payload_len > MAX_BLOOM_PAYLOAD_BYTES:
            raise ValueError('Payload length exceeds limit')

        cls._validate_parameters(
            error_rate, num_slices, bits_per_slice, num_bits, capacity, count)
        if payload_len != (num_bits + 7) // 8:
            raise ValueError('Bit length mismatch')
        if key_encoding != KEY_ENCODING_UTF8_OR_BYTES:
            raise ValueError('Unsupported key encoding')
        if uint_encoding != UINT_ENCODING_LITTLE_ENDIAN:
            raise ValueError('Unsupported integer byte order')
        if bit_order != BIT_ORDER_LSB_FIRST:
            raise ValueError('Unsupported bit order')
        if salt_scheme != SALT_SCHEME_U32LE_DOUBLE_DIGEST:
            raise ValueError('Unsupported salt construction')
        if hash_id not in _HASH_ALGORITHMS:
            raise ValueError('Unsupported hash algorithm')

        if chunk_size not in (2, 4, 8):
            raise ValueError('Unsupported integer width')
        if bits_per_slice >= (1 << 31):
            expected_chunk_size = 8
        elif bits_per_slice >= (1 << 15):
            expected_chunk_size = 4
        else:
            expected_chunk_size = 2
        if chunk_size != expected_chunk_size:
            raise ValueError('Hash integer width mismatch')

        expected_hashfn = _HASH_ALGORITHMS[hash_id]
        digest_size = expected_hashfn().digest_size
        expected_version = _HASH_VERSIONS[hash_id]
        if hash_version != expected_version.ljust(16, b'\x00'):
            raise ValueError('Hash algorithm or version mismatch')
        if hash_version[len(expected_version):] != b'\x00' * (16 - len(expected_version)):
            raise ValueError('Invalid hash version padding')

        salt_bytes = _read_exact(f, salt_len)
        expected_salt_count = (
            (num_slices + digest_size // chunk_size - 1) //
            (digest_size // chunk_size))
        if salt_len != expected_salt_count * digest_size:
            raise ValueError('Invalid salt data length')
        salts = tuple(
            salt_bytes[start:start + digest_size]
            for start in range(0, salt_len, digest_size))

        payload = _read_exact(f, payload_len)
        padding = num_bits % 8
        if padding and payload[-1] & (0xFF << padding):
            raise ValueError('Non-zero padding bits')

        restored = cls.__new__(cls)
        hash_spec = {
            'hash_id': hash_id,
            'hash_version': expected_version,
            'chunk_size': chunk_size,
            'uint_count': digest_size // chunk_size,
            'salt_scheme': salt_scheme,
            'key_encoding': key_encoding,
            'uint_encoding': uint_encoding,
            'bit_order': bit_order,
            'salts': salts,
        }
        restored._setup(error_rate, num_slices, bits_per_slice, capacity,
                        count, hash_spec)
        restored.bitarray = bitarray.bitarray(endian='little')
        restored.bitarray.frombytes(payload)
        del restored.bitarray[num_bits:]

        checksum = _read_exact(f, hashlib.sha256().digest_size)
        if checksum != hashlib.sha256(fixed + salt_bytes + payload).digest():
            raise ValueError('Bloom filter checksum mismatch')
        return restored

    @classmethod
    def _from_legacy(cls, f, magic, n):
        headerlen = calcsize(cls.FILE_FMT)
        header = magic + _read_exact(f, headerlen - 4)
        error_rate, num_slices, bits_per_slice, capacity, count = unpack(
            cls.FILE_FMT, header)
        restored = cls.__new__(cls)
        restored._setup(error_rate, num_slices, bits_per_slice, capacity, count)
        payload_len = (restored.num_bits + 7) // 8
        if n > 0 and n - headerlen < payload_len:
            raise ValueError('Truncated Bloom filter data')
        payload = _read_exact(f, payload_len)
        restored.bitarray = bitarray.bitarray(endian='little')
        restored.bitarray.frombytes(payload)
        del restored.bitarray[restored.num_bits:]
        return restored

    def __getstate__(self):
        """Prepare the filter state for pickling.

        Removes the hash function generator which cannot be pickled, as it
        will be reconstructed during unpickling.

        Returns:
            dict: Filter state dictionary without the make_hashes function
        """
        d = self.__dict__.copy()
        del d['make_hashes']  # Functions can't be pickled
        return d

    def __setstate__(self, d):
        """Restore the filter state from pickling.

        Reconstructs the hash function generator that was removed during
        pickling.

        Args:
            d (dict): Filter state dictionary from __getstate__
        """
        self.__dict__.update(d)
        # Reconstruct hash function generator
        self.make_hashes, self.hashfn = make_hashfuncs(self.num_slices, self.bits_per_slice)


class ScalableBloomFilter:
    """A Bloom filter that automatically scales as more elements are added.

    This implementation follows the algorithm described in:
    "Scalable Bloom Filters" by Almeida et al., GLOBECOM 2007.

    The filter automatically creates additional internal Bloom filters as
    capacity is reached, with each new filter having:
    - Exponentially larger capacity (2x or 4x growth)
    - Tighter error rate (multiplied by tightening ratio 0.9)

    This maintains overall false positive probability while accommodating
    unbounded growth.

    Class Attributes:
        SMALL_SET_GROWTH (int): Growth factor of 2 - slower growth, less memory
        LARGE_SET_GROWTH (int): Growth factor of 4 - faster growth (default)
        FILE_FMT (str): Binary format string for serialization
    """
    SMALL_SET_GROWTH = 2  # Growth factor for memory-constrained scenarios
    LARGE_SET_GROWTH = 4  # Growth factor for performance (default)
    FILE_FMT = '<idQd'

    def __init__(self, initial_capacity=100, error_rate=0.001,
                 mode=LARGE_SET_GROWTH):
        """Initialize a Scalable Bloom Filter.

        Args:
            initial_capacity (int, optional): Initial capacity of the first
                internal filter. Default is 100. Subsequent filters grow
                exponentially based on the mode.
            error_rate (float, optional): Target overall false positive
                probability. Must be > 0. Default is 0.001 (0.1%). Each
                internal filter uses a tighter error rate (× 0.9) to maintain
                this overall rate.
            mode (int, optional): Growth rate for new filters. Must be either
                SMALL_SET_GROWTH (2) or LARGE_SET_GROWTH (4). Default is
                LARGE_SET_GROWTH.

        Raises:
            ValueError: If error_rate is not positive

        Example:
            >>> sbf = ScalableBloomFilter(
            ...     initial_capacity=1000,
            ...     error_rate=0.001,
            ...     mode=ScalableBloomFilter.LARGE_SET_GROWTH
            ... )
            >>> for i in range(10000):
            ...     sbf.add(i)  # Automatically scales
            >>> len(sbf)
            10000

        Performance Notes:
            - SMALL_SET_GROWTH: Better for memory-constrained scenarios
            - LARGE_SET_GROWTH: Better for performance when memory allows
            - Tightening ratio of 0.9 balances space and accuracy

        Requires:
            Python 3.6+
        """
        if not error_rate or error_rate < 0:
            raise ValueError("Error_Rate must be a decimal less than 0.")
        self._setup(mode, 0.9, initial_capacity, error_rate)
        self.filters = []

    def _setup(self, mode, ratio, initial_capacity, error_rate):
        """Internal setup method for initializing or restoring filter state.

        Args:
            mode (int): Growth factor (SMALL_SET_GROWTH or LARGE_SET_GROWTH)
            ratio (float): Tightening ratio for error rates (typically 0.9)
            initial_capacity (int): Starting capacity
            error_rate (float): Target false positive probability
        """
        self.scale = mode
        self.ratio = ratio
        self.initial_capacity = initial_capacity
        self.error_rate = error_rate

    def __contains__(self, key):
        """Test whether an element is in the Scalable Bloom filter.

        Checks membership across all internal filters, searching from newest
        to oldest. This ordering optimizes for recently added elements.

        Args:
            key: The element to test for membership

        Returns:
            bool: True if element might be in the set, False if definitely not

        Time Complexity:
            O(k × n) where k is hash functions per filter and n is number of
            internal filters. In practice, most lookups hit the most recent
            filters quickly.

        Example:
            >>> sbf = ScalableBloomFilter()
            >>> sbf.add("test")
            >>> "test" in sbf
            True
            >>> "not_added" in sbf
            False

        Performance Optimization:
            - Searches filters in reverse order (newest first)
            - Early exit on first match
        """
        # Search newest filters first (most likely to contain recent additions)
        for f in reversed(self.filters):
            if key in f:
                return True
        return False

    def add(self, key):
        """Add an element to the Scalable Bloom filter.

        Automatically creates new internal filters as needed when capacity is
        reached. New filters are exponentially larger with tighter error rates
        to maintain the overall target false positive probability.

        Args:
            key: The element to add (any hashable type)

        Returns:
            bool: True if element was already in the filter (possible false
                positive), False if this is a new addition

        Time Complexity:
            O(k) for normal adds, O(k × n) when checking for duplicates where
            k is hash functions and n is number of internal filters

        Example:
            >>> sbf = ScalableBloomFilter(initial_capacity=100)
            >>> sbf.add("apple")  # Returns False (new)
            False
            >>> sbf.add("apple")  # Returns True (duplicate)
            True
            >>> # Automatically scales beyond initial capacity
            >>> for i in range(1000):
            ...     sbf.add(i)

        Performance Note:
            The duplicate check (key in self) adds overhead. For bulk loading
            of known-unique data, consider modifying to skip this check.
        """
        # Check for duplicates across all filters
        if key in self:
            return True

        # Create first filter if needed
        if not self.filters:
            filter = BloomFilter(
                capacity=self.initial_capacity,
                error_rate=self.error_rate * self.ratio)
            self.filters.append(filter)
        else:
            # Check if current filter is at capacity
            filter = self.filters[-1]
            if filter.count >= filter.capacity:
                # Create new filter with exponential growth and tighter error rate
                filter = BloomFilter(
                    capacity=filter.capacity * self.scale,
                    error_rate=filter.error_rate * self.ratio)
                self.filters.append(filter)

        # Add to most recent filter (skip_check=True since we already checked above)
        filter.add(key, skip_check=True)
        return False

    def union(self, other):
        """Calculate the union of two Scalable Bloom filters.

        Creates a new Scalable Bloom filter containing elements from either
        filter. The filters must have matching parameters (mode, initial
        capacity, error rate).

        Args:
            other (ScalableBloomFilter): Another ScalableBloomFilter to union
                with this one. Must have the same mode, initial_capacity, and
                error_rate.

        Returns:
            ScalableBloomFilter: A new filter representing the union

        Raises:
            ValueError: If filters have incompatible parameters

        Time Complexity:
            O(M × n) where M is bit array size and n is number of filters

        Example:
            >>> sbf1 = ScalableBloomFilter(initial_capacity=100)
            >>> sbf2 = ScalableBloomFilter(initial_capacity=100)
            >>> sbf1.add("apple")
            >>> sbf2.add("banana")
            >>> sbf_union = sbf1.union(sbf2)  # or sbf1 | sbf2
            >>> "apple" in sbf_union and "banana" in sbf_union
            True

        Implementation Details:
            - Unions corresponding internal filters using bitwise OR
            - Handles filters with different numbers of internal filters
            - Deep copies the larger filter to avoid modifying originals
        """
        if self.scale != other.scale or \
                self.initial_capacity != other.initial_capacity or \
                self.error_rate != other.error_rate:
            raise ValueError("Unioning two scalable bloom filters requires "
                           "both filters to have both the same mode, initial capacity and error rate")

        # Deep copy the larger filter to preserve more data
        if len(self.filters) > len(other.filters):
            larger_sbf = copy.deepcopy(self)
            smaller_sbf = other
        else:
            larger_sbf = copy.deepcopy(other)
            smaller_sbf = self

        # Union corresponding internal filters
        new_filters = []
        for i in range(len(smaller_sbf.filters)):
            new_filter = larger_sbf.filters[i] | smaller_sbf.filters[i]
            new_filters.append(new_filter)

        # Append remaining filters from larger filter
        for i in range(len(smaller_sbf.filters), len(larger_sbf.filters)):
            new_filters.append(larger_sbf.filters[i])

        larger_sbf.filters = new_filters
        return larger_sbf

    def __or__(self, other):
        """Implement the | operator for union operation.

        Example:
            >>> sbf_union = sbf1 | sbf2
        """
        return self.union(other)

    @property
    def capacity(self):
        """Get the total capacity across all internal filters.

        Returns:
            int: Sum of capacities of all internal BloomFilter instances

        Example:
            >>> sbf = ScalableBloomFilter(initial_capacity=100)
            >>> for i in range(500):
            ...     sbf.add(i)
            >>> sbf.capacity  # Will be > 100 due to scaling
            700
        """
        return sum(f.capacity for f in self.filters)

    @property
    def count(self):
        """Get the total number of elements added.

        This is an alias for len() provided for API consistency with
        BloomFilter.

        Returns:
            int: Total number of elements across all internal filters

        Example:
            >>> sbf = ScalableBloomFilter()
            >>> sbf.add("a")
            >>> sbf.add("b")
            >>> sbf.count
            2
        """
        return len(self)

    def tofile(self, f):
        """Serialize to the current portable scalable Bloom format."""
        self._validate_container()
        payload_parts = []
        for inner in self.filters:
            header, payload, checksum = inner._portable_parts()
            payload_parts.append(header + payload + checksum)
        payload = b''.join(payload_parts)
        if len(payload) > MAX_SCALABLE_PAYLOAD_BYTES:
            raise ValueError('Payload length exceeds limit')

        header = _SCALABLE_PORTABLE_HEADER.pack(
            SCALABLE_PORTABLE_MAGIC, SCALABLE_PORTABLE_VERSION,
            _SCALABLE_PORTABLE_HEADER.size, len(payload), self.scale,
            self.ratio, self.initial_capacity, self.error_rate,
            len(self.filters))
        f.write(header)
        f.write(payload)
        f.write(hashlib.sha256(header + payload).digest())

    def _validate_container(self):
        if self.scale not in (self.SMALL_SET_GROWTH, self.LARGE_SET_GROWTH):
            raise ValueError('Invalid growth mode')
        if not math.isfinite(self.ratio) or self.ratio <= 0 or self.ratio >= 1:
            raise ValueError('Invalid tightening ratio')
        if not 0 < self.initial_capacity <= MAX_BLOOM_CAPACITY:
            raise ValueError('Unsupported initial capacity')
        if not math.isfinite(self.error_rate) or not 0 < self.error_rate < 1:
            raise ValueError('Invalid error rate')
        if not 0 <= len(self.filters) <= MAX_SCALABLE_FILTERS:
            raise ValueError('Unsupported number of filters')

    @classmethod
    def fromfile(cls, f, legacy=False):
        """Read one self-delimited scalable Bloom object."""
        magic = _read_exact(f, 4)
        if magic == SCALABLE_PORTABLE_MAGIC:
            return cls._from_portable(f, magic)
        if legacy:
            return cls._from_legacy(f, magic)
        raise ValueError(
            'Unrecognized scalable Bloom filter format; enable legacy reading explicitly')

    @classmethod
    def _from_portable(cls, f, magic):
        header = magic + _read_exact(f, _SCALABLE_PORTABLE_HEADER.size - 4)
        (_magic, version, header_len, payload_len, scale, ratio,
         initial_capacity, error_rate, nfilters) = \
            _SCALABLE_PORTABLE_HEADER.unpack(header)

        if version != SCALABLE_PORTABLE_VERSION:
            raise ValueError('Unsupported portable scalable Bloom filter version')
        if header_len != _SCALABLE_PORTABLE_HEADER.size:
            raise ValueError('Invalid scalable header length')
        if payload_len > MAX_SCALABLE_PAYLOAD_BYTES:
            raise ValueError('Payload length exceeds limit')
        if scale not in (cls.SMALL_SET_GROWTH, cls.LARGE_SET_GROWTH):
            raise ValueError('Invalid growth mode')
        if not math.isfinite(ratio) or ratio <= 0 or ratio >= 1:
            raise ValueError('Invalid tightening ratio')
        if not 0 < initial_capacity <= MAX_BLOOM_CAPACITY:
            raise ValueError('Unsupported initial capacity')
        if not math.isfinite(error_rate) or not 0 < error_rate < 1:
            raise ValueError('Invalid error rate')
        if not 0 <= nfilters <= MAX_SCALABLE_FILTERS:
            raise ValueError('Unsupported number of filters')
        if nfilters == 0 and payload_len != 0:
            raise ValueError('Invalid empty scalable payload')

        payload_hasher = hashlib.sha256(header)
        payload_stream = _HashingReader(
            _BoundedReader(f, payload_len), payload_hasher)
        restored = cls.__new__(cls)
        restored._setup(scale, ratio, initial_capacity, error_rate)
        restored.filters = []
        expected_capacity = initial_capacity
        expected_error_rate = error_rate * ratio

        for _ in range(nfilters):
            inner = BloomFilter.fromfile(payload_stream)
            if inner.capacity != expected_capacity:
                raise ValueError('Scalable filter capacity mismatch')
            if inner.error_rate != expected_error_rate:
                raise ValueError('Scalable filter error rate mismatch')
            restored.filters.append(inner)
            expected_capacity *= scale
            expected_error_rate *= ratio

        if payload_stream.remaining != 0:
            raise ValueError('Scalable payload has trailing bytes')
        checksum = _read_exact(f, hashlib.sha256().digest_size)
        if checksum != payload_hasher.digest():
            raise ValueError('Scalable Bloom filter checksum mismatch')
        return restored

    @classmethod
    def _from_legacy(cls, f, magic):
        headerlen = calcsize(cls.FILE_FMT)
        header = magic + _read_exact(f, headerlen - 4)
        scale, ratio, initial_capacity, error_rate = unpack(cls.FILE_FMT, header)
        count_bytes = _read_exact(f, calcsize(b'<l'))
        nfilters, = unpack(b'<l', count_bytes)
        if nfilters < 0 or nfilters > MAX_SCALABLE_FILTERS:
            raise ValueError('Unsupported number of filters')

        restored = cls.__new__(cls)
        restored._setup(scale, ratio, initial_capacity, error_rate)
        restored.filters = []
        if nfilters == 0:
            return restored

        size_table = _read_exact(f, calcsize(b'Q') * nfilters)
        filter_lengths = unpack(b'<' + b'Q' * nfilters, size_table)
        if any(length > MAX_BLOOM_PAYLOAD_BYTES * 2 for length in filter_lengths):
            raise ValueError('Legacy Bloom filter length exceeds limit')
        for filter_len in filter_lengths:
            restored.filters.append(BloomFilter.fromfile(f, filter_len, legacy=True))
        return restored

    def __len__(self):
        """Return the total number of elements across all internal filters.

        Returns:
            int: Sum of counts from all internal BloomFilter instances

        Time Complexity:
            O(n) where n is the number of internal filters

        Example:
            >>> sbf = ScalableBloomFilter()
            >>> for i in range(100):
            ...     sbf.add(i)
            >>> len(sbf)
            100
        """
        return sum(f.count for f in self.filters)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
