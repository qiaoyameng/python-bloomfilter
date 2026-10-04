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
import zlib
from io import BytesIO
from struct import Struct, calcsize, pack, unpack

import xxhash

try:
    import bitarray
except ImportError:
    raise ImportError('pybloom_live requires bitarray >= 0.3.4')


PORTABLE_MAGIC = b'PBLF'
PORTABLE_VERSION = 1
PORTABLE_SCALABLE_MAGIC = b'PBSF'
PORTABLE_SCALABLE_VERSION = 1
KEY_CODEC_UTF8_OTHER = 1
BYTE_ORDER_LITTLE = 1
BIT_ORDER_LSB_FIRST = 1
CHECKSUM_CRC32 = 1
MAX_NUM_SLICES = 256
MAX_CAPACITY = (1 << 32) - 1
MAX_BLOOM_PAYLOAD_BYTES = 1 << 30
MAX_SCALABLE_PAYLOAD_BYTES = 1 << 30
MAX_SCALABLE_LAYERS = 256

_BLOOM_HEADER = Struct('<4sBBBBBHBBBHdQQQQQIQ4s12x')
_SCALABLE_HEADER = Struct('<4sBBHBBHIdQdQ4s12x')
_BLOOM_HEADER_LEN = _BLOOM_HEADER.size
_SCALABLE_HEADER_LEN = _SCALABLE_HEADER.size
_BLOOM_CRC_OFFSET = _BLOOM_HEADER_LEN - 16
_SCALABLE_CRC_OFFSET = _SCALABLE_HEADER_LEN - 16


def _read_exact(f, size, truncated_message):
    data = f.read(size)
    if data is None or len(data) != size:
        raise ValueError(truncated_message)
    return data


def _crc32(data):
    return zlib.crc32(data) & 0xffffffff


def _hash_configuration(num_slices, num_bits):
    if num_bits >= (1 << 31):
        fmt_code, chunk_size = 'Q', 8
    elif num_bits >= (1 << 15):
        fmt_code, chunk_size = 'I', 4
    else:
        fmt_code, chunk_size = 'H', 2

    total_hash_bits = 8 * num_slices * chunk_size
    if total_hash_bits > 384:
        hashfn = hashlib.sha512
        hash_algorithm = 5
    elif total_hash_bits > 256:
        hashfn = hashlib.sha384
        hash_algorithm = 4
    elif total_hash_bits > 160:
        hashfn = hashlib.sha256
        hash_algorithm = 3
    elif total_hash_bits > 128:
        hashfn = hashlib.sha1
        hash_algorithm = 2
    else:
        hashfn = xxhash.xxh128
        hash_algorithm = 1

    fmt = '<' + fmt_code * (hashfn().digest_size // chunk_size)
    digest_count = calcsize(fmt) // chunk_size
    num_salts, extra = divmod(num_slices, digest_count)
    if extra:
        num_salts += 1

    salts = tuple(
        hashfn(hashfn(pack('<I', salt_index)).digest()).digest()
        for salt_index in range(num_salts)
    )
    return hash_algorithm, hashfn, fmt, chunk_size, salts


def _encode_key(key):
    if isinstance(key, str):
        return key.encode('utf-8')
    if isinstance(key, (bytes, bytearray, memoryview)):
        return bytes(key)
    return str(key).encode('utf-8')


def _hash_maker_for(num_slices, num_bits, salts, hashfn, fmt):
    def _hash_maker(key):
        key = _encode_key(key)
        produced = 0
        for salt in salts:
            digest = hashfn(salt + key).digest()
            for uint in unpack(fmt, digest):
                yield uint % num_bits
                produced += 1
                if produced >= num_slices:
                    return

    return _hash_maker


def _validate_portable_parameters(error_rate, num_slices, bits_per_slice,
                                  capacity, count, effective_bits,
                                  payload_length, salt_length):
    if not math.isfinite(error_rate) or not (0 < error_rate < 1):
        raise ValueError('Error_Rate must be between 0 and 1.')
    if not 0 < num_slices <= MAX_NUM_SLICES:
        raise ValueError('num_slices must be between 1 and %d' % MAX_NUM_SLICES)
    if bits_per_slice <= 0 or capacity <= 0 or capacity > MAX_CAPACITY:
        raise ValueError('Invalid portable Bloom filter parameters')
    if not 0 <= count <= capacity:
        raise ValueError('count must be between 0 and capacity')
    if effective_bits != num_slices * bits_per_slice:
        raise ValueError('Effective bit count does not match slice parameters')
    bit_byte_length = (effective_bits + 7) // 8
    if payload_length < salt_length + bit_byte_length:
        raise ValueError('Payload length does not match salt and bit array data')
    if payload_length > MAX_BLOOM_PAYLOAD_BYTES:
        raise ValueError('Payload exceeds the portable length limit')
    expected_slices = int(math.ceil(math.log(1.0 / error_rate, 2)))
    expected_bits = int(math.ceil(
        (capacity * abs(math.log(error_rate))) /
        (num_slices * (math.log(2) ** 2))))
    if num_slices != expected_slices or bits_per_slice != expected_bits:
        raise ValueError('Filter parameters do not match the portable formulas')
    return bit_byte_length


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
    hash_algorithm, hashfn, fmt, chunk_size, salts = _hash_configuration(
        num_slices, num_bits)
    hash_maker = _hash_maker_for(num_slices, num_bits, salts, hashfn, fmt)

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
        yield from hash_maker(key)

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

        # Calculate optimal number of hash functions (k)
        # k = ceil(log2(1/P)) minimizes false positive rate
        num_slices = int(math.ceil(math.log(1.0 / error_rate, 2)))
        if num_slices > MAX_NUM_SLICES:
            raise ValueError('num_slices must be between 1 and %d' % MAX_NUM_SLICES)

        # Calculate optimal bits per slice (m)
        # Derived from: n ≈ M × (ln(2)²) / |ln(P)|
        # where M = k × m (total bits)
        bits_per_slice = int(math.ceil(
            (capacity * abs(math.log(error_rate))) /
            (num_slices * (math.log(2) ** 2))))

        self._setup(error_rate, num_slices, bits_per_slice, capacity, 0)
        self.bitarray = bitarray.bitarray(self.num_bits, endian='little')
        self.bitarray.setall(False)

    def _setup(self, error_rate, num_slices, bits_per_slice, capacity, count):
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
        if not isinstance(num_slices, int) or not 0 < num_slices <= MAX_NUM_SLICES:
            raise ValueError('num_slices must be between 1 and %d' % MAX_NUM_SLICES)
        if not isinstance(bits_per_slice, int) or bits_per_slice <= 0:
            raise ValueError('bits_per_slice must be > 0')
        if not isinstance(capacity, int) or not 0 < capacity <= MAX_CAPACITY:
            raise ValueError('capacity must be between 1 and %d' % MAX_CAPACITY)
        if not isinstance(count, int) or not 0 <= count <= capacity:
            raise ValueError('count must be between 0 and capacity')
        if num_slices * bits_per_slice > MAX_BLOOM_PAYLOAD_BYTES * 8:
            raise ValueError('Bit array exceeds the portable payload limit')

        self.error_rate = error_rate
        self.num_slices = num_slices
        self.bits_per_slice = bits_per_slice
        self.capacity = capacity
        self.num_bits = num_slices * bits_per_slice
        self.count = count
        self.hash_algorithm, self.hashfn, self.hash_format, self.hash_chunk_size, self.salts = \
            _hash_configuration(self.num_slices, self.bits_per_slice)
        self.hash_version = PORTABLE_VERSION
        self.make_hashes = _hash_maker_for(
            self.num_slices, self.bits_per_slice, self.salts,
            self.hashfn, self.hash_format)

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
        """Write one self-delimiting portable v1 Bloom filter frame."""
        if self.bitarray.endian != 'little':
            raise ValueError('Portable bit order must be least-significant-bit first')
        bit_bytes = self.bitarray.tobytes()
        salts = b''.join(self.salts)
        payload_length = len(salts) + len(bit_bytes)
        if payload_length > MAX_BLOOM_PAYLOAD_BYTES:
            raise ValueError('Payload exceeds the portable length limit')

        header = bytearray(_BLOOM_HEADER.pack(
            PORTABLE_MAGIC, PORTABLE_VERSION, KEY_CODEC_UTF8_OTHER,
            BYTE_ORDER_LITTLE, BIT_ORDER_LSB_FIRST, self.hash_algorithm,
            self.hash_version, CHECKSUM_CRC32, 0, 0, _BLOOM_HEADER_LEN,
            self.error_rate, self.num_slices, self.bits_per_slice,
            self.capacity, self.count, self.num_bits, len(salts),
            payload_length, b'\x00\x00\x00\x00'))
        checksum = _crc32(bytes(header) + salts + bit_bytes)
        header[_BLOOM_CRC_OFFSET:_BLOOM_CRC_OFFSET + 4] = pack('<I', checksum)
        f.write(header)
        f.write(salts)
        f.write(bit_bytes)

    @classmethod
    def fromfile(cls, f, n=-1, legacy=False):
        """Read one portable v1 Bloom filter frame, or legacy v4 when requested."""
        if n == 0:
            raise ValueError('n too small!')
        if 0 < n < 4:
            raise ValueError('n too small!')
        magic = _read_exact(f, 4, 'Truncated Bloom filter header')

        if magic == PORTABLE_MAGIC:
            if 0 < n < _BLOOM_HEADER_LEN:
                raise ValueError('n too small!')
            header_rest = _read_exact(
                f, _BLOOM_HEADER_LEN - 4, 'Truncated Bloom filter header')
            return cls._from_portable_file(f, n, magic + header_rest)

        if not legacy:
            raise ValueError(
                'Unrecognized Bloom filter format; pass legacy=True to read the old v4 format')
        headerlen = calcsize(cls.FILE_FMT)
        if 0 < n < headerlen:
            raise ValueError('n too small!')
        raw_header = magic + _read_exact(
            f, headerlen - 4, 'Truncated legacy Bloom filter header')
        return cls._from_legacy_file(f, n, raw_header)

    @classmethod
    def _from_portable_file(cls, f, n, raw_header):
        (magic, version, key_codec, byte_order, bit_order, hash_algorithm,
         hash_version, checksum_algorithm, flags, reserved, header_length,
         error_rate, num_slices, bits_per_slice, capacity, count,
         effective_bits, salt_length, payload_length, stored_checksum) = \
            _BLOOM_HEADER.unpack(raw_header)

        object_length = _BLOOM_HEADER_LEN + payload_length
        if n > 0 and object_length > n:
            raise ValueError('n too small!')
        actual_fields = (
            magic, version, key_codec, byte_order, bit_order,
            hash_algorithm, hash_version, checksum_algorithm, flags,
            reserved, header_length)
        expected_fields = (
            PORTABLE_MAGIC, PORTABLE_VERSION, KEY_CODEC_UTF8_OTHER,
            BYTE_ORDER_LITTLE, BIT_ORDER_LSB_FIRST, hash_algorithm,
            PORTABLE_VERSION, CHECKSUM_CRC32, 0, 0, _BLOOM_HEADER_LEN)
        if actual_fields != expected_fields or hash_algorithm not in range(1, 6):
            raise ValueError('Unsupported portable Bloom filter header')
        bit_byte_length = _validate_portable_parameters(
            error_rate, num_slices, bits_per_slice, capacity, count,
            effective_bits, payload_length, salt_length)

        payload = _read_exact(f, payload_length, 'Truncated Bloom filter payload')
        if salt_length > payload_length:
            raise ValueError('Truncated Bloom filter salt data')
        raw_salts, bit_bytes = payload[:salt_length], payload[salt_length:]

        check_header = bytearray(raw_header)
        check_header[_BLOOM_CRC_OFFSET:_BLOOM_CRC_OFFSET + 4] = b'\x00\x00\x00\x00'
        if stored_checksum != pack('<I', _crc32(bytes(check_header) + payload)):
            raise ValueError('Bloom filter checksum mismatch')

        expected_algorithm, hashfn, hash_format, chunk_size, expected_salts = \
            _hash_configuration(num_slices, bits_per_slice)
        if hash_algorithm != expected_algorithm:
            raise ValueError('Hash algorithm does not match slice parameters')
        digest_size = hashfn().digest_size
        if salt_length != len(expected_salts) * digest_size:
            raise ValueError('Salt data does not match hash algorithm and slices')
        if payload_length != salt_length + bit_byte_length:
            raise ValueError('Payload length does not match salt and bit array data')
        salts = tuple(
            raw_salts[index:index + digest_size]
            for index in range(0, salt_length, digest_size))
        if salts != expected_salts:
            raise ValueError('Salt data does not match the deterministic v1 construction')

        restored = cls.__new__(cls)
        restored.error_rate = error_rate
        restored.num_slices = num_slices
        restored.bits_per_slice = bits_per_slice
        restored.capacity = capacity
        restored.num_bits = effective_bits
        restored.count = count
        restored.hash_algorithm = hash_algorithm
        restored.hash_version = hash_version
        restored.hash_format = hash_format
        restored.hash_chunk_size = chunk_size
        restored.salts = salts
        restored.hashfn = hashfn
        restored.make_hashes = _hash_maker_for(
            num_slices, bits_per_slice, salts, hashfn, hash_format)

        restored.bitarray = bitarray.bitarray(endian='little')
        restored.bitarray.frombytes(bit_bytes)
        expected_byte_bits = ((effective_bits + 7) // 8) * 8
        if len(restored.bitarray) != expected_byte_bits:
            raise ValueError('Bit length mismatch!')
        if effective_bits % 8 and any(restored.bitarray[effective_bits:]):
            raise ValueError('Non-zero Bloom filter padding bits')
        restored.bitarray = restored.bitarray[:effective_bits]
        return restored

    @classmethod
    def _from_legacy_file(cls, f, n, raw_header):
        headerlen = calcsize(cls.FILE_FMT)
        try:
            error_rate, num_slices, bits_per_slice, capacity, count = unpack(
                cls.FILE_FMT, raw_header)
        except Exception as exc:
            raise ValueError('Invalid legacy Bloom filter header') from exc
        byte_length = (num_slices * bits_per_slice + 7) // 8
        if n > 0 and headerlen + byte_length > n:
            raise ValueError('n too small!')

        restored = cls.__new__(cls)
        if not math.isfinite(error_rate) or not (0 < error_rate < 1):
            raise ValueError('Error_Rate must be between 0 and 1.')
        if not 0 < num_slices <= MAX_NUM_SLICES or bits_per_slice <= 0:
            raise ValueError('Invalid legacy Bloom filter parameters')
        if not 0 < capacity <= MAX_CAPACITY or not 0 <= count <= capacity:
            raise ValueError('Invalid legacy Bloom filter parameters')
        restored._setup(error_rate, num_slices, bits_per_slice, capacity, count)
        bit_bytes = _read_exact(f, byte_length, 'Truncated legacy Bloom filter payload')
        restored.bitarray = bitarray.bitarray(endian='little')
        restored.bitarray.frombytes(bit_bytes)
        expected_byte_bits = ((restored.num_bits + 7) // 8) * 8
        if len(restored.bitarray) != expected_byte_bits:
            raise ValueError('Bit length mismatch!')
        if restored.num_bits % 8 and any(restored.bitarray[restored.num_bits:]):
            raise ValueError('Non-zero Bloom filter padding bits')
        restored.bitarray = restored.bitarray[:restored.num_bits]
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
        if 'salts' not in d:
            self.hash_algorithm, self.hashfn, self.hash_format, self.hash_chunk_size, self.salts = \
                _hash_configuration(self.num_slices, self.bits_per_slice)
            self.hash_version = PORTABLE_VERSION
        self.make_hashes = _hash_maker_for(
            self.num_slices, self.bits_per_slice, self.salts,
            self.hashfn, self.hash_format)


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
        if not (0 < error_rate < 1):
            raise ValueError('Error_Rate must be between 0 and 1.')
        if initial_capacity <= 0 or initial_capacity > MAX_CAPACITY:
            raise ValueError('initial_capacity must be between 1 and %d' % MAX_CAPACITY)
        if mode not in (self.SMALL_SET_GROWTH, self.LARGE_SET_GROWTH):
            raise ValueError('Unsupported growth mode')
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
        """Write one self-delimiting portable v1 scalable Bloom filter frame."""
        if len(self.filters) > MAX_SCALABLE_LAYERS:
            raise ValueError('Scalable filter exceeds the portable layer limit')
        if self.scale not in (self.SMALL_SET_GROWTH, self.LARGE_SET_GROWTH):
            raise ValueError('Unsupported growth mode')
        if self.ratio != 0.9:
            raise ValueError('Unsupported tightening ratio')

        layer_stream = BytesIO()
        for layer in self.filters:
            layer.tofile(layer_stream)
        layers = layer_stream.getvalue()
        if len(layers) > MAX_SCALABLE_PAYLOAD_BYTES:
            raise ValueError('Scalable payload exceeds the portable length limit')

        header = bytearray(_SCALABLE_HEADER.pack(
            PORTABLE_SCALABLE_MAGIC, PORTABLE_SCALABLE_VERSION, 0,
            len(self.filters), CHECKSUM_CRC32, BYTE_ORDER_LITTLE,
            _SCALABLE_HEADER_LEN, self.scale, self.ratio,
            self.initial_capacity, self.error_rate, len(layers),
            b'\x00\x00\x00\x00'))
        checksum = _crc32(bytes(header) + layers)
        header[_SCALABLE_CRC_OFFSET:_SCALABLE_CRC_OFFSET + 4] = pack('<I', checksum)
        f.write(header)
        f.write(layers)

    @classmethod
    def fromfile(cls, f, legacy=False):
        """Read one portable v1 scalable frame, or legacy v4 when requested."""
        magic = _read_exact(f, 4, 'Truncated scalable Bloom filter header')
        if magic == PORTABLE_SCALABLE_MAGIC:
            raw_header = magic + _read_exact(
                f, _SCALABLE_HEADER_LEN - 4,
                'Truncated scalable Bloom filter header')
            return cls._from_portable_file(f, raw_header)
        if not legacy:
            raise ValueError(
                'Unrecognized scalable Bloom filter format; pass legacy=True to read the old v4 format')
        raw_header = magic + _read_exact(
            f, calcsize(cls.FILE_FMT) - 4,
            'Truncated legacy scalable Bloom filter header')
        return cls._from_legacy_file(f, raw_header)

    @classmethod
    def _from_portable_file(cls, f, raw_header):
        (magic, version, flags, layer_count, checksum_algorithm, byte_order,
         header_length, scale, ratio, initial_capacity, error_rate,
         payload_length, stored_checksum) = \
            _SCALABLE_HEADER.unpack(raw_header)

        actual_fields = (
            magic, version, flags, checksum_algorithm, byte_order,
            header_length)
        expected_fields = (
            PORTABLE_SCALABLE_MAGIC, PORTABLE_SCALABLE_VERSION, 0,
            CHECKSUM_CRC32, BYTE_ORDER_LITTLE, _SCALABLE_HEADER_LEN)
        if actual_fields != expected_fields:
            raise ValueError('Unsupported portable scalable Bloom filter header')
        if layer_count > MAX_SCALABLE_LAYERS:
            raise ValueError('Scalable filter exceeds the portable layer limit')
        if payload_length > MAX_SCALABLE_PAYLOAD_BYTES:
            raise ValueError('Scalable payload exceeds the portable length limit')
        if scale not in (cls.SMALL_SET_GROWTH, cls.LARGE_SET_GROWTH):
            raise ValueError('Unsupported growth mode')

        payload = _read_exact(f, payload_length, 'Truncated scalable Bloom filter payload')
        check_header = bytearray(raw_header)
        check_header[_SCALABLE_CRC_OFFSET:_SCALABLE_CRC_OFFSET + 4] = b'\x00\x00\x00\x00'
        if stored_checksum != pack('<I', _crc32(bytes(check_header) + payload)):
            raise ValueError('Scalable Bloom filter checksum mismatch')

        restored = cls.__new__(cls)
        restored._setup(scale, ratio, initial_capacity, error_rate)
        restored.filters = []
        layer_stream = BytesIO(payload)
        for layer_index in range(layer_count):
            restored.filters.append(BloomFilter.fromfile(layer_stream))
        if layer_stream.read(1):
            raise ValueError('Scalable payload contains trailing bytes')
        cls._validate_layers(restored)
        return restored

    @classmethod
    def _from_legacy_file(cls, f, raw_header):
        headerlen = calcsize(cls.FILE_FMT)
        try:
            scale, ratio, initial_capacity, error_rate = unpack(
                cls.FILE_FMT, raw_header)
        except Exception as exc:
            raise ValueError('Invalid legacy scalable Bloom filter header') from exc

        layer_count_data = _read_exact(
            f, calcsize(b'<l'), 'Truncated legacy scalable Bloom filter size table')
        layer_count, = unpack(b'<l', layer_count_data)
        if layer_count < 0 or layer_count > MAX_SCALABLE_LAYERS:
            raise ValueError('Invalid legacy scalable Bloom filter layer count')

        restored = cls.__new__(cls)
        restored._setup(scale, ratio, initial_capacity, error_rate)
        restored.filters = []

        size_fmt = b'<' + b'Q' * layer_count
        size_data = _read_exact(
            f, calcsize(size_fmt),
            'Truncated legacy scalable Bloom filter size table')
        layer_lengths = unpack(size_fmt, size_data)
        for layer_length in layer_lengths:
            restored.filters.append(
                BloomFilter.fromfile(f, layer_length, legacy=True))
        cls._validate_layers(restored)
        return restored

    @staticmethod
    def _validate_layers(filter):
        if not (0 < filter.error_rate < 1):
            raise ValueError('Error_Rate must be between 0 and 1.')
        if filter.initial_capacity <= 0 or filter.initial_capacity > MAX_CAPACITY:
            raise ValueError('Invalid initial capacity')
        if filter.ratio != 0.9 or filter.scale not in (2, 4):
            raise ValueError('Unsupported scalable filter parameters')

        expected_capacity = filter.initial_capacity
        expected_error_rate = filter.error_rate * filter.ratio
        for layer in filter.filters:
            if layer.capacity != expected_capacity:
                raise ValueError('Scalable layer capacity mismatch')
            if not math.isclose(layer.error_rate, expected_error_rate,
                                rel_tol=1e-12, abs_tol=0.0):
                raise ValueError('Scalable layer error rate mismatch')
            expected_capacity *= filter.scale
            expected_error_rate *= filter.ratio

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
