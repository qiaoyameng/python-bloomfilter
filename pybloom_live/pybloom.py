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
import hashlib
import math
from io import BytesIO
from struct import calcsize, pack, unpack

import xxhash

try:
    import bitarray
except ImportError:
    raise ImportError('pybloom_live requires bitarray >= 0.3.4')


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
        hashfn = hashlib.sha512
    elif total_hash_bits > 256:
        hashfn = hashlib.sha384
    elif total_hash_bits > 160:
        hashfn = hashlib.sha256
    elif total_hash_bits > 128:
        hashfn = hashlib.sha1
    else:
        hashfn = xxhash.xxh128  # Fastest option for typical filters

    # Pre-compute format string for unpacking hash digests
    fmt = fmt_code * (hashfn().digest_size // chunk_size)

    # Calculate number of salts needed to generate enough hash values
    num_salts, extra = divmod(num_slices, len(fmt))
    if extra:
        num_salts += 1

    # Pre-compute salts for reuse (double-hashing for better distribution)
    salts = tuple(hashfn(hashfn(pack('I', i)).digest()) for i in range(num_salts))

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
        elif not isinstance(key, bytes):
            key = str(key).encode('utf-8')

        # Generate hash values using salted hashing
        i = 0
        for salt in salts:
            h = salt.copy()  # Copy pre-computed salt (fast operation)
            h.update(key)    # Hash the key
            # Unpack hash digest into multiple integers
            for uint in unpack(fmt, h.digest()):
                yield uint % num_bits  # Reduce to valid bit index
                i += 1
                if i >= num_slices:  # Early exit optimization
                    return

    return _hash_maker, hashfn


def filter_parameters(capacity, error_rate):
    """Return conservative Bloom-filter dimensions for a layer budget."""
    num_slices = int(math.ceil(math.log(1.0 / error_rate, 2)))
    bits_per_slice = int(math.ceil(
        (capacity * abs(math.log(error_rate))) /
        (num_slices * (math.log(2) ** 2))))

    while True:
        occupancy = 1 - (1 - 1 / bits_per_slice) ** capacity
        if occupancy ** num_slices <= error_rate:
            break
        bits_per_slice += 1

    return num_slices, bits_per_slice


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

        num_slices, bits_per_slice = filter_parameters(capacity, error_rate)

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
        self.error_rate = error_rate
        self.num_slices = num_slices
        self.bits_per_slice = bits_per_slice
        self.capacity = capacity
        self.num_bits = num_slices * bits_per_slice
        self.count = count
        self.slice_set_counts = [0] * num_slices
        self.make_hashes, self.hashfn = make_hashfuncs(self.num_slices, self.bits_per_slice)

    def _refresh_slice_set_counts(self):
        """Recover occupied-bit counts from the persisted bit array."""
        self.slice_set_counts = []
        for index in range(self.num_slices):
            start = index * self.bits_per_slice
            stop = start + self.bits_per_slice
            self.slice_set_counts.append(self.bitarray[start:stop].count(1))

    def current_error_rate(self):
        """Return the false-positive bound implied by actual bit occupancy."""
        rate = 1.0
        for set_count in self.slice_set_counts:
            rate *= set_count / self.bits_per_slice
        return rate

    def projected_error_rate(self, hashes):
        """Return the worst occupancy bound after setting one key's bits."""
        rate = 1.0
        for index, position in enumerate(hashes):
            offset = index * self.bits_per_slice
            set_count = self.slice_set_counts[index]
            if not self.bitarray[offset + position]:
                set_count += 1
            rate *= set_count / self.bits_per_slice
        return rate

    def estimated_count(self):
        """Estimate inserted keys from occupancy, without trusting count."""
        estimates = []
        for set_count in self.slice_set_counts:
            occupancy = set_count / self.bits_per_slice
            if occupancy == 0:
                estimates.append(0.0)
            elif occupancy == 1:
                estimates.append(float('inf'))
            else:
                estimates.append(
                    math.log(1 - occupancy) /
                    math.log(1 - 1 / self.bits_per_slice))
        return int(math.ceil(max(estimates))) if max(estimates) > 0 else 0

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
        hashes = tuple(self.make_hashes(key))
        if self.projected_error_rate(hashes) > self.error_rate:
            raise IndexError("BloomFilter would exceed its false-positive budget")

        # Optimized path for skip_check=True (bulk insertions)
        if skip_check:
            changed = False
            offset = 0
            for index, k in enumerate(hashes):
                position = offset + k
                if not bitarray[position]:
                    self.slice_set_counts[index] += 1
                    changed = True
                bitarray[position] = True
                offset += bits_per_slice
            if changed:
                self.count += 1
            return False

        # Check for duplicate and set bits in single pass
        found_all_bits = True
        offset = 0
        for index, k in enumerate(hashes):
            position = offset + k
            if found_all_bits and not bitarray[offset + k]:
                found_all_bits = False  # Found at least one unset bit
            if not bitarray[position]:
                self.slice_set_counts[index] += 1
            bitarray[position] = True
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
        new_filter = BloomFilter.__new__(BloomFilter)
        new_filter._setup(self.error_rate, self.num_slices,
                          self.bits_per_slice, self.capacity, self.count)
        new_filter.bitarray = self.bitarray.copy()
        new_filter._refresh_slice_set_counts()
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
        if self.capacity != other.capacity or \
                self.error_rate != other.error_rate or \
                self.num_slices != other.num_slices or \
                self.bits_per_slice != other.bits_per_slice:
            raise ValueError(
                "Unioning filters requires identical dimensions and error rate")
        new_bloom = self.copy()
        new_bloom.bitarray = new_bloom.bitarray | other.bitarray
        new_bloom._refresh_slice_set_counts()
        if new_bloom.current_error_rate() > self.error_rate:
            raise ValueError("Union would exceed the filter's false-positive budget")
        new_bloom.count = new_bloom.estimated_count()
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
        new_bloom._refresh_slice_set_counts()
        return new_bloom

    def __and__(self, other):
        """Implement the & operator for intersection operation.

        Example:
            >>> bf_intersect = bf1 & bf2
        """
        return self.intersection(other)

    def tofile(self, f):
        """Serialize the Bloom filter to a file.

        Writes the filter to a binary file format that is much more space
        efficient than Python's pickle format. The file contains the filter
        parameters followed by the raw bit array data.

        File Format:
            - Header (40 bytes): error_rate, num_slices, bits_per_slice,
              capacity, count (packed as '<dQQQQ')
            - Body: Raw bit array data

        Args:
            f: File-like object opened in binary write mode ('wb'). Can be a
                regular file or BytesIO.

        Example:
            >>> bf = BloomFilter(1000, 0.01)
            >>> bf.add("test")
            >>> with open('filter.bin', 'wb') as f:
            ...     bf.tofile(f)

        See Also:
            fromfile: Class method to deserialize a filter from a file
        """
        # Write filter parameters as binary header
        f.write(pack(self.FILE_FMT, self.error_rate, self.num_slices,
                     self.bits_per_slice, self.capacity, self.count))
        # Write bit array data
        if isinstance(f, BytesIO):
            f.write(self.bitarray.tobytes())
        else:
            self.bitarray.tofile(f)

    @classmethod
    def fromfile(cls, f, n=-1):
        """Deserialize a Bloom filter from a file.

        Reads a Bloom filter that was previously saved using the tofile()
        method. The file must contain a valid filter in the binary format
        written by tofile().

        Args:
            f: File-like object opened in binary read mode ('rb'). Can be a
                regular file or BytesIO.
            n (int, optional): Maximum number of bytes to read. If n > 0,
                reads at most n bytes. If n <= 0 (default), reads the entire
                filter. Must be at least as large as the header (40 bytes).

        Returns:
            BloomFilter: A new BloomFilter instance with the state loaded
                from the file

        Raises:
            ValueError: If n is too small (< header size)
            ValueError: If the bit array length doesn't match the expected size

        Example:
            >>> with open('filter.bin', 'rb') as f:
            ...     bf = BloomFilter.fromfile(f)
            >>> "test" in bf
            True

        Note:
            Filters saved with version < 4.0.0 are incompatible due to hash
            function changes (MD5 → xxHash).
        """
        headerlen = calcsize(cls.FILE_FMT)

        # Validate minimum read size
        if 0 < n < headerlen:
            raise ValueError('n too small!')

        # Create temporary instance and restore state from file
        filter = cls(1)  # Minimal initialization, will be overwritten
        filter._setup(*unpack(cls.FILE_FMT, f.read(headerlen)))

        # Read bit array data
        filter.bitarray = bitarray.bitarray(endian='little')
        if isinstance(f, BytesIO):
            if n > 0:
                filter.bitarray.frombytes(f.read(n - headerlen))
            else:
                filter.bitarray.frombytes(f.read())
        else:
            if n > 0:
                filter.bitarray.fromfile(f, n - headerlen)
            else:
                filter.bitarray.fromfile(f)

        # Validate bit array length (accounting for byte padding)
        if filter.num_bits != len(filter.bitarray) and \
                (filter.num_bits + (8 - filter.num_bits % 8) != len(filter.bitarray)):
            raise ValueError('Bit length mismatch!')
        filter._refresh_slice_set_counts()

        return filter

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
        if not hasattr(self, 'slice_set_counts'):
            self._refresh_slice_set_counts()


class ScalableBloomFilter:
    """A scalable Bloom filter with a joint false-positive upper bound.

    Layer i receives budget ``error_rate * (1 - ratio) * ratio ** i``. The
    budgets form a geometric partition of the total target, so the union bound
    across all present and future layers never exceeds ``error_rate``.
    """
    SMALL_SET_GROWTH = 2  # Growth factor for memory-constrained scenarios
    LARGE_SET_GROWTH = 4  # Growth factor for performance (default)
    FILE_FMT = '<idQd'

    def __init__(self, initial_capacity=100, error_rate=0.001,
                 mode=LARGE_SET_GROWTH, ratio=0.9):
        """Initialize a scalable filter with one total error budget."""
        if not 0 < error_rate < 1:
            raise ValueError("Error_Rate must be between 0 and 1.")
        if not isinstance(initial_capacity, int) or initial_capacity <= 0:
            raise ValueError("Initial capacity must be a positive integer.")
        if not isinstance(mode, int) or mode < 2:
            raise ValueError("Growth mode must be an integer >= 2.")
        if not 0 < ratio < 1:
            raise ValueError("Budget ratio must be between 0 and 1.")

        self._setup(mode, ratio, initial_capacity, error_rate)
        self.filters = []

    def _setup(self, mode, ratio, initial_capacity, error_rate):
        """Internal setup method for initializing or restoring filter state.

        Validate parameters when restoring a persisted filter as well.
        """
        if not 0 < error_rate < 1:
            raise ValueError("Error_Rate must be between 0 and 1.")
        if not isinstance(initial_capacity, int) or initial_capacity <= 0:
            raise ValueError("Initial capacity must be a positive integer.")
        if not isinstance(mode, int) or mode < 2:
            raise ValueError("Growth mode must be an integer >= 2.")
        if not 0 < ratio < 1:
            raise ValueError("Budget ratio must be between 0 and 1.")

        self.scale = mode
        self.ratio = ratio
        self.initial_capacity = initial_capacity
        self.error_rate = error_rate

    def layer_budget(self, index):
        """Return the false-positive budget allocated to ``index``."""
        return self.error_rate * (1 - self.ratio) * (self.ratio ** index)

    @property
    def layer_budgets(self):
        """Return the budgets allocated to the currently existing layers."""
        return [self.layer_budget(i) for i in range(len(self.filters))]

    @property
    def error_upper_bound(self):
        """Joint union bound implied by the current bit arrays."""
        return min(1.0, sum(layer.current_error_rate() for layer in self.filters))

    def _create_layer(self, index):
        """Create a layer whose dimensions are set by its own budget."""
        budget = self.layer_budget(index)
        if budget <= 0.0 or not math.isfinite(budget):
            raise IndexError("ScalableBloomFilter has exhausted its floating-point budget")

        capacity = self.initial_capacity * (self.scale ** index)
        return BloomFilter(capacity=capacity, error_rate=budget)

    def _set_key(self, layer, key, hashes):
        """Set key bits and count only insertions that change occupancy."""
        changed = False
        offset = 0
        for index, position in enumerate(hashes):
            bit_position = offset + position
            if not layer.bitarray[bit_position]:
                layer.bitarray[bit_position] = True
                layer.slice_set_counts[index] += 1
                changed = True
            offset += layer.bits_per_slice

        if changed:
            layer.count += 1
        return changed

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

        Creates a new layer when projected occupancy would exceed its budget.

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
        if key in self:
            return True

        if not self.filters:
            layer = self._create_layer(0)
            self.filters.append(layer)
            hashes = tuple(layer.make_hashes(key))
        else:
            layer = self.filters[-1]
            hashes = tuple(layer.make_hashes(key))
            if layer.projected_error_rate(hashes) > layer.error_rate:
                layer = self._create_layer(len(self.filters))
                self.filters.append(layer)
                hashes = tuple(layer.make_hashes(key))

        self._set_key(layer, key, hashes)
        return False

    def union(self, other):
        """Calculate the union of two Scalable Bloom filters.

        Creates a new Scalable Bloom filter containing elements from either
        filter. Corresponding layers must have matching dimensions, and merged
        occupancy must remain within that layer's budget.

        Args:
            other (ScalableBloomFilter): Another ScalableBloomFilter to union
                with this one. Must have the same mode, ratio,
                initial_capacity and error_rate.

        Returns:
            ScalableBloomFilter: A new filter representing the union

        Raises:
            ValueError: If filters are incompatible or merged occupancy breaks
                the layer budget

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
            - Re-estimates counts from actual bit occupancy
            - Rejects combinations that cannot preserve the budget contract
        """
        if self.scale != other.scale or \
                self.ratio != other.ratio or \
                self.initial_capacity != other.initial_capacity or \
                self.error_rate != other.error_rate:
            raise ValueError("Unioning two scalable bloom filters requires "
                           "the same mode, ratio, initial capacity and error rate")

        result = self.__class__.__new__(self.__class__)
        result._setup(self.scale, self.ratio,
                      self.initial_capacity, self.error_rate)
        result.filters = []

        layer_count = max(len(self.filters), len(other.filters))
        for index in range(layer_count):
            if index < len(self.filters) and index < len(other.filters):
                left = self.filters[index]
                right = other.filters[index]
                if left.capacity != right.capacity or \
                        left.error_rate != right.error_rate or \
                        left.num_slices != right.num_slices or \
                        left.bits_per_slice != right.bits_per_slice:
                    raise ValueError("Union requires matching parameters at every layer")

                merged = left.copy()
                merged.bitarray = merged.bitarray | right.bitarray
                merged._refresh_slice_set_counts()
                if merged.current_error_rate() > merged.error_rate:
                    raise ValueError(
                        "Union would exceed this layer's false-positive budget")
                merged.count = merged.estimated_count()
            elif index < len(self.filters):
                merged = self.filters[index].copy()
            else:
                merged = other.filters[index].copy()

            if merged.error_rate != self.layer_budget(index):
                raise ValueError("Layer budgets do not match the stored policy")
            if merged.current_error_rate() > merged.error_rate:
                raise ValueError(
                    "Union would exceed this layer's false-positive budget")
            merged.count = merged.estimated_count()
            result.filters.append(merged)

        return result

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
        """Serialize this Scalable Bloom Filter to a file.

        Writes the filter and all internal BloomFilter instances to a binary
        file format. The format includes metadata about each internal filter
        for efficient deserialization.

        File Format:
            1. Header: scale, ratio, initial_capacity, error_rate (32 bytes)
            2. Number of internal filters (4 bytes)
            3. Size table: size of each internal filter in bytes
            4. Internal filter data: each filter serialized sequentially

        Args:
            f: File-like object opened in binary write mode ('wb')

        Example:
            >>> sbf = ScalableBloomFilter()
            >>> for i in range(1000):
            ...     sbf.add(i)
            >>> with open('scalable_filter.bin', 'wb') as f:
            ...     sbf.tofile(f)

        See Also:
            fromfile: Class method to deserialize a filter from a file
        """
        # Write main filter parameters
        f.write(pack(self.FILE_FMT, self.scale, self.ratio,
                     self.initial_capacity, self.error_rate))

        # Write number of internal filters
        f.write(pack(b'<l', len(self.filters)))

        if len(self.filters) > 0:
            # Write size table for internal filters, then filter data
            headerpos = f.tell()
            headerfmt = b'<' + b'Q' * (len(self.filters))
            # Reserve space for size table
            f.write(b'.' * calcsize(headerfmt))

            # Write each internal filter and record its size
            filter_sizes = []
            for filter in self.filters:
                begin = f.tell()
                filter.tofile(f)
                filter_sizes.append(f.tell() - begin)

            # Go back and write size table
            f.seek(headerpos)
            f.write(pack(headerfmt, *filter_sizes))

    @classmethod
    def fromfile(cls, f):
        """Deserialize a Scalable Bloom Filter from a file.

        Reads a Scalable Bloom Filter that was previously saved using the
        tofile() method, including all internal BloomFilter instances.

        Args:
            f: File-like object opened in binary read mode ('rb')

        Returns:
            ScalableBloomFilter: A new ScalableBloomFilter instance with the
                state loaded from the file

        Example:
            >>> with open('scalable_filter.bin', 'rb') as f:
            ...     sbf = ScalableBloomFilter.fromfile(f)
            >>> 500 in sbf
            True

        Note:
            Filters saved with version < 4.0.0 are incompatible due to hash
            function changes (MD5 → xxHash).
        """
        restored = cls.__new__(cls)
        restored._setup(*unpack(cls.FILE_FMT, f.read(calcsize(cls.FILE_FMT))))
        restored.filters = []

        # Read number of internal filters
        nfilters, = unpack(b'<l', f.read(calcsize(b'<l')))

        if nfilters > 0:
            # Read size table
            header_fmt = b'<' + b'Q' * nfilters
            bytes = f.read(calcsize(header_fmt))
            filter_lengths = unpack(header_fmt, bytes)

            # Read each internal filter
            for fl in filter_lengths:
                restored.filters.append(BloomFilter.fromfile(f, fl))
        else:
            restored.filters = []

        for index, layer in enumerate(restored.filters):
            expected_capacity = restored.initial_capacity * (restored.scale ** index)
            if layer.capacity != expected_capacity:
                raise ValueError("Persisted layer capacity does not match growth policy")
            if layer.error_rate != restored.layer_budget(index):
                raise ValueError("Persisted layer budget does not match growth policy")
            if layer.current_error_rate() > layer.error_rate:
                raise ValueError("Persisted layer exceeds its false-positive budget")

        return restored

    def __getstate__(self):
        state = self.__dict__.copy()
        state['filters'] = [layer.__getstate__() for layer in self.filters]
        return state

    def __setstate__(self, state):
        layer_states = state.pop('filters')
        self.__dict__.update(state)
        self.filters = []
        for layer_state in layer_states:
            layer = BloomFilter.__new__(BloomFilter)
            layer.__setstate__(layer_state)
            self.filters.append(layer)

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
