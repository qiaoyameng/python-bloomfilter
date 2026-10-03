"""Counting Bloom Filter with credential-based, per-insertion revocation.

A plain Bloom filter cannot support deletion: clearing bits for one key may
erase bits shared with other keys, introducing false negatives. A counting
filter replaces bits with small counters, so an insertion can be undone by
decrementing the same counters -- but only if we know *which* insertion is
being undone. Decrementing because "the key queries as present" is unsafe:
membership queries have false positives, and the same key may have been
inserted multiple times.

This module therefore implements revocation through opaque credentials:

- ``add(key)`` returns an opaque credential (bytes) that identifies exactly
  that one insertion.
- ``revoke(credential)`` undoes exactly that insertion. There is no
  "delete by key" operation.
- Inserting the same key twice yields two independent credentials; revoking
  one leaves the other insertion's contribution intact.
- Credentials are bound to the filter's random epoch and secret key and to
  the exact counter positions of the insertion via HMAC. Forged credentials,
  credentials from another filter, and already-revoked credentials are
  rejected with :class:`InvalidCredentialError` and never change state.

Memory model (bounded, and exposed for capacity planning):

- Counters: ``num_slices * bits_per_slice`` counters of ``counter_bits``
  bits each (see ``counter_bytes``).
- Active credential table: at most ``max_credentials`` entries, each holding
  a sequence number and the insertion's counter positions
  (see ``credential_table_bytes``).
- Persistence: ``tofile`` writes counters plus the credential table and the
  secret/epoch, so credentials remain valid across save/load. The exact
  serialized size is available ahead of time via ``persistent_bytes``.

Counter overflow is rejected atomically: if any counter touched by an
insertion is already at its maximum, the *entire* insertion is refused with
:class:`CounterOverflowError` before any counter is modified. Counters are
never saturated-and-wrapped, so a later revocation can never decrement a
counter to a wrong value.
"""
import hashlib
import hmac
import math
import os
from array import array
from io import BytesIO
from struct import calcsize, pack, unpack

from .pybloom import make_hashfuncs

_COUNTER_TYPECODES = {8: 'B', 16: 'H', 32: 'I', 64: 'Q'}
_SECRET_SIZE = 32
_MAC_SIZE = 16
_TOKEN_FMT = '<QQ'  # (epoch, sequence)
TOKEN_SIZE = calcsize(_TOKEN_FMT) + _MAC_SIZE
_NO_LIMIT = 0xFFFFFFFFFFFFFFFF  # serialized sentinel for max_credentials=None
_DEFAULT_BOUND = object()  # sentinel: max_credentials not specified
_MAGIC = b'CBF1'


class CounterOverflowError(Exception):
    """An insertion was rejected because a counter is at its maximum.

    The filter state is completely unchanged; no counter was modified.
    """


class CredentialTableFullError(Exception):
    """The bounded active-credential table is full.

    Revoke some credentials (or raise ``max_credentials``) before inserting
    more elements. The insertion was not performed.
    """


class InvalidCredentialError(ValueError):
    """A credential is forged, belongs to another filter, or was revoked."""


class CountingBloomFilter:
    """A counting Bloom filter whose insertions can be individually revoked.

    Args:
        capacity (int): Expected maximum number of simultaneously active
            insertions. Used (with ``error_rate``) to size the counter array.
        error_rate (float): Target false positive probability, in (0, 1).
        max_credentials (int or None): Bound on the active credential table.
            ``add`` raises :class:`CredentialTableFullError` once this many
            un-revoked credentials exist. Pass ``None`` for an unbounded
            table. If omitted, defaults to ``capacity``.
        counter_bits (int): Width of each counter: 8, 16, 32, or 64.
            An insertion that would exceed ``2**counter_bits - 1`` in any
            counter is rejected atomically with :class:`CounterOverflowError`.

    Persistence cost:
        ``persistent_bytes()`` returns the exact number of bytes ``tofile``
        will write: a fixed header, the counter array, and
        ``8 + 8 * num_slices`` bytes per active credential. Credentials stay
        valid after a save/load round trip.
    """

    # magic, error_rate, num_slices, bits_per_slice, capacity, count,
    # counter_bits, epoch, next_seq, max_credentials, num_credentials
    FILE_FMT = b'<4sdQQQQBQQQQ'

    def __init__(self, capacity, error_rate=0.001, max_credentials=_DEFAULT_BOUND,
                 counter_bits=8):
        if not (0 < error_rate < 1):
            raise ValueError("Error_Rate must be between 0 and 1.")
        if not capacity > 0:
            raise ValueError("Capacity must be > 0")
        if counter_bits not in _COUNTER_TYPECODES:
            raise ValueError("counter_bits must be one of 8, 16, 32, 64")
        if max_credentials is _DEFAULT_BOUND:
            max_credentials = capacity
        elif max_credentials is not None and max_credentials < 0:
            raise ValueError("max_credentials must be >= 0")

        num_slices = int(math.ceil(math.log(1.0 / error_rate, 2)))
        bits_per_slice = int(math.ceil(
            (capacity * abs(math.log(error_rate))) /
            (num_slices * (math.log(2) ** 2))))

        self._setup(error_rate, num_slices, bits_per_slice, capacity, 0,
                    counter_bits, max_credentials)
        self._epoch = int.from_bytes(os.urandom(8), 'little')
        self._secret = os.urandom(_SECRET_SIZE)
        self._next_seq = 0
        self._credentials = {}
        self._counters = array(_COUNTER_TYPECODES[counter_bits],
                               bytes(self._counter_nbytes()))

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _setup(self, error_rate, num_slices, bits_per_slice, capacity, count,
               counter_bits, max_credentials):
        self.error_rate = error_rate
        self.num_slices = num_slices
        self.bits_per_slice = bits_per_slice
        self.capacity = capacity
        self.count = count
        self.counter_bits = counter_bits
        self.max_credentials = max_credentials
        self.num_counters = num_slices * bits_per_slice
        self.make_hashes, self.hashfn = make_hashfuncs(num_slices, bits_per_slice)

    def _counter_nbytes(self):
        return self.num_counters * (self.counter_bits // 8)

    @property
    def _max_count(self):
        return (1 << self.counter_bits) - 1

    def _positions(self, key):
        """Counter positions touched by inserting ``key`` (k distinct ints)."""
        positions = []
        offset = 0
        for h in self.make_hashes(key):
            positions.append(offset + h)
            offset += self.bits_per_slice
        return tuple(positions)

    def _mac(self, epoch, seq, positions):
        msg = pack(_TOKEN_FMT, epoch, seq)
        msg += pack('<%dQ' % len(positions), *positions)
        return hmac.new(self._secret, msg, hashlib.sha256).digest()[:_MAC_SIZE]

    def _make_token(self, seq, positions):
        return pack(_TOKEN_FMT, self._epoch, seq) + \
            self._mac(self._epoch, seq, positions)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def add(self, key):
        """Insert ``key`` and return an opaque credential for this insertion.

        Every call returns an independent credential, even for a key that was
        inserted before. Keep the credential if you may later want to undo
        exactly this insertion via :meth:`revoke`.

        Returns:
            bytes: Opaque credential (``TOKEN_SIZE`` bytes). Treat it as an
                uninterpreted blob.

        Raises:
            CredentialTableFullError: If the bounded credential table is
                full. Nothing was inserted.
            CounterOverflowError: If any affected counter is at its maximum.
                Nothing was inserted; all counters are unchanged.
        """
        if self.max_credentials is not None and \
                len(self._credentials) >= self.max_credentials:
            raise CredentialTableFullError(
                "active credential table is full (%d entries); "
                "revoke credentials or increase max_credentials"
                % self.max_credentials)

        positions = self._positions(key)
        counters = self._counters
        max_count = self._max_count
        # Atomicity: validate every counter before mutating any of them, so
        # an overflow rejects the whole insertion and can never leave a
        # saturated counter that a later revoke would decrement wrongly.
        for p in positions:
            if counters[p] >= max_count:
                raise CounterOverflowError(
                    "counter overflow at position %d (max %d); insertion "
                    "rejected, filter unchanged" % (p, max_count))
        for p in positions:
            counters[p] += 1

        seq = self._next_seq
        self._next_seq += 1
        self._credentials[seq] = positions
        self.count += 1
        return self._make_token(seq, positions)

    def revoke(self, token):
        """Undo the exact insertion identified by ``token``.

        Decrements precisely the counters that this insertion incremented.
        Because overflow is rejected on insertion and each credential is
        single-use, every decremented counter is guaranteed to be >= 1.

        Raises:
            InvalidCredentialError: If the token is malformed, forged, belongs
                to a different filter (different epoch/secret), or was already
                revoked. The filter state is not modified in any of these
                cases.
        """
        epoch, seq, positions = self._verify(token)
        counters = self._counters
        for p in positions:
            counters[p] -= 1
        del self._credentials[seq]
        self.count -= 1

    def _verify(self, token):
        """Validate a credential; return (epoch, seq, positions) if authentic."""
        if not isinstance(token, (bytes, bytearray)) or len(token) != TOKEN_SIZE:
            raise InvalidCredentialError("malformed credential")
        epoch, seq = unpack(_TOKEN_FMT, bytes(token[:calcsize(_TOKEN_FMT)]))
        if epoch != self._epoch:
            raise InvalidCredentialError(
                "credential belongs to a different filter (epoch mismatch)")
        positions = self._credentials.get(seq)
        if positions is None:
            raise InvalidCredentialError(
                "unknown or already-revoked credential")
        expected = self._mac(epoch, seq, positions)
        if not hmac.compare_digest(expected, bytes(token[calcsize(_TOKEN_FMT):])):
            raise InvalidCredentialError("credential authentication failed")
        return epoch, seq, positions

    def is_valid(self, token):
        """Return True if ``token`` is an authentic, un-revoked credential."""
        try:
            self._verify(token)
        except InvalidCredentialError:
            return False
        return True

    def __contains__(self, key):
        """Test membership: True if all of the key's counters are non-zero.

        Never produces false negatives for insertions that have not been
        revoked; may produce false positives at roughly ``error_rate``.
        """
        counters = self._counters
        offset = 0
        for h in self.make_hashes(key):
            if counters[offset + h] == 0:
                return False
            offset += self.bits_per_slice
        return True

    def __len__(self):
        """Number of active (inserted and not yet revoked) insertions."""
        return self.count

    # ------------------------------------------------------------------
    # capacity / cost introspection
    # ------------------------------------------------------------------
    @property
    def active_credentials(self):
        """Number of un-revoked credentials currently tracked."""
        return len(self._credentials)

    @property
    def counter_bytes(self):
        """Bytes used by the counter array."""
        return self._counter_nbytes()

    @property
    def credential_entry_bytes(self):
        """Persistent bytes per active credential (sequence + positions)."""
        return 8 + 8 * self.num_slices

    @property
    def credential_table_bytes(self):
        """Estimated in-memory bytes of the active credential table."""
        # CPython overhead per dict entry + int key + tuple of ints.
        per_entry = 232 + 56 * self.num_slices
        return 64 + per_entry * len(self._credentials)

    def persistent_bytes(self):
        """Exact number of bytes ``tofile`` will write at this moment."""
        return (calcsize(self.FILE_FMT) + _SECRET_SIZE +
                self._counter_nbytes() +
                len(self._credentials) * self.credential_entry_bytes)

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def tofile(self, f):
        """Serialize counters, epoch/secret, and the credential table.

        Credentials remain valid after :meth:`fromfile` restores the filter.
        The file contains the HMAC secret, so protect it like the in-memory
        filter itself.

        Args:
            f: File-like object opened in binary write mode ('wb').
        """
        f.write(pack(self.FILE_FMT, _MAGIC, self.error_rate, self.num_slices,
                     self.bits_per_slice, self.capacity, self.count,
                     self.counter_bits, self._epoch, self._next_seq,
                     _NO_LIMIT if self.max_credentials is None
                     else self.max_credentials,
                     len(self._credentials)))
        f.write(self._secret)
        counters = self._counters
        if isinstance(f, BytesIO):
            f.write(counters.tobytes())
        else:
            counters.tofile(f)
        for seq, positions in self._credentials.items():
            f.write(pack('<Q', seq))
            f.write(pack('<%dQ' % len(positions), *positions))

    @classmethod
    def fromfile(cls, f):
        """Restore a filter saved with :meth:`tofile`.

        Both the counters and the validity of outstanding credentials are
        preserved: credentials issued before saving can be revoked on the
        restored filter, and credentials revoked before saving stay revoked.

        Args:
            f: File-like object opened in binary read mode ('rb').
        """
        header = f.read(calcsize(cls.FILE_FMT))
        (magic, error_rate, num_slices, bits_per_slice, capacity, count,
         counter_bits, epoch, next_seq, max_credentials,
         num_credentials) = unpack(cls.FILE_FMT, header)
        if magic != _MAGIC:
            raise ValueError('not a CountingBloomFilter file')

        filt = cls.__new__(cls)
        filt._setup(error_rate, num_slices, bits_per_slice, capacity, count,
                    counter_bits,
                    None if max_credentials == _NO_LIMIT else max_credentials)
        filt._epoch = epoch
        filt._next_seq = next_seq
        filt._secret = f.read(_SECRET_SIZE)
        if len(filt._secret) != _SECRET_SIZE:
            raise ValueError('truncated file (secret)')

        counters = array(_COUNTER_TYPECODES[counter_bits])
        nbytes = filt._counter_nbytes()
        data = f.read(nbytes)
        if len(data) != nbytes:
            raise ValueError('truncated file (counters)')
        counters.frombytes(data)
        filt._counters = counters

        credentials = {}
        entry_fmt = '<%dQ' % (1 + num_slices)
        entry_size = calcsize(entry_fmt)
        for _ in range(num_credentials):
            raw = f.read(entry_size)
            if len(raw) != entry_size:
                raise ValueError('truncated file (credentials)')
            values = unpack(entry_fmt, raw)
            credentials[values[0]] = tuple(values[1:])
        filt._credentials = credentials
        return filt

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['make_hashes']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.make_hashes, self.hashfn = make_hashfuncs(self.num_slices,
                                                       self.bits_per_slice)
