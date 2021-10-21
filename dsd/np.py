"""
Library for doing sequence design that can be expressed as linear algebra
operations for rapid processing by numpy (e.g., generating all DNA sequences
of a certain length and calculating all their full duplex binding energies
in the nearest neighbor model and filtering those outside a given range).

Based on the DNA single-stranded tile (SST) sequence designer used in the following publication.

"Diverse and robust molecular algorithms using reprogrammable DNA self-assembly"
Woods\*, Doty\*, Myhrvold, Hui, Zhou, Yin, Winfree. (\*Joint first co-authors)
"""  # noqa

from __future__ import annotations

from typing import Tuple, List, Collection, Optional, Union, Sequence, Dict, Iterable
from dataclasses import dataclass
import math
import itertools as it
from functools import lru_cache

import numpy as np

default_rng: np.random.Generator = np.random.default_rng()  # noqa

bits2base = ['A', 'C', 'G', 'T']
base2bits = {'A': 0b00, 'C': 0b01, 'G': 0b10, 'T': 0b11,
             'a': 0b00, 'c': 0b01, 'g': 0b10, 't': 0b11}


def idx2seq(idx: int, length: int) -> str:
    """Return the lexicographic idx'th DNA sequence of given length."""
    seq = ['x'] * length
    for i in range(length - 1, -1, -1):
        seq[i] = bits2base[idx & 0b11]
        idx >>= 2
    return ''.join(seq)


def seq2arr(seq: str, base2bits_local: Optional[Dict[str, int]] = None) -> np.ndarray:
    """Convert seq (string with DNA alphabet) to numpy array with integers 0,1,2,3."""
    if base2bits_local is None:
        base2bits_local = base2bits
    return np.array([base2bits_local[base] for base in seq], dtype=np.ubyte)


def seqs2arr(seqs: Sequence[str]) -> np.ndarray:
    """Return numpy 2D array converting the given DNA sequences to integers."""
    if len(seqs) == 0:
        return np.empty((0, 0), dtype=np.ubyte)
    seq_len = len(seqs[0])
    for seq in seqs:
        if len(seq) != seq_len:
            raise ValueError('All sequences in seqs must be equal length')
    num_seqs = len(seqs)
    arr = np.empty((num_seqs, seq_len), dtype=np.ubyte)
    for i in range(num_seqs):
        arr[i] = [base2bits[base] for base in seqs[i]]
    return arr


def arr2seq(arr: np.ndarray) -> str:
    bases_ch = [bits2base[base] for base in arr]
    return ''.join(bases_ch)


def make_array_with_all_sequences(length: int, digits: Sequence[int]) -> np.ndarray:
    num_digits = len(digits)
    num_seqs = num_digits ** length
    powers_num_digits = [num_digits ** k for k in range(length)]
    digits = np.array(digits, dtype=np.ubyte)

    arr = np.zeros((num_seqs, length), dtype=np.ubyte)
    for i, j, c in zip(reversed(powers_num_digits), powers_num_digits, list(range(length))):
        arr[:, c] = np.tile(np.repeat(digits, i), j)

    return arr


def make_array_with_all_dna_seqs(length: int, bases: Collection[str] = ('A', 'C', 'G', 'T')) -> np.ndarray:
    """Return 2D numpy array with all DNA sequences of given length in
    lexicographic order. Bases contains bases to be used: ('A','C','G','T') by
    default, but can be set to a subset of these.

    Uses the encoding described in the documentation for DNASeqList. The result
    is a 2D array, where each row represents a DNA sequence, and that row
    has one byte per base."""

    if len(bases) == 0:
        raise ValueError('bases cannot be empty')
    if not set(bases) <= {'A', 'C', 'G', 'T'}:
        raise ValueError(f"bases must be a subset of {'A', 'C', 'G', 'T'}; cannot be {bases}")

    base_bits = [base2bits[base] for base in bases]
    digits = np.array(base_bits, dtype=np.ubyte)

    return make_array_with_all_sequences(length, digits)
    # num_bases = len(bases)
    # num_seqs = num_bases ** length
    #
    # # the former code took up too much memory (using int32 or int64)
    # # the following code makes sure it's 1 byte per base
    # powers_num_bases = [num_bases ** k for k in range(length)]
    #
    # list_of_arrays = False
    # if list_of_arrays:
    #     # this one seems to be faster but takes more memory, probably because just before the last command
    #     # there are two copies of the array in memory at once
    #     columns = []
    #     for i, j, c in zip(reversed(powers_num_bases), powers_num_bases, list(range(length))):
    #         columns.append(np.tile(np.repeat(bases, i), j))
    #     arr = np.vstack(columns).transpose()
    # else:
    #     # this seems to be slightly slower but takes less memory, since it
    #     # allocates only the final array, plus one extra column of that
    #     # array at a time
    #     arr = np.empty((num_seqs, length), dtype=np.ubyte)
    #     for i, j, c in zip(reversed(powers_num_bases), powers_num_bases, list(range(length))):
    #         arr[:, c] = np.tile(np.repeat(bases, i), j)
    #
    # return arr


def make_array_with_random_subset_of_dna_seqs(
        length: int, num_random_seqs: int, rng: np.random.Generator = default_rng,
        bases: Collection[str] = ('A', 'C', 'G', 'T')) -> np.ndarray:
    """
    Return 2D numpy array with random subset of size `num_seqs` of DNA sequences of given length.
    Bases contains bases to be used: ('A','C','G','T') by default, but can be set to a subset of these.

    Uses the encoding described in the documentation for DNASeqList. The result is a 2D array,
    where each row represents a DNA sequence, and that row has one byte per base.

    Sequences returned will be unique (i.e., sampled without replacement) and in a random order

    :param length:
        length of each row
    :param num_random_seqs:
        number of rows
    :param bases:
        DNA bases to use
    :param rng:
        numpy random number generator (type returned by numpy.random.default_rng())
    :return:
        2D numpy array with random subset of size `num_seqs` of DNA sequences of given length
    """
    if length < 0:
        raise ValueError(f'length = {num_random_seqs} must be nonnegative')
    elif length == 0:
        return np.array([[]], dtype=np.ubyte)
    if num_random_seqs <= 0:
        raise ValueError(f'num_seqs = {num_random_seqs} must be positive')
    if not set(bases) <= {'A', 'C', 'G', 'T'}:
        raise ValueError(f"bases must be a subset of {'A', 'C', 'G', 'T'}; cannot be {bases}")
    if len(bases) == 0:
        raise ValueError('bases cannot be empty')
    elif len(bases) == 1:
        raise ValueError('bases must have at least two elements')

    max_possible = len(bases) ** length
    if num_random_seqs > max_possible:
        raise ValueError(f'''\
num_random_seqs = {num_random_seqs} is greater than the total number {max_possible} 
of sequences of length {length} using alphabet {bases}, so we cannot guarantee 
that many unique sequences. Please set num_random_seqs <= {max_possible}.''')

    # If we want sufficiently many sequences, then it's simpler to just generate all sequences
    # of that length and choose a random subset of size num_seqs.
    if num_random_seqs >= max_possible / 4:
        all_seqs = make_array_with_all_dna_seqs(length=length, bases=bases)
        # https://stackoverflow.com/a/27815343/5339430
        idxs = rng.choice(all_seqs.shape[0], num_random_seqs, replace=False)
        sampled_seqs = all_seqs[idxs]
        return sampled_seqs

    # This comment justifies why we sample 2*num_random_seqs sequences randomly (with replacement) in order
    # to get our goal of at least num_random_seqs *unique* sequences. We sample with replacement since that's
    # the only option using numpy's random number generation routines: we are generating a 2D array
    # of numbers and want each *row* to be unique, but numpy can only let us say each *number* is unique,
    # which doesn't describe what we want.
    #
    # Define
    #
    #   m = num_seqs_to_sample = number of sequences we sample with replacement
    #   n = max_possible       = total number of sequences of that length
    #   k = num_random_seqs    = desired number of unique sequences
    #
    # If we sample m sequences with replacement, out of n total, this is tossing m balls into n bins.
    # We want m sufficiently large that at least k bins are non-empty. We throw m=2*k balls.
    #
    # Above we check if k >= n/4 and generate all sequences if so. Thus, if we need the random selection
    # below, then k < n/4. Thus at least 3/4 of bins are empty the entire time, so each time we throw a ball,
    # there is a < 1/4 chance for it to land in a non-empty bin. We are throwing m=2*k balls,
    # so the number of these that land in non-empty bins is at most X=Binomial(2*k, 1/4), with E[X] = k/2.
    # Using Chernoff bounds for X=Binomial(n,p) with E[X]=np and delta=2 giving
    #   Pr[we generate fewer than k unique sequences]
    #      = Pr[more than k out of 2*k balls land in non-empty bins]
    #      = Pr[X > (1+delta)E[X]]
    #      < e^{-delta^2 E[X] / 3}
    #      = e^{-2*k/3},
    # i.e., for large k, a very small probability. Even for k=1, this is about 1/2,
    # so we'll only need to execute expected 2 iterations of the while loop below.

    base_bits = np.array([base2bits[base] for base in bases], dtype=np.ubyte)
    num_seqs_to_sample = 2 * num_random_seqs  # c*k in analysis above
    unique_sorted_arr = None

    # odds are low to have a collision, so for simplicity we just repeat the whole process if needed
    while unique_sorted_arr is None or len(unique_sorted_arr) < num_random_seqs:
        arr = rng.choice(a=base_bits, size=(num_seqs_to_sample, length))
        unique_sorted_arr = np.unique(arr, axis=0)
        if len(unique_sorted_arr) < num_random_seqs:
            print(f'WARNING: did not find {num_random_seqs} unique sequences. If you are seeing this warning '
                  f'repeatedly, check the parameters to make_array_with_random_subset_of_dna_seqs.')

    # We probably have too many, so pick a random subset of sequences to return.
    # We need a random subset, rather than just taking the first num_random_seqs elements,
    # since numpy.unique above sorts the elements.
    idxs = rng.choice(unique_sorted_arr.shape[0], num_random_seqs, replace=False)
    sampled_seqs = unique_sorted_arr[idxs]
    return sampled_seqs


# https://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
# In Python 3.8 there's math.comb, but this is about 3x faster somehow.
import operator as op
from functools import reduce


def comb(n: int, k: int) -> int:
    # n choose k = n! / (k! * (n-k)!)
    k = min(k, n - k)
    numer = reduce(op.mul, range(n, n - k, -1), 1)
    denom = reduce(op.mul, range(1, k + 1), 1)
    return numer // denom


def make_array_with_all_dna_seqs_hamming_distance(
        dist: int, seq: str, bases: Collection[str] = ('A', 'C', 'G', 'T')) -> np.ndarray:
    """
    Return 2D numpy array with all DNA sequences of given length in lexicographic order. Bases contains
    bases to be used: ('A','C','G','T') by default, but can be set to a subset of these.

    Uses the encoding described in the documentation for DNASeqList. The result is a 2D array, where each
    row represents a DNA sequence, and that row has one byte per base.
    """
    length = len(seq)
    assert 1 <= dist <= length

    num_bases = len(bases)

    if num_bases == 0:
        raise ValueError('bases cannot be empty')
    if not set(bases) <= {'A', 'C', 'G', 'T'}:
        raise ValueError(f"bases must be a subset of {'A', 'C', 'G', 'T'}; cannot be {bases}")

    num_ways_to_choose_subsequence_indices = comb(length, dist)
    num_different_bases = len(bases) - 1
    num_subsequences = num_different_bases ** dist
    num_seqs = num_ways_to_choose_subsequence_indices * num_subsequences

    # for simplicity of modular arithmetic, we use integers 0,...,len(bases)-1 to represent the bases,
    # then map these back to the correct subset of 0,1,2,3 when we are done
    offsets = range(1, num_different_bases + 1)
    subseq_offsets = make_array_with_all_sequences(length=dist, digits=offsets)
    assert len(subseq_offsets) == num_subsequences
    subseq_offsets_repeats = \
        np.tile(subseq_offsets.flatten(), num_ways_to_choose_subsequence_indices).reshape(num_seqs, dist)

    # all (length choose dist) indices where we could change the bases
    idxs = combnr_idxs(length, dist)
    assert len(idxs) == num_ways_to_choose_subsequence_indices
    idxs_repeat = np.tile(idxs, num_subsequences).reshape(num_seqs, length)
    assert len(idxs_repeat) == num_seqs

    # map subset of bases used to *prefix* of 0,1,2,3
    base2bits_local = {base: digit for base, digit in zip(bases, range(4))}
    seq_as_arr = seq2arr(seq, base2bits_local=base2bits_local)
    new_arr = np.tile(seq_as_arr, num_seqs).reshape(num_seqs, length)

    new_arr[idxs_repeat] += subseq_offsets_repeats.flatten()
    new_arr %= num_bases

    # now map back to correct subset of 0,1,2,3 to represent bases
    for base, digit in zip(['A', 'C', 'G', 'T'], range(4)):
        if base not in bases:
            idxs_to_inc = new_arr >= digit
            new_arr[idxs_to_inc] += 1

    return new_arr


def combnr_idxs(length: int, number: int) -> np.ndarray:
    # Gives 2D Boolean numpy array, with `length` columns and (`length` choose `number`) rows,
    # representing all ways to set exactly `number` elements of the row True and the others to False.
    # Useful for indexing into a same-shape numpy array, changing exactly `number` elements in each row.
    #
    # :param length:
    #     number of columns
    # :param number:
    #     number of True values in each row
    # :return:
    #     numpy array, with `length` columns and (`length` choose `number`) rows,
    #     representing all ways to set exactly `number` elements of the row True and the others to False.
    if number == 0:
        return np.array([[False] * length], dtype=bool)
    elif number > length // 2:
        vals = combnr_idxs(length, length - number)
        return np.logical_not(vals)
    x = np.array(np.meshgrid(*([np.arange(0, length)] * number))).T.reshape(-1, number)
    z = np.sum(np.identity(length)[x], 1, dtype=bool).astype(int)
    return np.unique(z[np.sum(z, axis=1) == number], axis=0).astype(bool)


def make_array_with_random_subset_of_dna_seqs_hamming_distance(
        num_seqs: int, dist: int, seq: str, rng: np.random.Generator = default_rng,
        bases: Collection[str] = ('A', 'C', 'G', 'T')) -> np.ndarray:
    """
    Return 2D numpy array with random subset of size `num_seqs` of DNA sequences of given length.
    Bases contains bases to be used: ('A','C','G','T') by default, but can be set to a subset of these.

    Uses the encoding described in the documentation for DNASeqList. The result is a 2D array,
    where each row represents a DNA sequence, and that row has one byte per base.

    Sampled *with* replacement, so the same row may appear twice in the returned array

    :param num_seqs:
        number of sequences to generate
    :param dist:
        Hamming distance to be from `seq`
    :param seq:
        sequence to generate other sequences close to
    :param bases:
        DNA bases to use
    :param rng:
        numpy random number generator (type returned by numpy.random.default_rng())
    :return:
        2D numpy array with random subset of size `num_seqs` of DNA sequences of given length
    """
    if not set(bases) <= {'A', 'C', 'G', 'T'}:
        raise ValueError(f"bases must be a subset of {'A', 'C', 'G', 'T'}; cannot be {bases}")
    if len(bases) == 0:
        raise ValueError('bases cannot be empty')
    elif len(bases) == 1:
        raise ValueError('bases must have at least two elements')

    length = len(seq)

    if dist < 1:
        raise ValueError(f'dist must be positive, but dist = {dist}')
    if dist > length:
        raise ValueError(f'should have dist <= len("{seq}") = {length}, but dist = {dist}')

    num_different_bases = len(bases) - 1

    # map subset of bases used to *prefix* of 0,1,2,3
    base2bits_local = {base: digit for base, digit in zip(bases, range(4))}
    seq_as_arr = seq2arr(seq, base2bits_local=base2bits_local)
    seqs = np.tile(seq_as_arr, num_seqs).reshape(num_seqs, length)

    # for simplicity of modular arithmetic, we use integers 0,...,len(bases)-1 to represent the bases,
    # then map these back to the correct subset of 0,1,2,3 when we are done
    subseq_offsets = rng.integers(low=1, high=num_different_bases + 1, size=(num_seqs, dist))
    assert len(subseq_offsets) == num_seqs

    # all (length choose dist) indices where we could change the bases
    idxs = random_choice_noreplace(np.arange(length), dist, num_seqs, rng)
    assert len(idxs) == num_seqs
    changes = rng.integers(1, num_different_bases + 1, size=idxs.shape, dtype=np.uint8)
    seqs[np.arange(num_seqs)[:, None], idxs] += changes
    seqs = np.mod(seqs, num_different_bases + 1)

    # now map back to correct subset of 0,1,2,3 to represent bases
    for base, digit in zip(['A', 'C', 'G', 'T'], range(4)):
        if base not in bases:
            idxs_to_inc = seqs >= digit
            seqs[idxs_to_inc] += 1

    # use the next two lines to return only unique rows
    # seqs = np.unique(seqs, axis=0)
    # rng.shuffle(seqs)

    return seqs


# def random_hamming(sequence: Union[List[int], np.ndarray], distance: int, number: int,
#                    rng: np.random.Generator) -> np.ndarray:
#     sequence = np.array(sequence, dtype=np.uint8)
#     length = len(sequence)
#     seqrepeats = np.tile(sequence, number).reshape((number, length))
#     places = random_choice_noreplace(np.arange(length), distance, number, rng)
#     changes = rng.integers(1, 4, size=places.shape, dtype=np.uint8)
#     seqrepeats[np.arange(number)[:, None], places] += changes
#     seqrepeats = np.mod(seqrepeats, 4)
#     return seqrepeats


# https://stackoverflow.com/a/59328647/5339430
def random_choice_noreplace(l: np.ndarray, n_sample: int, num_draw: int,
                            rng: np.random.Generator) -> np.ndarray:
    '''
    l: 1-D array or list
    n_sample: sample size for each draw
    num_draw: number of draws

    Intuition: Randomly generate numbers, get the index of the smallest n_sample number for each row.
    '''
    l = np.array(l)
    random_array_floats = rng.random((num_draw, len(l)))
    return l[np.argpartition(random_array_floats, n_sample - 1, axis=-1)[:, :n_sample]]


# @lru_cache(maxsize=10000000)
def longest_common_substring(a1: np.ndarray, a2: np.ndarray, vectorized: bool = True) -> Tuple[int, int, int]:
    """Return start and end indices (a1start, a2start, length) of longest common
    substring (subarray) of 1D arrays a1 and a2."""
    assert len(a1.shape) == 1
    assert len(a2.shape) == 1
    counter = np.zeros(shape=(len(a1) + 1, len(a2) + 1), dtype=np.int)
    a1idx_longest = a2idx_longest = -1
    len_longest = 0

    if vectorized:
        for i1 in range(len(a1)):
            idx = (a2 == a1[i1])
            idx_shifted = np.hstack([[False], idx])
            counter[i1 + 1, idx_shifted] = counter[i1, idx] + 1
        idx_longest = np.unravel_index(np.argmax(counter), counter.shape)
        if idx_longest[0] > 0:
            len_longest = counter[idx_longest]
            a1idx_longest = int(idx_longest[0] - len_longest)
            a2idx_longest = int(idx_longest[1] - len_longest)
    else:
        for i1 in range(len(a1)):
            for i2 in range(len(a2)):
                if a1[i1] == a2[i2]:
                    c = counter[i1, i2] + 1
                    counter[i1 + 1, i2 + 1] = c
                    if c > len_longest:
                        len_longest = c
                        a1idx_longest = i1 + 1 - c
                        a2idx_longest = i2 + 1 - c
    return a1idx_longest, a2idx_longest, len_longest


# @lru_cache(maxsize=10000000)
def longest_common_substrings_singlea1(a1: np.ndarray, a2s: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return start and end indices (a1starts, a2starts, lengths) of longest common
    substring (subarray) of 1D array a1 and rows of 2D array a2s.

    If length[i]=0, then a1starts[i]=a2starts[i]=0 (not -1), so be sure to check
    length[i] to see if any substrings actually matched."""
    assert len(a1.shape) == 1
    assert len(a2s.shape) == 2
    numa2s = a2s.shape[0]
    len_a1 = len(a1)
    len_a2 = a2s.shape[1]
    counter = np.zeros(shape=(len_a1 + 1, numa2s, len_a2 + 1), dtype=np.int)

    for i1 in range(len(a1)):
        idx = (a2s == a1[i1])
        idx_shifted = np.insert(idx, 0, np.zeros(numa2s, dtype=np.bool), axis=1)
        counter[i1 + 1, idx_shifted] = counter[i1, idx] + 1

    counter = np.swapaxes(counter, 0, 1)

    counter_flat = counter.reshape(numa2s, (len_a1 + 1) * (len_a2 + 1))
    idx_longest_raveled = np.argmax(counter_flat, axis=1)
    len_longest = counter_flat[np.arange(counter_flat.shape[0]), idx_longest_raveled]

    idx_longest = np.unravel_index(idx_longest_raveled, dims=(len_a1 + 1, len_a2 + 1))
    a1idx_longest = idx_longest[0] - len_longest
    a2idx_longest = idx_longest[1] - len_longest

    return a1idx_longest, a2idx_longest, len_longest


# @lru_cache(maxsize=10000000)
def longest_common_substrings_product(a1s: np.ndarray, a2s: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return start and end indices (a1starts, a2starts, lengths) of longest common
    substring (subarray) of each pair in the cross product of rows of a1s and a2s.

    If length[i]=0, then a1starts[i]=a2starts[i]=0 (not -1), so be sure to check
    length[i] to see if any substrings actually matched."""
    numa1s = a1s.shape[0]
    numa2s = a2s.shape[0]
    a1s_cp = np.repeat(a1s, numa2s, axis=0)
    a2s_cp = np.tile(a2s, (numa1s, 1))

    a1idx_longest, a2idx_longest, len_longest = _longest_common_substrings_pairs(a1s_cp, a2s_cp)

    a1idx_longest = a1idx_longest.reshape(numa1s, numa2s)
    a2idx_longest = a2idx_longest.reshape(numa1s, numa2s)
    len_longest = len_longest.reshape(numa1s, numa2s)

    return a1idx_longest, a2idx_longest, len_longest


def pair_index(n: int) -> np.ndarray:
    index = np.fromiter(it.chain.from_iterable(it.combinations(range(n), 2)), int, count=n * (n - 1))
    return index.reshape(-1, 2)


def _longest_common_substrings_pairs(a1s: np.ndarray, a2s: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert len(a1s.shape) == 2
    assert len(a2s.shape) == 2
    assert a1s.shape[0] == a2s.shape[0]

    numpairs = a1s.shape[0]

    len_a1 = a1s.shape[1]
    len_a2 = a2s.shape[1]

    counter = np.zeros(shape=(len_a1 + 1, numpairs, len_a2 + 1), dtype=np.int)

    for i1 in range(len_a1):
        a1s_cp_col = a1s[:, i1].reshape(numpairs, 1)
        a1s_cp_col_rp = np.repeat(a1s_cp_col, len_a2, axis=1)

        idx = (a2s == a1s_cp_col_rp)
        idx_shifted = np.hstack([np.zeros(shape=(numpairs, 1), dtype=np.bool), idx])
        counter[i1 + 1, idx_shifted] = counter[i1, idx] + 1

    counter = np.swapaxes(counter, 0, 1)

    counter_flat = counter.reshape(numpairs, (len_a1 + 1) * (len_a2 + 1))
    idx_longest_raveled = np.argmax(counter_flat, axis=1)
    len_longest = counter_flat[np.arange(counter_flat.shape[0]), idx_longest_raveled]

    idx_longest = np.unravel_index(idx_longest_raveled, dims=(len_a1 + 1, len_a2 + 1))
    a1idx_longest = idx_longest[0] - len_longest
    a2idx_longest = idx_longest[1] - len_longest

    return a1idx_longest, a2idx_longest, len_longest


def longest_common_substrings_all_pairs_strings(seqs1: Sequence[str], seqs2: Sequence[str]) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For Python strings"""
    a1s = seqs2arr(seqs1)
    a2s = seqs2arr(seqs2)
    return _longest_common_substrings_pairs(a1s, a2s)


def _strongest_common_substrings_all_pairs_return_energies_and_counter(
        a1s: np.ndarray, a2s: np.ndarray, temperature: float) \
        -> Tuple[np.ndarray, np.ndarray]:
    assert len(a1s.shape) == 2
    assert len(a2s.shape) == 2
    assert a1s.shape[0] == a2s.shape[0]

    numpairs = a1s.shape[0]
    len_a1 = a1s.shape[1]
    len_a2 = a2s.shape[1]
    counter = np.zeros(shape=(len_a1 + 1, numpairs, len_a2 + 1), dtype=np.int)
    energies = np.zeros(shape=(len_a1 + 1, numpairs, len_a2 + 1), dtype=np.float)

    #     if not loop_energies:
    loop_energies = calculate_loop_energies(temperature)

    prev_match_shifted_idxs = None

    for i1 in range(len_a1):
        a1s_col = a1s[:, i1].reshape(numpairs, 1)
        a1s_col_rp = np.repeat(a1s_col, len_a2, axis=1)

        # find matching chars and extend length of substring
        match_idxs = (a2s == a1s_col_rp)
        match_shifted_idxs = np.hstack([np.zeros(shape=(numpairs, 1), dtype=np.bool), match_idxs])
        counter[i1 + 1, match_shifted_idxs] = counter[i1, match_idxs] + 1

        if i1 > 0:
            # calculate energy if matching substring has length > 1
            prev_bases = a1s[:, i1 - 1]
            cur_bases = a1s[:, i1]
            loops = (prev_bases << 2) + cur_bases
            latest_energies = loop_energies[loops].reshape(numpairs, 1)
            latest_energies_rp = np.repeat(latest_energies, len_a2, axis=1)
            match_idxs_false_at_end = np.hstack([match_idxs, np.zeros(shape=(numpairs, 1), dtype=np.bool)])
            both_match_idxs = match_idxs_false_at_end & prev_match_shifted_idxs
            prev_match_shifted_shifted_idxs = np.hstack(
                [np.zeros(shape=(numpairs, 1), dtype=np.bool), prev_match_shifted_idxs])[:, :-1]
            both_match_shifted_idxs = match_shifted_idxs & prev_match_shifted_shifted_idxs
            energies[i1 + 1, both_match_shifted_idxs] = energies[i1, both_match_idxs] + latest_energies_rp[
                both_match_idxs]

        #         prev_match_idxs = match_idxs
        prev_match_shifted_idxs = match_shifted_idxs

    counter = counter.swapaxes(0, 1)
    energies = energies.swapaxes(0, 1)

    return counter, energies


def internal_loop_penalty(n: int, temperature: float) -> float:
    return 1.5 + (2.5 * 0.002 * temperature * math.log(1 + n))


def _strongest_common_substrings_all_pairs(a1s: np.ndarray, a2s: np.ndarray, temperature: float) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    numpairs = a1s.shape[0]
    len_a1 = a1s.shape[1]
    len_a2 = a2s.shape[1]

    counter, energies = _strongest_common_substrings_all_pairs_return_energies_and_counter(a1s, a2s,
                                                                                           temperature)

    counter_flat = counter.reshape(numpairs, (len_a1 + 1) * (len_a2 + 1))
    energies_flat = energies.reshape(numpairs, (len_a1 + 1) * (len_a2 + 1))

    idx_strongest_raveled = np.argmax(energies_flat, axis=1)
    len_strongest = counter_flat[np.arange(counter_flat.shape[0]), idx_strongest_raveled]
    energy_strongest = energies_flat[np.arange(counter_flat.shape[0]), idx_strongest_raveled]

    idx_strongest = np.unravel_index(idx_strongest_raveled, dims=(len_a1 + 1, len_a2 + 1))
    a1idx_strongest = idx_strongest[0] - len_strongest
    a2idx_strongest = idx_strongest[1] - len_strongest

    return a1idx_strongest, a2idx_strongest, len_strongest, energy_strongest


def strongest_common_substrings_all_pairs_string(seqs1: Sequence[str], seqs2: Sequence[str],
                                                 temperature: float) \
        -> Tuple[List[float], List[float], List[float], List[float]]:
    """For Python strings representing DNA; checks for reverse complement matches
    rather than direct matches, and evaluates nearest neighbor energy, returning
    indices lengths, and energies of strongest complementary substrings."""
    a1s = seqs2arr(seqs1)
    a2s = seqs2arr(seqs2)
    a1idx_strongest, a2idx_strongest, len_strongest, energy_strongest = _strongest_common_substrings_all_pairs(
        a1s,
        wc_arr(a2s),
        temperature)
    return list(a1idx_strongest), list(a2idx_strongest), list(len_strongest), list(energy_strongest)


def energies_strongest_common_substrings(seqs1: Sequence[str], seqs2: Sequence[str], temperature: float) \
        -> List[float]:
    a1s = seqs2arr(seqs1)
    a2s = seqs2arr(seqs2)
    a1idx_strongest, a2idx_strongest, len_strongest, energy_strongest = \
        _strongest_common_substrings_all_pairs(a1s, wc_arr(a2s), temperature)
    return list(energy_strongest)


@dataclass
class DNASeqList:
    """
    Represents a list of DNA sequences of identical length. The sequences are stored as a 2D numpy array
    of bytes :py:data:`DNASeqList.seqarr`. Each byte represents a single DNA base (so it is not a compact
    representation; the most significant 6 bits of the byte will always be 0).
    """

    seqarr: np.ndarray
    """
    Uses a (noncompact) internal representation using 8 bits (1 byte, dtype = np.ubyte) per base,
    stored in a numpy 2D array of bytes.
    Each row (axis 0) is a DNA sequence, and each column (axis 1) is a base in a sequence.
    
    The code used is :math:`A \\to 0, C \\to 1, G \\to 2, T \\to 3`.
    """

    numseqs: int
    """Number of DNA sequences (number of rows, axis 0, in :py:data:`DNASeqList.seqarr`)"""

    seqlen: int
    """Length of each DNA sequence (number of columns, axis 1, in :py:data:`DNASeqList.seqarr`)"""

    rng: np.random.Generator
    """Random number generator to use."""

    def __init__(self,
                 length: Optional[int] = None,
                 num_random_seqs: Optional[int] = None,
                 shuffle: bool = False,
                 alphabet: Collection[str] = ('A', 'C', 'G', 'T'),
                 seqs: Optional[Sequence[str]] = None,
                 seqarr: np.ndarray = None,
                 filename: Optional[str] = None,
                 rng: np.random.Generator = default_rng,
                 hamming_distance_from_sequence: Optional[Tuple[int, str]] = None):
        """
        Creates a set of DNA sequences, all of the same length.

        Create either all sequences of a given length if seqs is not specified,
        or all sequences in seqs if seqs is specified. If neither is specified
        then all sequences of length 3 are created.

        *Exactly one* of the following should be specified:

        - `length` (possibly along with `alphabet` and `num_random_seqs`)

        - `seqs`

        - `seqarr`

        - `filename`

        - `hamming_distance_from_sequence` (possibly along with `alphabet` and `num_random_seqs`)

        :param length:
            length of sequences; `num_seqs` and `alphabet` can also be specified along with it
        :param hamming_distance_from_sequence:
            if specified and equal to `(dist, seq)` of type (int, str),
            then only sequences at Hamming distance `dist` from `seq` will be generated.
            Raises error if `length`, `seqs`, `seqarr`, or `filename` is specified.
        :param num_random_seqs:
            number of sequences to generate; if not specified, then all sequences
            of length `length` using bases from `alphabet` are generated.
            Sequences are sampled *with* replacement, so the same sequence may appear twice.
        :param shuffle:
            whether to shuffle sequences
        :param alphabet:
            a subset of {'A', 'C', 'G', 'T'}
        :param seqs:
            sequence (e.g., list or tuple) of strings, all of the same length
        :param seqarr:
            2D NumPy array, with axis 0 moving between sequences,
            and axis 1 moving between consecutive DNA bases in a sequence
        :param filename:
            name of file containing a :any:`DNASeqList`
            as written by :py:meth:`DNASeqList.write_to_file`
        :param rng:
            numpy random number generator (type returned by numpy.random.default_rng())
        """
        for v1, v2 in it.combinations([length, seqs, seqarr, filename, hamming_distance_from_sequence], 2):
            if v1 is not None and v2 is not None:
                raise ValueError('exactly one of length, seqs, seqarr, filename, or '
                                 'hamming_distance_from_sequence must be non-None')
        self.rng = rng
        if seqarr is not None:
            self.seqarr = seqarr
            self._update_size()

        elif seqs is not None:
            if len(seqs) == 0:
                raise ValueError('seqs must have positive length')
            self.seqlen = len(seqs[0])
            for seq in seqs:
                if len(seq) != self.seqlen:
                    raise ValueError('All sequences in seqs must be equal length')
            self.numseqs = len(seqs)
            self.seqarr = seqs2arr(seqs)

        elif filename is not None:
            self._read_from_file(filename)

        elif length is not None:
            if num_random_seqs is None:
                self.seqarr = make_array_with_all_dna_seqs(length=length, bases=alphabet)
            else:
                self.seqarr = make_array_with_random_subset_of_dna_seqs(
                    length=length, num_random_seqs=num_random_seqs, rng=self.rng, bases=alphabet)
            self.seqlen = length
            self.numseqs = len(self.seqarr) if self.seqlen > 0 else 1

        elif hamming_distance_from_sequence is not None:
            dist, seq = hamming_distance_from_sequence
            if num_random_seqs is None:
                self.seqarr = make_array_with_all_dna_seqs_hamming_distance(dist=dist, seq=seq,
                                                                            bases=alphabet)
            else:
                self.seqarr = make_array_with_random_subset_of_dna_seqs_hamming_distance(
                    num_seqs=num_random_seqs, dist=dist, seq=seq, rng=self.rng, bases=alphabet)
            self.seqlen = len(seq)
            self.numseqs = len(self.seqarr) if self.seqlen > 0 else 1

        else:
            raise ValueError('at least one of length, seqs, seqarr, filename, or '
                             'hamming_distance_from_sequence must be specified')

        self.shift = np.arange(2 * (self.seqlen - 1), -1, -2)

        if shuffle:
            self.shuffle()

    def random_choice(self, num: int, rng: np.random.Generator = default_rng,
                      replace: bool = False) -> List[str]:
        """
        Returns random choice of `num` DNA sequence(s) (represented as list of Python strings).

        :param num:
            number of sequences to sample
        :param replace:
            whether to sample with replacement
        :return:
            sampled sequences
        """
        idxs = rng.choice(np.arange(self.numseqs), num, replace=replace)
        seqs = [self[int(idx)] for idx in idxs]
        return seqs

    def random_sequence(self, rng: np.random.Generator = default_rng) -> str:
        """
        Returns random DNA sequence (represented as Python string).

        :return:
            sampled sequence
        """
        idx = int(rng.integers(0, self.numseqs))
        return self[idx]

    def _update_size(self) -> None:
        # updates numseqs and seqlen based on shape of seqarr
        self.numseqs, self.seqlen = self.seqarr.shape

    def __len__(self) -> int:
        return self.numseqs

    def __contains__(self, seq: str) -> bool:
        if len(seq) != self.seqlen:
            return False
        arr = seq2arr(seq)
        return np.any(~np.any(self.seqarr - arr, 1))

    def _read_from_file(self, filename: str) -> None:
        """Reads from fileName in the format defined in writeToFile.
        Only meant to be called from constructor."""
        with open(filename, 'r+') as f:
            first_line = f.readline()
            num_seqs_str, seq_len_str, temperature = first_line.split()
            self.numseqs = int(num_seqs_str)
            self.seqlen = int(seq_len_str)
            self.seqarr = np.empty((self.numseqs, self.seqlen), dtype=np.ubyte)
            for i in range(self.numseqs):
                line = f.readline()
                seq = line.strip()
                self.seqarr[i] = [base2bits[base] for base in seq]

    def write_to_file(self, filename: str) -> None:
        """Writes text file describing DNA sequence list, in format

        numseqs seqlen
        seq1
        seq2
        seq3
        ...

        where numseqs, seqlen are integers, and seq1,
        ... are strings from {A,C,G,T}"""
        with open(filename, 'w+') as f:
            f.write(str(self.numseqs) + ' ' + str(self.seqlen) + '\n')
            for i in range(self.numseqs):
                f.write(self.get_seq_str(i) + '\n')

    def wcenergy(self, idx: int, temperature: float) -> float:
        """Return energy of idx'th sequence binding to its complement."""
        return wcenergy(self.seqarr[idx], temperature)

    def __repr__(self) -> str:
        return 'DNASeqSet(seqs={})'.format(str([self[i] for i in range(self.numseqs)]))

    def __str__(self) -> str:
        if self.numseqs <= 256:
            ret = [self.get_seq_str(i) for i in range(self.numseqs)]
            return ','.join(ret)
        else:
            ret = [self.get_seq_str(i) for i in range(3)] + ['...'] + \
                  [self.get_seq_str(i) for i in range(self.numseqs - 3, self.numseqs)]
            return ','.join(ret)

    def shuffle(self) -> None:
        self.rng.shuffle(self.seqarr)

    def to_list(self) -> List[str]:
        """Return list of strings representing the sequences, e.g. ['ACG','TAA']"""
        return [self.get_seq_str(idx) for idx in range(self.numseqs)]

    def get_seq_str(self, idx: int) -> str:
        """Return idx'th DNA sequence as a string."""
        return arr2seq(self.seqarr[idx])

    def get_seqs_str_list(self, slice_: slice) -> List[str]:
        """Return a list of strings specified by slice."""
        bases_lst = self.seqarr[slice_]
        ret = []
        for bases in bases_lst:
            bases_ch = [bits2base[base] for base in bases]
            ret.append(''.join(bases_ch))
        return ret

    def keep_seqs_at_indices(self, indices: Iterable[int]) -> None:
        """Keeps only sequences at the given indices."""
        if not isinstance(indices, list):
            indices = list(indices)
        self.seqarr = self.seqarr[indices]
        self._update_size()

    def __getitem__(self, slice_: Union[int, slice]) -> Union[str, List[str]]:
        if isinstance(slice_, int):
            return self.get_seq_str(slice_)
        elif isinstance(slice_, slice):
            return self.get_seqs_str_list(slice_)
        else:
            raise ValueError('slice_ must be int or slice')

    def __setitem__(self, idx: int, seq: str) -> None:
        # cannot set on slice
        self.seqarr[idx] = seq2arr(seq)

    def pop(self) -> str:
        """Remove and return last seq, as a string."""
        seq_str = self.get_seq_str(-1)
        self.seqarr = np.delete(self.seqarr, -1, 0)
        self.numseqs -= 1
        return seq_str

    def pop_array(self) -> np.ndarray:
        """Remove and return last seq, as a string."""
        arr = self.seqarr[-1]
        self.seqarr = np.delete(self.seqarr, -1, 0)
        self.numseqs -= 1
        return arr

    def append_seq(self, newseq: str) -> None:
        self.append_arr(seq2arr(newseq))

    def append_arr(self, newarr: np.ndarray) -> None:
        self.seqarr = np.vstack([self.seqarr, newarr])
        self.numseqs += 1

    def sequences_at_hamming_distance(self, sequence: str, distance: int) -> DNASeqList:
        sequence_1d_array = seq2arr(sequence)
        distances = np.sum(np.bitwise_xor(self.seqarr, sequence_1d_array) != 0, axis=1)
        indices_at_distance = distances == distance
        arr = self.seqarr[indices_at_distance]
        return DNASeqList(seqarr=arr)

    def hamming_map(self, sequence: str) -> Dict[int, DNASeqList]:
        """Return dict mapping each length `d` to a :any:`DNASeqList` of sequences that are
        Hamming distance `d` from `seq`."""
        # import time
        # times = []
        # before = time.perf_counter_ns()
        sequence_1d_array = seq2arr(sequence)
        distances = np.sum(np.bitwise_xor(self.seqarr, sequence_1d_array) != 0, axis=1)
        distance_map = {}
        # after_it_prev = time.perf_counter_ns()
        # print(f'time to calculate Hamming distances: {(after_it_prev - before) / 1e6:.1f} ms')
        for distance in range(self.seqlen + 1):
            indices_at_distance = distances == distance
            arr = self.seqarr[indices_at_distance]
            if arr.shape[0] > 0:  # don't bother putting empty array into map
                distance_map[distance] = DNASeqList(seqarr=arr)
            # after_it_next = time.perf_counter_ns()
            # times.append((after_it_next - after_it_prev) / 1e6)
            # after_it_prev = after_it_next
        # after = time.perf_counter_ns()
        # print(f'time spent finding neighbors: {(after - before) / 1e6:.1f} ms')
        # print(f'times in each iteration: {times}')
        return distance_map

    def sublist(self, start: int, end: Optional[int] = None) -> DNASeqList:
        """Return sublist of DNASeqList from `start`, inclusive, to `end`, exclusive.

        If `end` is not specified, goes until the end of the list."""
        if end is None:
            end = self.numseqs
        arr = self.seqarr[start:end]
        return DNASeqList(seqarr=arr)

    # def filter_hamming(self, threshold: int) -> None:
    #     seq = self.pop_array()
    #     arr_keep = np.array([seq])
    #     self.shuffle()
    #     while self.seqarr.shape[0] > 0:
    #         seq = self.pop_array()
    #         while self.seqarr.shape[0] > 0:
    #             hamming_min = np.min(np.sum(np.bitwise_xor(arr_keep, seq) != 0, axis=1))
    #             too_close = (hamming_min < threshold)
    #             if not too_close:
    #                 break
    #             seq = self.pop_array()
    #         arr_keep = np.vstack([arr_keep, seq])
    #     self.seqarr = arr_keep
    #     self.numseqs = self.seqarr.shape[0]
    #
    # def hamming_min(self, arr: np.ndarray) -> int:
    #     """Returns minimum Hamming distance between arr and any sequence in
    #     this DNASeqList."""
    #     distances = np.sum(np.bitwise_xor(self.seqarr, arr) != 0, axis=1)
    #     return np.min(distances)

    def filter_energy(self, low: float, high: float, temperature: float) -> DNASeqList:
        """Return new DNASeqList with seqs whose wc complement energy is within
        the given range."""
        wcenergies = calculate_wc_energies(self.seqarr, temperature)
        within_range = (low <= wcenergies) & (wcenergies <= high)
        new_seqarr = self.seqarr[within_range]
        return DNASeqList(seqarr=new_seqarr)

    def energies(self, temperature: float) -> np.ndarray:
        """
        :param temperature:
            temperature in Celsius
        :return:
            nearest-neighbor energies of each sequence with its perfect Watson-Crick complement
        """
        wcenergies = calculate_wc_energies(self.seqarr, temperature)
        return wcenergies

    def filter_end_gc(self) -> DNASeqList:
        """Remove any sequence with A or T on the end. Also remove domains that
        do not have an A or T either next to that base, or one away. Otherwise
        we could get a domain ending in {C,G}^3, which, placed next to any
        domain ending in C or G, will create a substring in {C,G}^4 and be
        rejected if we are filtering those."""
        left = self.seqarr[:, 0]
        right = self.seqarr[:, -1]
        left_p1 = self.seqarr[:, 1]
        left_p2 = self.seqarr[:, 2]
        right_m1 = self.seqarr[:, -2]
        right_m2 = self.seqarr[:, -3]
        abits = base2bits['A']
        cbits = base2bits['C']
        gbits = base2bits['G']
        tbits = base2bits['T']
        good = (((left == cbits) | (left == gbits)) & ((right == cbits) | (right == gbits)) &
                ((left_p1 == abits) | (left_p1 == tbits) | (left_p2 == abits) | (left_p2 == tbits)) &
                ((right_m1 == abits) | (right_m1 == tbits) | (right_m2 == abits) | (right_m2 == tbits)))
        seqarrpass = self.seqarr[good]
        return DNASeqList(seqarr=seqarrpass)

    def filter_end_at(self, gc_near_end: bool = False) -> DNASeqList:
        """Remove any sequence with C or G on the end. Also, if gc_near_end is True,
        remove domains that do not have an C or G either next to that base,
        or one away, to prevent breathing."""
        left = self.seqarr[:, 0]
        right = self.seqarr[:, -1]
        abits = base2bits['A']
        tbits = base2bits['T']
        good = ((left == abits) | (left == tbits)) & ((right == abits) | (right == tbits))
        if gc_near_end:
            cbits = base2bits['C']
            gbits = base2bits['G']
            left_p1 = self.seqarr[:, 1]
            left_p2 = self.seqarr[:, 2]
            right_m1 = self.seqarr[:, -2]
            right_m2 = self.seqarr[:, -3]
            good = (good &
                    ((left_p1 == cbits) | (left_p1 == gbits) | (left_p2 == cbits) | (left_p2 == gbits)) &
                    ((right_m1 == cbits) | (right_m1 == gbits) | (right_m2 == cbits) | (right_m2 == gbits)))
        seqarrpass = self.seqarr[good]
        return DNASeqList(seqarr=seqarrpass)

    def filter_base_nowhere(self, base: str) -> DNASeqList:
        """Remove any sequence that has given base anywhere."""
        good = (self.seqarr != base2bits[base]).all(axis=1)
        seqarrpass = self.seqarr[good]
        return DNASeqList(seqarr=seqarrpass)

    def filter_base_count(self, base: str, low: int, high: int) -> DNASeqList:
        """Remove any sequence not satisfying low <= #base <= high."""
        sumarr = np.sum(self.seqarr == base2bits[base], axis=1)
        good = (low <= sumarr) & (sumarr <= high)
        seqarrpass = self.seqarr[good]
        return DNASeqList(seqarr=seqarrpass)

    def filter_base_at_pos(self, pos: int, base: str) -> DNASeqList:
        """Remove any sequence that does not have given base at position pos."""
        mid = self.seqarr[:, pos]
        good = (mid == base2bits[base])
        seqarrpass = self.seqarr[good]
        return DNASeqList(seqarr=seqarrpass)

    def filter_substring(self, subs: Sequence[str]) -> DNASeqList:
        """Remove any sequence with any elements from subs as a substring."""
        if len(set([len(sub) for sub in subs])) != 1:
            raise ValueError('All substrings in subs must be equal length: %s' % subs)
        sublen = len(subs[0])
        subints = [[base2bits[base] for base in sub] for sub in subs]
        powarr = [4 ** k for k in range(sublen)]
        subvals = np.dot(subints, powarr)
        toeplitz = create_toeplitz(self.seqlen, sublen)
        convolution = np.dot(toeplitz, self.seqarr.transpose())
        passall = np.ones(self.numseqs, dtype=np.bool)
        for subval in subvals:
            passsub = np.all(convolution != subval, axis=0)
            passall = passall & passsub
        seqarrpass = self.seqarr[passall]
        return DNASeqList(seqarr=seqarrpass)

    def filter_seqs_by_g_quad(self) -> DNASeqList:
        """Removes any sticky ends with 4 G's in a row (a G-quadruplex)."""
        return self.filter_substring(['GGGG'])

    def filter_seqs_by_g_quad_c_quad(self) -> DNASeqList:
        """Removes any sticky ends with 4 G's or C's in a row (a quadruplex)."""
        return self.filter_substring(['GGGG', 'CCCC'])

    def index(self, sequence: Union[str, np.ndarray]) -> int:
        # finds index of sequence in (rows of) self.seqarr
        # raises IndexError if not present
        # taken from https://stackoverflow.com/questions/40382384/finding-a-matching-row-in-a-numpy-matrix
        if isinstance(sequence, str):
            sequence = seq2arr(sequence)
        matching_condition = (self.seqarr == sequence).all(axis=1)
        all_indices_tuple = np.where(matching_condition)
        all_indices = all_indices_tuple[0]
        first_index = all_indices[0]
        return int(first_index)


def create_toeplitz(seqlen: int, sublen: int, indices: Optional[Sequence[int]] = None) -> np.ndarray:
    """Creates a toeplitz matrix, useful for finding subsequences.

    `seqlen` is length of larger sequence; `sublen` is length of substring we're checking for.
    If `indices` is None, then all rows are created, otherwise only rows for checking those indices
    are created."""
    powarr = [4 ** k for k in range(sublen)]
    if indices is None:
        rows = list(range(seqlen - (sublen - 1)))
    else:
        rows = sorted(list(set(indices)))
        for idx in rows:
            if idx < 0:
                raise ValueError(f'index must be nonnegative, but {idx} is not; all indices = {indices}')
            if idx >= seqlen - (sublen - 1):
                raise ValueError(f'index must be less than {seqlen - (sublen - 1)}, '
                                 f'but {idx} is not; all indices = {indices}')
    num_rows = len(rows)
    num_cols = seqlen
    toeplitz = np.zeros((num_rows, num_cols), dtype=np.int)
    toeplitz[:, 0:sublen] = [powarr] * num_rows
    shift = list(rows)
    for i in range(len(rows)):
        toeplitz[i] = np.roll(toeplitz[i], shift[i])
    return toeplitz


@lru_cache(maxsize=32)
def calculate_loop_energies(temperature: float, negate: bool = False) -> np.ndarray:
    """Get SantaLucia and Hicks nearest-neighbor loop energies for given temperature,
    1 M Na+. """
    energies = (_dH - (temperature + 273.15) * _dS / 1000.0)
    if negate:
        energies = -energies
    return energies
    # SantaLucia & Hicks' values are in cal/mol/K for dS, and kcal/mol for dH.
    # Here we divide dS by 1000 to get the RHS term into units of kcal/mol/K
    # which gives an overall dG in units of kcal/mol.
    # One reason we want dG to be in units of kcal/mol is to
    # give reasonable/readable numbers close to 0 for dG(Assembly).
    # The reason we might want to flip the sign is that, by convention, in the kTAM, G_se
    # (which is computed from the usually negative dG here) is usually positive.


# _dH and _dS come from Table 1 in SantaLucia and Hicks, Annu Rev Biophys Biomol Struct. 2004;33:415-40.
#                 AA    AC    AG    AT    CA    CC     CG    CT
_dH = np.array([-7.6, -8.4, -7.8, -7.2, -8.5, -8.0, -10.6, -7.8,
                # GA    GC    GG    GT    TA    TC    TG    TT
                -8.2, -9.8, -8.0, -8.4, -7.2, -8.2, -8.5, -7.6],
               dtype=np.float32)

#                  AA     AC     AG     AT     CA     CC     CG     CT
_dS = np.array([-21.3, -22.4, -21.0, -20.4, -22.7, -19.9, -27.2, -21.0,
                #  GA     GC     GG     GT     TA     TC     TG     TT
                -22.2, -24.4, -19.9, -22.4, -21.3, -22.2, -22.7, -21.3],
               dtype=np.float32)

#  AA  AC  AG  AT  CA  CC  CG  CT  GA  GC  GG  GT  TA  TC  TG  TT
#  00  01  02  03  10  11  12  13  20  21  22  23  30  31  32  34

# nearest-neighbor energies for Watson-Crick complements at 37C
# (Table 1 in SantaLucia and Hicks 2004)
# ordering of array is
# #                      AA    AC    AG    AT    CA    CC    CG    CT
# _nndGwc = np.array([-1.00,-1.44,-1.28,-0.88,-1.45,-1.84,-2.17,-1.28,
#                     #  GA    GC    GG    GT    TA    TC    TG    TT
#                     -1.30,-2.24,-1.84,-1.44,-0.58,-1.30,-1.45,-1.00],
#                    dtype=np.float32)
#                    # AA   AC   AG   AT   CA   CC   CG   CT
# _nndGwc = np.array([1.00,1.44,1.28,0.88,1.45,1.84,2.17,1.28,
#                    # GA   GC   GG   GT   TA   TC   TG   TT
#                    1.30,2.24,1.84,1.44,0.58,1.30,1.45,1.00],
#                   dtype=np.float32)
# _nndGwcStr = {'AA':1.00,'AC':1.44,'AG':1.28,'AT':0.88,'CA':1.45,'CC':1.84,
#              'CG':2.17,'CT':1.28,'GA':1.30,'GC':2.24,'GG':1.84,'GT':1.44,
#              'TA':0.58,'TC':1.30,'TG':1.45,'TT':1.00}

# nearest-neighbor energies for single mismatches (Table 2 in SantaLucia)
# ordering of array is
#                  # GA/CA GA/CG    AG    AT    CA    CC    CG    CT   GA    GC    GG    GT    TA   TC    TG   TT
# _nndGsmm = np.array([0.17,-1.44,-1.28,-0.88,-1.45,-1.84,-2.17,-1.28,-1.3,-2.24,-1.84,-1.44,-0.58,-1.3,-1.45,-1.0], dtype=np.float32)

_all_pairs = [((i << 2) + j, bits2base[i] + bits2base[j])
              for i in range(4) for j in range(4)]


@lru_cache(maxsize=32)
def calculate_loop_energies_dict(temperature: float, negate: bool = False) -> Dict[str, float]:
    loop_energies = calculate_loop_energies(temperature, negate)
    return {pair[1]: loop_energies[pair[0]] for pair in _all_pairs}


@lru_cache(maxsize=100000)
def wcenergy(seq: str, temperature: float, negate: bool = False) -> float:
    """Return the wc energy of seq binding to its complement."""
    loop_energies = calculate_loop_energies_dict(temperature, negate)
    return sum(loop_energies[seq[i:i + 2]] for i in range(len(seq) - 1))


def wcenergies_str(seqs: Sequence[str], temperature: float, negate: bool = False) -> List[float]:
    seqarr = seqs2arr(seqs)
    return list(calculate_wc_energies(seqarr, temperature, negate))


def wcenergy_str(seq: str, temperature: float, negate: bool = False) -> float:
    seqarr = seqs2arr([seq])
    return list(calculate_wc_energies(seqarr, temperature, negate))[0]


def hash_ndarray(arr: np.ndarray) -> int:
    writeable = arr.flags.writeable
    if writeable:
        arr.flags.writeable = False
    h = hash(bytes(arr.data))  # hash(arr.data)
    arr.flags.writeable = writeable
    return h


CACHE_WC = False
_calculate_wc_energies_cache: Optional[np.ndarray] = None
_calculate_wc_energies_cache_hash: int = 0


def calculate_wc_energies(seqarr: np.ndarray, temperature: float, negate: bool = False) -> np.ndarray:
    """Calculate and store in an array all energies of all sequences in seqarr
    with their Watson-Crick complements."""
    global _calculate_wc_energies_cache
    global _calculate_wc_energies_cache_hash
    if CACHE_WC and _calculate_wc_energies_cache is not None:
        if _calculate_wc_energies_cache_hash == hash_ndarray(seqarr):
            return _calculate_wc_energies_cache
    loop_energies = calculate_loop_energies(temperature, negate)
    left_index_bits = seqarr[:, :-1] << 2
    right_index_bits = seqarr[:, 1:]
    pair_indices = left_index_bits + right_index_bits
    pair_energies = loop_energies[pair_indices]
    energies: np.ndarray = np.sum(pair_energies, axis=1)
    if CACHE_WC:
        _calculate_wc_energies_cache = energies
        _calculate_wc_energies_cache_hash = hash_ndarray(_calculate_wc_energies_cache)
    return energies


def wc_arr(seqarr: np.ndarray) -> np.ndarray:
    """Return numpy array of complements of sequences in `seqarr`."""
    return (3 - seqarr)[:, ::-1]


def prefilter_length_10_11(low_dg: float, high_dg: float, temperature: float, end_gc: bool,
                           convert_to_list: bool = True) \
        -> Union[Tuple[List[str], List[str]], Tuple[DNASeqList, DNASeqList]]:
    """Return sequences of length 10 and 11 with wc energies between given values."""
    s10: DNASeqList = DNASeqList(length=10)
    s11: DNASeqList = DNASeqList(length=11)
    s10 = s10.filter_energy(low=low_dg, high=high_dg, temperature=temperature)
    s11 = s11.filter_energy(low=low_dg, high=high_dg, temperature=temperature)
    forbidden_subs = [f'{a}{b}{c}{d}' for a in ['G', 'C']
                      for b in ['G', 'C']
                      for c in ['G', 'C']
                      for d in ['G', 'C']]
    s10 = s10.filter_substring(forbidden_subs)
    s11 = s11.filter_substring(forbidden_subs)
    if end_gc:
        print(
            'Removing any domains that end in either A or T; '
            'also ensuring every domain has an A or T within 2 indexes of the end')
        s10 = s10.filter_end_gc()
        s11 = s11.filter_end_gc()
    for seqs in (s10, s11):
        if len(seqs) == 0:
            raise ValueError(
                f'low_dg {low_dg:.2f} and high_dg {high_dg:.2f} too strict! '
                f'no sequences of length {seqs.seqlen} found')
    return (s10.to_list(), s11.to_list()) if convert_to_list else (s10, s11)


def all_cats(seq: Sequence[int], seqs: Sequence[int]) -> np.ndarray:
    """
    Return all sequences obtained by concatenating seq to either end of a sequence in seqs.

    For example,

    .. code-block:: Python

        all_cats([0,1,2,3], [[3,3,3], [0,0,0]])

    returns the numpy array

    .. code-block:: Python

        [[0,1,2,3,3,3,3],
         [3,3,3,0,1,2,3],
         [0,1,2,3,0,0,0],
         [0,0,0,0,1,2,3]]
    """
    seqarr = np.asarray([seq])
    seqsarr = np.asarray(seqs)
    ar = seqarr.repeat(seqsarr.shape[0], axis=0)
    ret = np.concatenate((seqsarr, ar), axis=1)
    ret2 = np.concatenate((ar, seqsarr), axis=1)
    ret = np.concatenate((ret, ret2))
    return ret
