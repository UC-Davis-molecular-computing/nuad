"""
This module defines types for helping to define DNA sequence design constraints.

The key classes are :any:`Design`, :any:`Strand`, :any:`Domain` to define a DNA design,
and various subclasses of :any:`Constraint`, such as :any:`StrandConstraint` or :any:`StrandPairConstraint`,
to define constraints on the sequences assigned to each :any:`Domain` when calling
:meth:`search.search_for_dna_sequences`.

Also important are two other types of constraints
(not subclasses of :any:`Constraint`), which are used prior to the search to determine if it is even
legal to use a DNA sequence: subclasses of the abstract base class :any:`NumpyConstraint`,
and  :any:`SequenceConstraint`, an alias for a function taking a string as input and returning a bool.

See the README on the GitHub page for more detailed explaination of these classes:
https://github.com/UC-Davis-molecular-computing/dsd#data-model
"""

from __future__ import annotations

import dataclasses
import enum
import os
import math
import json
from typing import List, Set, Optional, Dict, Callable, Iterable, Tuple, Union, Collection, TypeVar, Any, \
    cast, Generic, DefaultDict, FrozenSet, Iterator, Sequence, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict
import itertools
import logging
from multiprocessing.pool import ThreadPool
from numbers import Number
from enum import Enum, auto, unique

import numpy as np  # noqa
from ordered_set import OrderedSet

import scadnano as sc  # type: ignore

import nuad.vienna_nupack as dv
import nuad.np as dn
import nuad.modifications as dm
from nuad.json_noindent_serializer import JSONSerializable, json_encode, NoIndent

# need typing_extensions package prior to Python 3.8 to get Protocol object
try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

# from dsd.stopwatch import Stopwatch

try:
    from scadnano import Design as scDesign  # type: ignore
    from scadnano import Strand as scStrand  # type: ignore
    from scadnano import Domain as scDomain  # type: ignore
    from scadnano import m13 as m13_sc  # type: ignore
except ModuleNotFoundError:
    scDesign = Any
    scStrand = Any
    scDomain = Any

name_key = 'name'
sequence_key = 'sequence'
fixed_key = 'fixed'
label_key = 'label'
strands_key = 'strands'
domains_key = 'domains'
domain_pools_key = 'domain_pools'
domain_pools_num_sampled_key = 'domain_pools_num_sampled'
domain_names_key = 'domain_names'
starred_domain_indices_key = 'starred_domain_indices'
group_key = 'group'
domain_pool_name_key = 'pool_name'
length_key = 'length'
substring_length_key = 'substring_length'
except_indices_key = 'except_start_indices'
circular_key = 'circular'
strand_name_in_strand_pool_key = 'strand_name'
sequences_key = 'sequences'
replace_with_close_sequences_key = 'replace_with_close_sequences'
hamming_probability_key = 'hamming_probability'
possible_sequences_key = 'possible_sequences'
num_sequences_to_generate_upper_limit_key = 'num_sequences_to_generate_upper_limit'
num_sequences_to_generate_key = 'num_sequences_to_generate'
rng_state_key = 'rng_state'

idt_key = 'idt'
idt_scale_key = 'scale'
idt_purification_key = 'purification'
idt_plate_key = 'plate'
idt_well_key = 'well'

default_idt_scale = "25nm"
default_idt_purification = "STD"

T = TypeVar('T')
KeyFunction = Callable[[T], Any]

# Complex = Tuple['Strand', ...]
# """A Complex is a group of :any:`Strand`'s, in general that we expect to be bound by complementary
# :any:`Domain`'s."""

all_dna_bases: Set[str] = {'A', 'C', 'G', 'T'}
"""
Set of all DNA bases.
"""


class M13Variant(enum.Enum):
    """Variants of M13mp18 viral genome. "Standard" variant is p7249. Other variants are longer."""

    p7249 = "p7249"
    """"Standard" variant of M13mp18; 7249 bases long, available from, for example

    https://www.tilibit.com/collections/scaffold-dna/products/single-stranded-scaffold-dna-type-p7249

    https://www.neb.com/products/n4040-m13mp18-single-stranded-dna

    http://www.bayoubiolabs.com/biochemicat/vectors/pUCM13/ 
    """  # noqa

    p7560 = "p7560"
    """Variant of M13mp18 that is 7560 bases long. Available from, for example

    https://www.tilibit.com/collections/scaffold-dna/products/single-stranded-scaffold-dna-type-p7560
    """

    p8064 = "p8064"
    """Variant of M13mp18 that is 8064 bases long. Available from, for example

    https://www.tilibit.com/collections/scaffold-dna/products/single-stranded-scaffold-dna-type-p8064
    """

    def length(self) -> int:
        """
        :return: length of this variant of M13 (e.g., 7249 for variant :data:`M13Variant.p7249`)
        """
        if self is M13Variant.p7249:
            return 7249
        if self is M13Variant.p7560:
            return 7560
        if self is M13Variant.p8064:
            return 8064
        raise AssertionError('should be unreachable')

    def scadnano_variant(self) -> sc.M13Variant:
        if self is M13Variant.p7249:
            return sc.M13Variant.p7249
        if self is M13Variant.p7560:
            return sc.M13Variant.p7560
        if self is M13Variant.p8064:
            return sc.M13Variant.p8064
        raise AssertionError('should be unreachable')


def m13(rotation: int = 5587, variant: M13Variant = M13Variant.p7249) -> str:
    """
    The M13mp18 DNA sequence (commonly called simply M13).

    By default, starts from cyclic rotation 5587 
    (with 0-based indexing;  commonly this is called rotation 5588, which assumes that indexing begins at 1), 
    as defined in
    `GenBank <https://www.ncbi.nlm.nih.gov/nuccore/X02513.1>`_.

    By default, returns the "standard" variant of consisting of 7249 bases, sold by companies such as  
    `Tilibit <https://cdn.shopify.com/s/files/1/1299/5863/files/Product_Sheet_single-stranded_scaffold_DNA_type_7249_M1-10.pdf?14656642867652657391>`_
    and
    `New England Biolabs <https://www.neb.com/~/media/nebus/page%20images/tools%20and%20resources/interactive%20tools/dna%20sequences%20and%20maps/m13mp18_map.pdf>`_.

    For a more detailed discussion of why the default rotation 5587 of M13 is used,
    see 
    `Supplementary Note S8 <http://www.dna.caltech.edu/Papers/DNAorigami-supp1.linux.pdf>`_ 
    in
    [`Folding DNA to create nanoscale shapes and patterns. Paul W. K. Rothemund, Nature 440:297-302 (2006) <http://www.nature.com/nature/journal/v440/n7082/abs/nature04586.html>`_].

    :param rotation: rotation of circular strand. Valid values are 0 through length-1.
    :param variant: variant of M13 strand to use
    :return: M13 strand sequence
    """  # noqa
    return m13_sc(rotation=rotation, variant=variant.scadnano_variant())


def m13_substrings_of_length(length: int, except_indices: Iterable[int] = tuple(range(5514, 5557)),
                             variant: M13Variant = M13Variant.p7249) -> List[str]:
    """
    *WARNING*: This function was previously recommended to use with :any:`DomainPool.possible_sequences`
    to specify possible rotations of M13 to use. However, it creates a large file size to
    write all those sequences to disk on every update in the search. A better method now exists
    to specify this, which is to specify a :any:`SubstringSampler` object as the value for
    :any:`DomainPool.possible_sequences` instead of calling this function.

    Return all substrings of the M13mp18 DNA sequence of length `length`,
    except those overlapping indices in `except_start_indices`.

    This is useful with the field :data:`DomainPool.possible_sequences`, when one strand in the
    :any:`Design` represents a small portion of the full M13 sequence,
    and part of the sequence design process is to choose a rotation of M13 to use.
    One can set that strand to have a single :any:`Domain`,
    which contains dependent subdomains (those with :data:`Domain.dependent` set to True).
    These subdomains are the smaller domains where M13 attaches to other :any:`Strand`'s in the
    :any:`Design`. Then, give the parent :any:`Domain` a :any:`DomainPool` with
    :data:`DomainPool.possible_sequences` set to the return value of this function,
    to allow the search to explore different rotations of M13.

    For example, suppose `m13_subdomains` is a list containing :any:`Domain`'s from the :any:`Design`,
    which are consecutive subdomains of M13 from 5' to 3' (all with :data:`Domain.dependent` set to True),
    and `m13_length` is the sum of their lengths (note this needs to be calculated manually since the
    following code assumes no :any:`Domain` in `m13_subdomains` has a :any:`DomainPool` yet, thus none
    yet have a length).
    Then the following code creates a :any:`Strand` representing the M13 portion
    that binds to other :any:`Strand`'s in the :any:`Design`.

    .. code-block:: python

        m13_subdomains = # subdomains of M13 used in the design
        m13_length = # sum of lengths of domains in m13_subdomains
        m13_substrings = dc.m13_substrings_of_length(m13_length)
        m13_domain_pool = dc.DomainPool(name='m13 domain pool', possible_sequences=m13_substrings)
        m13_domain = dc.Domain(name='m13', subdomains=m13_subdomains, pool=m13_domain_pool)
        m13_strand = dc.Strand(name='m13', domains=[m13_domain])

    :param length:
        length of substrings to return
    :param except_indices:
        Indices of M13 to avoid in any part of the substring.
        If not specified, based on `length`, indices 5514-5556 are avoided,
        which are known to contain a long hairpin.
        (When using 1-based indexing, these are indices 5515-5557.)
        For example, if `length` = 10, then the *starting* indices of substrings will avoid the list
        [5505, 5506, ..., 5556]
    :param variant:
        :any:`M13Variant` to use
    :return:
        All substrings of the M13mp18 DNA sequence, except those that overlap any index in
        `except_start_indices`.
    """
    m13_ = m13_sc(rotation=0, variant=variant)

    # append start of m13 to its end to help with circular condition
    m13_ += m13_[:length]

    # add indices beyond 7248 (or whatever is the length) that correspond to indices near the start
    extended_except_indices = list(except_indices)
    for skip_idx in except_indices:
        if skip_idx > length:
            break
        extended_except_indices.append(skip_idx + variant.length())

    substrings = []
    for start_idx in range(variant.length()):
        end_idx = start_idx + length
        skip = False
        for skip_idx in except_indices:
            if start_idx <= skip_idx < end_idx:
                skip = True
                break
        if skip:
            continue
        substring = m13_[start_idx:end_idx]
        substrings.append(substring)

    return substrings


def default_score_transfer_function(x: float) -> float:
    """
    A quadratic transfer function.
    
    :return:
        max(0.0, x^2)
    """
    return max(0.0, x ** 2)


logger = logging.Logger('dsd', level=logging.DEBUG)
"""
Global logger instance used throughout dsd.

Call ``logger.removeHandler(logger.handlers[0])`` to stop screen output (assuming that you haven't added
or removed any handlers to the dsd logger instance already; by default there is one StreamHandler, and
removing it will stop screen output).

Call ``logger.addHandler(logging.FileHandler(filename))`` to direct to a file.
"""


def _configure_logger() -> None:
    screen_handler = logging.StreamHandler()
    screen_handler.setLevel(logging.INFO)
    logger.addHandler(screen_handler)


_configure_logger()


def all_pairs(values: Iterable[T],
              with_replacement: bool = True,
              where: Callable[[T, T], bool] = lambda _, __: True) -> List[Tuple[T, T]]:
    """
    Strongly typed function to get list of all pairs from `iterable`. (for using with mypy)

    :param values:
        Iterable of values.
    :param with_replacement:
        Whether to include self pairs, i.e., pairs (a,a)
    :param where:
        Predicate indicating whether to include a specific pair.
        Must take two parameters, each of type T, and return a bool.
    :return:
        List of all pairs of values from `iterable`.
    """

    def where_tuple(pair: Tuple[T, T]) -> bool:
        item1, item2 = pair
        return where(item1, item2)

    return list(all_pairs_iterator(values, with_replacement=with_replacement, where=where_tuple))


def all_pairs_iterator(values: Iterable[T],
                       with_replacement: bool = True,
                       where: Callable[[Tuple[T, T]], bool] = lambda _: True) -> Iterator[Tuple[T, T]]:
    """
    Strongly typed function to get iterator of all pairs from `iterable`. (for using with mypy)

    This is WITH replacement; to specify without replacement, set `with_replacement` = False

    :param values:
        Iterable of values.
    :param with_replacement:
        Whether to include self pairs, i.e., pairs (a,a)
    :param where:
        Predicate indicating whether to include a specific pair.
    :return:
        Iterator of all pairs of values from `iterable`.
        Unlike :py:meth:`all_pairs`, which returns a list,
        the iterator returned may be iterated over only ONCE.
    """
    comb_iterator = itertools.combinations_with_replacement if with_replacement else itertools.combinations
    it = cast(Iterator[Tuple[T, T]],
              filter(where, comb_iterator(values, 2)))  # noqa
    return it


SequenceConstraint = Callable[[str], bool]
"""
Constraint that applies to a DNA sequence; the difference between this an a :any:`DomainConstraint` is
that these are applied before a sequence is assigned to a :any:`Domain`, so the constraint can only
be based on the DNA sequence, and not, for instance, on the :any:`Domain`'s :any:`DomainPool`.

Consequently :any:`SequenceConstraint`'s, like :any:`NumpyConstraint`'s, are treated differently than
subtypes of :any:`Constraint`, since a DNA sequence failing any :any:`SequenceConstraint`'s or
:any:`NumpyConstraint`'s is never allowed to be assigned into any :any:`Domain`.

The difference with :any:`NumpyConstraint` is that a :any:`NumpyConstraint` requires one to express the
constraint in a way that is efficient for the linear algebra operations of numpy. If you cannot figure out
how to do this, a :any:`SequenceConstraint` can be expressed in pure Python, but typically will be much
slower to apply than a :any:`NumpyConstraint`.
"""


# The Mypy error being ignored is a bug and is described here:
# https://github.com/python/mypy/issues/5374#issuecomment-650656381
@dataclass  # type: ignore
class NumpyConstraint(ABC):
    """
    Abstract base class for numpy constraints. These are constraints that can be efficiently encoded
    as numpy operations on 2D arrays of bytes representing DNA sequences, through the class
    :any:`np.DNASeqList` (which uses such a 2D array as the field :py:data:`np.DNASeqList.seqarr`).

    Subclasses should set the value self.name, inherited from this class.

    Pre-made subclasses of :any:`NumpyConstraint` provided in this library,
    such as :any:`RestrictBasesConstraint` or :any:`NearestNeighborEnergyConstraint`,
    are dataclasses (https://docs.python.org/3/library/dataclasses.html).
    There is no requirement that your custom subclasses be dataclasses, but since the subclasses will
    inherit the field :py:data:`NumpyConstraint.name`, you can easily make them dataclasses to get,
    for example, free ``repr`` and ``str`` implementations. See the source code for the example subclasses.
    """

    name: str = field(init=False, default='TODO: give a concrete name to this NumpyConstraint')
    """Name of this :any:`NumpyConstraint`."""

    @abstractmethod
    def remove_violating_sequences(self, seqs: dn.DNASeqList) -> dn.DNASeqList:
        """
        Subclasses should override this method.

        Since these are constraints that use numpy, generally they will access the numpy ndarray instance
        `seqs.seqarr`, operate on it, and then create a new :any:`np.DNASeqList` instance via the constructor
        :any:`np.DNASeqList` taking an numpy ndarray as input.

        See the source code of included constraints for examples, such as
        :py:meth:`NearestNeighborEnergyConstraint.remove_violating_sequences`
        or
        :py:meth:`BaseCountConstraint.remove_violating_sequences`.
        These are usually quite tricky to write, requiring one to think in terms of linear algebra
        operations. The code tends not to be easy to read. But when a constraint can be expressed
        in this way, it is typically *very* fast to apply; many millions of sequences can
        be processed in a few seconds.

        :param seqs:
            :any:`np.DNASeqList` object representing DNA sequences
        :return:
            a new :any:`np.DNASeqList` object representing the DNA sequences in `seqs` that
            satisfy the constraint
        """
        pass


@dataclass
class RestrictBasesConstraint(NumpyConstraint):
    """
    Restricts the sequence to use only a subset of bases. This can be used to implement
    a so-called "three-letter code", for instance, in which a certain subset of :any:`Strand` uses only the
    bases A, T, C (and :any:`Strand`'s with complementary :any:`Domain` use only A, T, G), to help
    reduce secondary structure of those :any:`Strand`'s.
    See for example Supplementary Section S1.1 of
    "Scaling Up Digital Circuit Computation with DNA Strand Displacement Cascades", Qian and Winfree,
    *Science* 332:1196–1201, 2011.
    DOI: 10.1126/science.1200520,
    https://science.sciencemag.org/content/332/6034/1196,
    http://www.qianlab.caltech.edu/seesaw_digital_circuits2011_SI.pdf

    Note, however, that this is a constraint :any:`Domain`'s, not :any:`Strand`'s, so for a three-letter
    code to work, you must take care not to mixed :any:`Domain`'s on a :any:`Strand` that will use
    different alphabets.
    """  # noqa

    bases: Collection[str]
    """Bases to use. Must be a strict subset of {'A', 'C', 'G', 'T'} with at least two bases."""

    def __post_init__(self) -> None:
        self.name = 'restrict_bases'
        if not set(self.bases) < {'A', 'C', 'G', 'T'}:
            raise ValueError("bases must be a proper subset of {'A', 'C', 'G', 'T'}; "
                             f'cannot be {self.bases}')
        if len(self.bases) <= 1:
            raise ValueError('bases cannot be size 1 or smaller')

    def remove_violating_sequences(self, seqs: dn.DNASeqList) -> dn.DNASeqList:
        """Should never be called directly; it is handled specially by the library when initially
        generating sequences."""
        raise AssertionError('This should never be called directly.')


@dataclass
class NearestNeighborEnergyConstraint(NumpyConstraint):
    """
    This constraint calculates the nearest-neighbor binding energy of a domain with its perfect complement
    (summing over all length-2 substrings of the domain's sequence),
    using parameters from the 2004 Santa-Lucia and Hicks paper
    (https://www.annualreviews.org/doi/abs/10.1146/annurev.biophys.32.110601.141800,
    see Table 1, and example on page 419).
    It rejects any sequences whose energy according to this sum is outside the range
    [:py:data:`NearestNeighborEnergyConstraint.low_energy`,
    :py:data:`NearestNeighborEnergyConstraint.high_energy`].
    """

    low_energy: float
    """Low threshold for nearest-neighbor energy."""

    high_energy: float
    """High threshold for nearest-neighbor energy."""

    temperature: float = field(default=37.0)
    """Temperature in Celsius at which to calculate nearest-neighbor energy."""

    def __post_init__(self) -> None:
        self.name = 'nearest_neighbor_energy'
        if self.low_energy > self.high_energy:
            raise ValueError(f'low_energy = {self.low_energy} must be less than '
                             f'high_energy = {self.high_energy}')

    def remove_violating_sequences(self, seqs: dn.DNASeqList) -> dn.DNASeqList:
        """Remove sequences with nearest-neighbor energies outside of an interval."""
        wcenergies = dn.calculate_wc_energies(seqs.seqarr, self.temperature)
        within_range = (self.low_energy <= wcenergies) & (wcenergies <= self.high_energy)  # type: ignore
        seqarr_pass = seqs.seqarr[within_range]  # type: ignore
        return dn.DNASeqList(seqarr=seqarr_pass)


@dataclass
class BaseCountConstraint(NumpyConstraint):
    """
    Restricts the sequence to contain a certain number of occurences of a given base.
    """

    base: str
    """Base to count."""

    high_count: Optional[int] = None
    """
    Count of :py:data:`BaseCountConstraint.base` must be at most :py:data:`BaseCountConstraint.high_count`.
    """

    low_count: Optional[int] = None
    """
    Count of :py:data:`BaseCountConstraint.base` must be at least :py:data:`BaseCountConstraint.low_count`.
    """

    def __post_init__(self) -> None:
        self.name = 'base_count'
        if self.low_count is None and self.high_count is None:
            raise ValueError('at least one of low_count or high_count must be specified')

    def remove_violating_sequences(self, seqs: dn.DNASeqList) -> dn.DNASeqList:
        """Remove sequences whose counts of a certain base are outside of an interval."""
        low_count = self.low_count if self.low_count is not None else 0
        high_count = self.high_count if self.high_count is not None else seqs.seqlen
        sumarr = np.sum(seqs.seqarr == dn.base2bits[self.base], axis=1)
        good = (low_count <= sumarr) & (sumarr <= high_count)
        seqarr_pass = seqs.seqarr[good]
        return dn.DNASeqList(seqarr=seqarr_pass)


@dataclass
class BaseEndConstraint(NumpyConstraint):
    """
    Restricts the sequence to contain only certain bases on
    (or near, if :py:data:`BaseEndConstraint.distance` > 0) each end.
    """

    bases: Collection[str]
    """Bases to require on ends."""

    distance_from_end: int = 0
    """Distance from end."""

    five_prime: bool = True
    """ Whether to apply to 5' end of sequence (left end of DNA sequence, lowest index)."""

    three_prime: bool = True
    """ Whether to apply to 3' end of sequence (right end of DNA sequence, highest index)."""

    def __post_init__(self) -> None:
        self.name = 'base_end'
        if not self.five_prime and not self.three_prime:
            raise ValueError('at least one of five_prime or three_prime must be True')
        if not (set(self.bases) < {'A', 'C', 'G', 'T'}):
            raise ValueError('bases must be a strict subset of {A,C,G,T} but is '
                             f'{self.bases}')
        if len(self.bases) == 0:
            raise ValueError('bases cannot be empty')

    def remove_violating_sequences(self, seqs: dn.DNASeqList) -> dn.DNASeqList:
        """Keeps sequences with the given bases at given distance from the 5' or 3' end."""
        all_bits = [dn.base2bits[base] for base in self.bases]

        if seqs.seqlen <= self.distance_from_end:
            raise ValueError(f'cannot specify distance from end of {self.distance_from_end} '
                             f'when sequences only have length {seqs.seqlen}')

        if self.five_prime:
            good_left = np.zeros(shape=len(seqs), dtype=np.bool)
            left = seqs.seqarr[:, self.distance_from_end]
            for bits in all_bits:
                if good_left is None:
                    good_left = (left == bits)
                else:
                    good_left |= (left == bits)

        if self.three_prime:
            good_right = np.zeros(shape=len(seqs), dtype=np.bool)
            right = seqs.seqarr[:, -1 - self.distance_from_end]
            for bits in all_bits:
                if good_right is None:
                    good_right = (right == bits)
                else:
                    good_right |= (right == bits)

        if self.five_prime and self.three_prime:
            seqarr_pass = seqs.seqarr[good_left & good_right]  # noqa
        elif self.five_prime:
            seqarr_pass = seqs.seqarr[good_left]  # noqa
        elif self.three_prime:
            seqarr_pass = seqs.seqarr[good_right]  # noqa
        else:
            raise AssertionError('unreachable')

        return dn.DNASeqList(seqarr=seqarr_pass)


@dataclass
class BaseAtPositionConstraint(NumpyConstraint):
    """
    Restricts the sequence to contain only certain base(s) on at a particular position.

    One use case is that many internal modifications (e.g., biotin or fluorophore)
    can only be placed on an T.
    """

    bases: Union[str, Collection[str]]
    """
    Base(s) to require at position :py:data:`BasePositionConstraint.position`.

    Can either be a single base, or a collection (e.g., list, tuple, set).
    If several bases are specified, the base at :py:data:`BasePositionConstraint.position`
    must be one of the bases in :py:data:`BasePositionConstraint.bases`.
    """

    position: int
    """Position of base to check."""

    def __post_init__(self) -> None:
        self.name = 'base_at_position'
        self.bases = [self.bases] if isinstance(self.bases, str) else list(self.bases)
        if not (set(self.bases) < all_dna_bases):
            raise ValueError(f'bases must be a strict subset of {all_dna_bases} but is '
                             f'{self.bases}')
        if len(self.bases) == 0:
            raise ValueError('bases cannot be empty')

    def remove_violating_sequences(self, seqs: dn.DNASeqList) -> dn.DNASeqList:
        """Remove sequences that don't have one of the given bases at the given position."""
        assert isinstance(self.bases, list)
        if not 0 <= self.position < seqs.seqlen:
            raise ValueError(f'position must be between 0 and {seqs.seqlen} but it is {self.position}')
        mid = seqs.seqarr[:, self.position]
        good = np.zeros(shape=len(seqs), dtype=np.bool)
        for base in self.bases:
            good |= (mid == dn.base2bits[base])
        seqarr_pass = seqs.seqarr[good]
        return dn.DNASeqList(seqarr=seqarr_pass)


@dataclass
class ForbiddenSubstringConstraint(NumpyConstraint):
    """
    Restricts the sequence not to contain a certain substring(s), e.g., GGGG.
    """

    substrings: Union[str, Collection[str]]
    """
    Substring(s) to forbid.

    Can either be a single substring, or a collection (e.g., list, tuple, set).
    If a collection, all substrings must have the same length.
    """

    indices: Optional[Sequence[int]] = None
    """
    Indices at which to check for each substring in :data:`ForbiddenSubstringConstraint.substrings`.
    If not specified, all appropriate indices are checked.
    """

    def __post_init__(self) -> None:
        self.name = 'forbidden_substrings'

        self.substrings = [self.substrings] if isinstance(self.substrings, str) else list(self.substrings)

        lengths = {len(substring) for substring in self.substrings}
        if len(lengths) > 1:
            raise ValueError(f'all substrings must have same length, but they have these lengths: '
                             f'{lengths}\n'
                             f'substrings: {self.substrings}')

        for substring in self.substrings:
            if not (set(substring) < all_dna_bases):
                raise ValueError('must contain only letters from {A,C,G,T} but it is '
                                 f'{substring}, which has extra letters '
                                 f'{set(substring) - all_dna_bases}')
            if len(substring) == 0:
                raise ValueError('substring cannot be empty')

    def length(self) -> int:
        """
        :return:
            length of substring(s) to check
        """
        if isinstance(self.substrings, str):
            return len(self.substrings)
        else:
            # should be a collection
            first_substring = list(self.substrings)[0]
            assert len(first_substring) != 0
            return len(first_substring)

    def remove_violating_sequences(self, seqs: dn.DNASeqList) -> dn.DNASeqList:
        """Remove sequences that have a string in :py:data:`ForbiddenSubstringConstraint.substrings`
        as a substring."""
        assert isinstance(self.substrings, list)
        sub_len = len(self.substrings[0])
        sub_ints = [[dn.base2bits[base] for base in sub] for sub in self.substrings]
        pow_arr = [4 ** k for k in range(sub_len)]
        sub_vals = np.dot(sub_ints, pow_arr)
        toeplitz = dn.create_toeplitz(seqs.seqlen, sub_len, self.indices)
        convolution = np.dot(toeplitz, seqs.seqarr.transpose())
        pass_all = np.ones(seqs.numseqs, dtype=np.bool)
        for sub_val in sub_vals:
            pass_sub = np.all(convolution != sub_val, axis=0)
            pass_all = pass_all & pass_sub
        seqarr_pass = seqs.seqarr[pass_all]
        return dn.DNASeqList(seqarr=seqarr_pass)


@dataclass
class RunsOfBasesConstraint(NumpyConstraint):
    """
    Restricts the sequence not to contain runs of a certain length from a certain subset of bases,
    (e.g., forbidding any substring in {C,G}^3;
    no four bases can appear in a row that are either C or G)

    This works by simply generating all strings representing the runs of bases,
    and then using a :any:`ForbiddenSubstringConstraint` with those strings. So this will not be efficient
    for forbidding, for example {A,C,T}^20 (i.e., all runs of A's, C's, or T's of length 20),
    which would generate all 3^20 = 3,486,784,401 strings of length 20 from the alphabet {A,C,T}^20.
    Hopefully such a constraint would not be used in practice.
    """

    bases: Collection[str]
    """
    Bases to forbid in runs of length :py:data:`RunsOfBasesConstraint.length`.
    """

    length: int
    """Length of run to forbid."""

    def __init__(self, bases: Union[str, Collection[str]], length: int):
        """
        :param bases: Can either be a single base, or a collection (e.g., list, tuple, set).
        :param length: length of run to forbid
        """
        self.name = 'runs_of_bases'
        self.bases = [bases] if isinstance(bases, str) else list(bases)
        self.length = length
        if not (set(self.bases) < all_dna_bases):
            raise ValueError('bases must be a strict subset of {A,C,G,T} but is '
                             f'{self.bases}')
        if len(self.bases) == 0:
            raise ValueError('bases cannot be empty')
        if self.length <= 0:
            raise ValueError(f'length must be positive, but it is {self.length}')
        if self.length == 1:
            allowed_bases = all_dna_bases - set(self.bases)
            logger.warning('You have specified a RunsOfBasesConstraint with length = 1. '
                           'Although this will work, it essentially says to forbid using any of the bases '
                           f'in {set(self.bases)}, i.e., only use bases in {allowed_bases}. '
                           f'It is more efficient to use the constraint '
                           f'RestrictBasesConstraint({allowed_bases}).')

    def remove_violating_sequences(self, seqs: dn.DNASeqList) -> dn.DNASeqList:
        """Remove sequences that have a run of given length of bases from given bases."""
        substrings = list(
            map(lambda lst: ''.join(lst), itertools.product(self.bases, repeat=self.length)))
        constraint = ForbiddenSubstringConstraint(substrings)
        return constraint.remove_violating_sequences(seqs)


log_numpy_generation = True


# log_numpy_generation = False

@dataclass
class SubstringSampler(JSONSerializable):
    """
    A :any:`SubstringSampler` is an object for specifying a common case for the field
    :data:`DomainPool.possible_sequences`, namely where we want the set of possible sequences to be
    all (or many) substrings of a single longer sequence.

    For example, this can be used to choose a rotation of the M13mp18 strand in sequence design.
    If for example 300 consecutive bases of M13 will be used in the design, and we want to choose
    the rotation, but disallow the substring of length 300 to overlap the hairpin at indices
    5514-5556, then one would do the following

    .. code-block:: python

        possible_sequences = SubstringSampler(
            supersequence=m13(), substring_length=300,
            except_overlapping_indices=range(5514, 5557), circular=True)
        pool = DomainPool('M13 rotations', possible_sequences=possible_sequences)
    """

    supersequence: str
    """The longer sequence from which to sample substrings."""

    substring_length: int
    """Length of substrings to sample."""

    except_start_indices: Tuple[int]
    """*Start* indices in :data:`SubstringSampler.supersequence` to avoid. In the constructor this can 
    be specified directly. Another option (mutually exclusive with the parameter `except_start_indices`)
    is to specify the parameter `except_overlapping_indices`, which sets 
    :data:`SubstringSampler.except_start_indices` so that substrings will not intersect any indices in 
    `except_overlapping_indices`."""

    circular: bool
    """Whether :data:`SubstringSampler.supersequence` is circular. If so, then we can sample indices near the 
    end and the substrings will start at the end and wrap around to the start."""

    start_indices: Tuple[int]
    """List of start indices from which to sample when calling :meth:`SubstringSampler.sample_substring`.
    Computed in constructor from other arguments."""

    extended_supersequence: str
    """If :data:`SubstringSampler.circular` is True, then this is :data:`SubstringSampler.supersequence` 
    extended by its own prefix of length :data:`SubstringSampler.substring_length - 1`,
    to make sampling easier. Otherwise it is simply identical to :data:`SubstringSampler.supersequence`.
    Computed in constructor from other arguments."""

    def __init__(self, supersequence: str, substring_length: int,
                 except_start_indices: Optional[Iterable[int]] = None,
                 except_overlapping_indices: Optional[Iterable[int]] = None,
                 circular: bool = False,
                 ) -> None:
        if except_start_indices is not None and except_overlapping_indices is not None:
            raise ValueError('at most one of the parameters except_start_indices or '
                             'except_overlapping_indices can be specified, but you specified both of them')
        self.supersequence = supersequence
        self.substring_length = substring_length
        self.circular = circular

        if except_start_indices is not None:
            self.except_start_indices = tuple(sorted(except_start_indices))
        elif except_overlapping_indices is None:
            self.except_start_indices = cast((), Tuple[int])
        else:
            # compute except_start_indices based on except_overlapping_indices
            assert except_start_indices is None
            assert except_overlapping_indices is not None
            set_except_start_indices: Set[int] = set()  # type: ignore
            # iterate over all idx's in except_overlapping_indices and add all indices between
            # it and the index `self.substring_length + 1` less than it
            for skip_idx in except_overlapping_indices:
                min_start_idx_overlapping_skip_idx = max(0, skip_idx - self.substring_length + 1)
                indices_to_avoid = range(min_start_idx_overlapping_skip_idx, skip_idx + 1)
                set_except_start_indices.update(indices_to_avoid)
            except_start_indices = sorted(list(set_except_start_indices))
            self.except_start_indices = tuple(except_start_indices)

        # compute set of indices to sample from
        self.extended_supersequence = self.supersequence
        if self.circular:
            indices = set(range(len(self.supersequence)))

            # append start of sequence to its end to help with circular condition
            self.extended_supersequence += self.supersequence[:self.substring_length - 1]

            # add indices beyond supersequence length that correspond to indices near the start
            extended_except_indices = list(self.except_start_indices)
            for skip_idx in self.except_start_indices:
                if skip_idx >= self.substring_length - 1:
                    break
                extended_except_indices.append(skip_idx + len(self.supersequence))

            indices -= set(extended_except_indices)
        else:
            indices = set(range(len(self.supersequence) - self.substring_length + 1))
            indices -= set(self.except_start_indices)

        # need to sort so iteration order does not affect RNG
        indices_list: List[int] = list(indices)
        indices_list.sort()
        self.start_indices = tuple(indices_list)

    def sample_substring(self, rng: np.random.Generator) -> str:
        """
        :return: a random substring of :data:`SubstringSampler.supersequence`
                 of length :data:`SubstringSampler.substring_length`.
        """
        start_idx = rng.choice(self.start_indices)
        end_idx = start_idx + self.substring_length
        supersequence = self.extended_supersequence if self.circular else self.supersequence
        assert end_idx <= len(supersequence)
        substring = supersequence[start_idx:end_idx]
        return substring

    @staticmethod
    def from_json_serializable(json_map: Dict[str, Any]) -> SubstringSampler:
        sequence = json_map[name_key]
        substring_length = json_map[length_key]
        except_indices = json_map[replace_with_close_sequences_key]
        circular = json_map[circular_key]
        return SubstringSampler(supersequence=sequence, substring_length=substring_length,
                                except_start_indices=except_indices, circular=circular)

    def to_json_serializable(self, suppress_indent: bool = True) -> Dict[str, Any]:  # noqa
        except_indices = NoIndent(self.except_start_indices) if suppress_indent else self.except_start_indices
        dct = {
            sequence_key: self.supersequence,
            substring_length_key: self.substring_length,
            except_indices_key: except_indices,
            circular_key: self.circular,
        }
        return dct

    def to_json(self) -> str:
        json_map = self.to_json_serializable(suppress_indent=False)
        json_str = json.dumps(json_map, indent=2)
        return json_str


@dataclass
class DomainPool(JSONSerializable):
    """
    Represents a group of related :any:`Domain`'s that share common properties in their sequence design,
    such as length of DNA sequence, or bounds on nearest-neighbor duplex energy.

    Also serves as a "source" of DNA sequences for :any:`Domain`'s in this :any:`DomainPool`.
    By calling :py:meth:`DomainPool.generate_sequence` repeatedly, we can produce DNA sequences satisfying
    the constraints defining this :any:`DomainPool`.
    """

    name: str
    """Name of this :any:`DomainPool`. Must be unique."""

    length: Optional[int] = None
    """Length of DNA sequences generated by this :any:`DomainPool`. 
    
    Should be None if :data:`DomainPool.possible_sequences` is specified."""

    possible_sequences: Union[None, List[str], SubstringSampler] = None
    """
    If specified, all other fields except :data:`DomainPool.name` and :data:`DomainPool.length` 
    are ignored.
    This is an explicit list of sequences to consider for :any:`Domain`'s using this :any:`DomainPool`.
    During the search, if a domain with this :any:`DomainPool` is picked to have its sequence changed,
    then a sequence will be picked uniformly at random from this list. Note that no 
    :any:`NumpyConstraint`'s or :any:`SequenceConstraint`'s will be applied.
    
    Alternatively, the field can be an instance of :any:`SubstringSampler` for the common case that the 
    set of possible sequences is simple all (or many) substrings of a single longer sequence.
    For example, this can be used to choose a rotation of the M13mp18 strand in sequence design.
    
    Should be None if :data:`DomainPool.length` is specified.
    """

    replace_with_close_sequences: bool = True
    """
    If True, instead of picking a sequence uniformly at random from all those satisfying the constraints
    when returning a sequence from :meth:`DomainPool.generate_sequence`,
    one is picked "close" in Hamming distance to the previous sequence of the :any:`Domain`.
    The field :data:`DomainPool.hamming_probability` is used to pick a distance at random, after which
    a sequence that distance from the previous sequence is selected to return.
    """

    hamming_probability: Dict[int, float] = field(default_factory=dict)
    """
    Dictionary that specifies probability of taking a new sequence from the pool that is some integer 
    number of bases different from the previous sequence (Hamming distance). 
    """

    numpy_constraints: List[NumpyConstraint] = field(
        compare=False, hash=False, default_factory=list, repr=False)
    """
    :any:`NumpyConstraint`'s shared by all :any:`Domain`'s in this :any:`DomainPool`.
    This is used to choose potential sequences to assign to the :any:`Domain`'s in this :any:`DomainPool`
    in the method :py:meth:`DomainPool.generate_sequence`.

    The difference with :py:data:`DomainPool.sequence_constraints` is that these constraints can be applied
    efficiently to many sequences at once, represented as a numpy 2D array of bytes (via the class
    :any:`np.DNASeqList`), so they are done in large batches in advance.
    In contrast, the constraints in :py:data:`DomainPool.sequence_constraints` are done on Python strings
    representing DNA sequences, and they are called one at a time when a new sequence is requested in
    :py:meth:`DomainPool.generate_sequence`.

    Optional; default is empty.
    """

    sequence_constraints: List[SequenceConstraint] = field(
        compare=False, hash=False, default_factory=list, repr=False)
    """
    :any:`SequenceConstraint`'s shared by all :any:`Domain`'s in this :any:`DomainPool`.
    This is used to choose potential sequences to assign to the :any:`Domain`'s in this :any:`DomainPool`
    in the method :py:meth:`DomainPool.generate`.

    See :py:data:`DomainPool.numpy_constraints` for an explanation of the difference between them.

    See :py:data:`DomainPool.domain_constraints` for an explanation of the difference between them.

    Optional; default is empty.
    """

    def __post_init__(self) -> None:
        if ((self.length is None and self.possible_sequences is None) or
                (self.length is not None and self.possible_sequences is not None)):
            raise ValueError('exactly one of length or possible_sequences should be specified')

        if self.possible_sequences is not None:
            if isinstance(self.possible_sequences, list):
                if len(self.possible_sequences) == 0:
                    raise ValueError('possible_sequences cannot be empty')
                first_seq = self.possible_sequences[0]
                length = len(first_seq)
                for idx, seq in enumerate(self.possible_sequences):
                    if len(seq) != length:
                        raise ValueError(f'ERROR: Two sequences in possible_sequence of DomainPool '
                                         f'"{self.name}" have different lengths:\n'
                                         f'first sequence {first_seq} has length {length}\n'
                                         f'and sequence "{seq}", index {idx} in the list possible_sequences,\n'
                                         f'has length {len(seq)}.')

            if len(self.numpy_constraints) > 0:
                raise ValueError('If possible_sequences is specified, then numpy_constraints should '
                                 'not be specified.')
            if len(self.sequence_constraints) > 0:
                raise ValueError('If possible_sequences is specified, then sequence_constraints should '
                                 'not be specified.')

        if self.length is not None:
            if len(self.hamming_probability) == 0:  # sets default probability distribution if the user does not
                # exponentially decreasing probability of making i+1 (since i starts at 0) base changes
                # for i in range(self.length):
                #     self.hamming_probability[i + 1] = 1 / 2 ** (i + 1)
                # self.hamming_probability[self.length] *= 2

                # linearly decreasing probability of making i+1 (since i starts at 0) base changes
                total = 0.0
                for i in range(self.length):
                    prob = 1 / (i + 1)
                    self.hamming_probability[i + 1] = prob
                    total += prob
                # normalize to be a probability measure
                for length in self.hamming_probability:
                    self.hamming_probability[length] /= total

            idx = 0
            for numpy_constraint in self.numpy_constraints:
                if not isinstance(numpy_constraint, NumpyConstraint):
                    raise ValueError('each element of numpy_constraints must be an instance of '
                                     'NumpyConstraint, '
                                     f'but the element at index {idx} is of type {type(numpy_constraint)}')
                elif isinstance(numpy_constraint, RunsOfBasesConstraint):
                    if numpy_constraint.length > self.length:
                        raise ValueError(f'DomainPool "{self.name}" has length {self.length}, but a '
                                         f'RunsOfBasesConstraint was specified with larger length '
                                         f'{numpy_constraint.length}, which is not allowed')
                elif isinstance(numpy_constraint, ForbiddenSubstringConstraint):
                    if numpy_constraint.length() > self.length:
                        raise ValueError(f'DomainPool "{self.name}" has length {self.length}, but a '
                                         f'ForbiddenSubstringConstraint was specified with larger length '
                                         f'{numpy_constraint.length()}, which is not allowed')
                idx += 1

            idx = 0
            for seq_constraint in self.sequence_constraints:
                # SequenceConstraint is an alias for Callable[[str], float],
                # which is not checkable using isinstance
                # https://stackoverflow.com/questions/624926/how-do-i-detect-whether-a-python-variable-is-a-function
                if not callable(seq_constraint):
                    raise ValueError('each element of numpy_constraints must be an instance of '
                                     'SequenceConstraint (i.e., be a function that takes a single string '
                                     'and returns a bool), '
                                     f'but the element at index {idx} is of type {type(seq_constraint)}')
                idx += 1

    def __hash__(self) -> int:
        return hash((self.name, self.length))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DomainPool):
            return False
        return self.name == other.name and self.length == other.length

    def to_json(self) -> str:
        json_map = self.to_json_serializable(suppress_indent=False)
        json_str = json.dumps(json_map, indent=2)
        return json_str

    def to_json_serializable(self, suppress_indent: bool = True) -> Dict[str, Any]:
        if self.length is None and self.possible_sequences is None:
            raise ValueError('exactly one of length or possible_sequences should be None, but both are')
        if self.length is not None and self.possible_sequences is not None:
            raise ValueError('exactly one of length or possible_sequences should be None, but neither is')

        dct = {
            name_key: self.name,
            replace_with_close_sequences_key: self.replace_with_close_sequences,
            hamming_probability_key: self.hamming_probability,
        }

        if self.possible_sequences is not None:
            if isinstance(self.possible_sequences, list):
                dct[possible_sequences_key] = self.possible_sequences
            elif isinstance(self.possible_sequences, SubstringSampler):
                dct[possible_sequences_key] = self.possible_sequences.to_json_serializable(suppress_indent)
            else:
                raise ValueError('possible_sequences should be list of strings or SuperSequence but is '
                                 f'{type(self.possible_sequences)}: {self.possible_sequences}')
        if self.length is not None:
            dct[length_key] = self.length

        return dct

    @staticmethod
    def from_json_serializable(json_map: Dict[str, Any]) -> DomainPool:
        name = json_map[name_key]
        replace_with_close_sequences = json_map[replace_with_close_sequences_key]
        hamming_probability_str_keys = json_map[hamming_probability_key]
        hamming_probability = {int(key): val for key, val in hamming_probability_str_keys.items()}

        length = json_map.get(length_key)
        possible_sequences = json_map.get(possible_sequences_key)

        if length is None and possible_sequences is None:
            raise ValueError('exactly one of length or possible_sequences should be None, but both are')
        if length is not None and possible_sequences is not None:
            raise ValueError('exactly one of length or possible_sequences should be None, but neither is')

        return DomainPool(name=name, length=length,
                          replace_with_close_sequences=replace_with_close_sequences,
                          hamming_probability=hamming_probability,
                          possible_sequences=possible_sequences,
                          )

    def _first_sequence_satisfying_sequence_constraints(self, seqs: dn.DNASeqList) -> Optional[str]:
        if len(seqs) == 0:
            return None
        if len(self.sequence_constraints) == 0:
            return seqs.get_seq_str(0)
        for idx in range(seqs.numseqs):
            seq = seqs.get_seq_str(idx)
            if self.satisfies_sequence_constraints(seq):
                return seq
        return None

    def satisfies_sequence_constraints(self, sequence: str) -> bool:
        """
        :param sequence:
            DNA sequence to check
        :return:
            whether `sequence` satisfies all constraints in :data:`DomainPool.sequence_constraints`
        """
        return all(constraint(sequence) for constraint in self.sequence_constraints)

    def generate_sequence(self, rng: np.random.Generator, previous_sequence: Optional[str] = None) -> str:
        """
        Returns a DNA sequence of given length satisfying :py:data:`DomainPool.numpy_constraints` and
        :py:data:`DomainPool.sequence_constraints`

        **Note:** By default, there is no check that the sequence returned is unequal to one already
        assigned somewhere in the design, since both :py:data:`DomainPool.numpy_constraints` and
        :data:`DomainPool.sequence_constraints` do not have access to the whole :any:`Design`.
        But the :any:`DomainPairConstraint` returned by
        :meth:`domains_not_substrings_of_each_other_constraint`
        can be used to specify this :any:`Design`-wide constraint.

        Note that if :data:`DomainPool.possible_sequences` is specified, then all constraints are ignored,
        and instead a sequence is chosen randomly to be returned from that list.

        :param rng:
            numpy random number generator to use. To use a default, pass :py:data:`np.default_rng`.
        :param previous_sequence:
            previously generated sequence to be replaced by a new sequence; None if no previous
            sequence exists. Used to choose a new sequence "close" to itself in Hamming distance,
            if the field :data:`DomainPool.replace_with_close_sequences` is True and `previous_sequence`
            is not None.
            The number of differences between `previous_sequence` and its neighbors is determined by randomly
            picking a Hamming distance from :data:`DomainPool.hamming_probability` with
            weighted probabilities of choosing each distance.
        :return:
            DNA sequence of given length satisfying :py:data:`DomainPool.numpy_constraints` and
            :py:data:`DomainPool.sequence_constraints`
        """
        if self.possible_sequences is not None:
            if isinstance(self.possible_sequences, list):
                sequence = rng.choice(self.possible_sequences)
            elif isinstance(self.possible_sequences, SubstringSampler):
                sequence = self.possible_sequences.sample_substring(rng)
            else:
                raise ValueError('possible_sequences should be list of strings or SuperSequence but is '
                                 f'{type(self.possible_sequences)}: {self.possible_sequences}')
        elif not self.replace_with_close_sequences or previous_sequence is None:
            sequence = self._get_next_sequence_satisfying_numpy_and_sequence_constraints(rng)
        else:
            sequence = self._sample_hamming_distance_from_sequence(previous_sequence, rng)

        return sequence

    def _sample_hamming_distance_from_sequence(self, previous_sequence: str, rng: np.random.Generator) -> str:
        # all possible distances from 1 to len(previous_sequence) are calculated.

        hamming_probabilities = np.array(list(self.hamming_probability.values()))

        # pick a distance at random, then re-pick if no sequences are at that distance
        available_distances_list = list(range(1, len(previous_sequence) + 1))

        while True:  # each iteration of this loop tries one sampled distance
            num_to_generate = 100

            if len(available_distances_list) == 0:
                raise ValueError('out of Hamming distances to try, quitting')

            # sample a Hamming distance that we haven't tried yet
            available_distances_arr = np.array(available_distances_list)
            existing_hamming_probabilities = hamming_probabilities[available_distances_arr - 1]
            prob_sum = existing_hamming_probabilities.sum()
            assert prob_sum > 0.0
            existing_hamming_probabilities /= prob_sum
            sampled_distance: int = rng.choice(available_distances_arr, p=existing_hamming_probabilities)

            sequence: Optional[str] = None

            while sequence is None:  # each iteration of this loop tries one value of num_to_generate
                bases = self._bases_to_use()
                length = self.length

                num_ways_to_choose_subsequence_indices = dn.comb(length, sampled_distance)
                num_different_bases = len(bases) - 1
                num_subsequences = num_different_bases ** sampled_distance
                num_sequences_at_sampled_distance = num_ways_to_choose_subsequence_indices * num_subsequences

                if num_to_generate > num_sequences_at_sampled_distance:
                    num_to_generate = num_sequences_at_sampled_distance

                if num_to_generate >= num_sequences_at_sampled_distance // 2:
                    num_to_generate = num_sequences_at_sampled_distance
                    # if we want sufficiently many random sequences, just generate all possible sequences
                    seqs = dn.DNASeqList(
                        hamming_distance_from_sequence=(sampled_distance, previous_sequence), alphabet=bases,
                        shuffle=True, rng=rng)
                    generated_all_seqs = True
                else:
                    # otherwise sample num_to_generate with replacement
                    seqs = dn.DNASeqList(
                        hamming_distance_from_sequence=(sampled_distance, previous_sequence), alphabet=bases,
                        shuffle=True, num_random_seqs=num_to_generate, rng=rng)
                    generated_all_seqs = False

                seqs_satisfying_numpy_constraints = self._filter_numpy_constraints(seqs)
                self._log_numpy_generation(length, num_to_generate, len(seqs_satisfying_numpy_constraints))
                sequence = self._first_sequence_satisfying_sequence_constraints(
                    seqs_satisfying_numpy_constraints)
                if sequence is not None:
                    return sequence

                max_to_generate_before_moving_on = 10 ** 6

                if generated_all_seqs:
                    logger.info(f"""\
We've generated all possible DNA sequences at Hamming distance {sampled_distance} 
from the previous sequence {previous_sequence} and not found one that passed your 
NumpyConstraints and SequenceConstraints. Trying another distance.""")
                    available_distances_list.remove(sampled_distance)
                elif num_to_generate >= max_to_generate_before_moving_on:
                    logger.info(f"""\
We've generated over {max_to_generate_before_moving_on} DNA sequences at Hamming distance {sampled_distance} 
from the previous sequence {previous_sequence} and not found one that passed your 
NumpyConstraints and SequenceConstraints. Trying another distance.""")
                    available_distances_list.remove(sampled_distance)

                if sequence is None and (
                        generated_all_seqs or num_to_generate >= max_to_generate_before_moving_on):
                    # found no sequences passing constraints at distance `sampled_distance`
                    # (either through exhaustive search, or trying at least 1 billion),
                    # need to try a new Hamming distance
                    break

                num_to_generate *= 2

        # mypy actually flags the next line as unreachable
        # raise AssertionError('should be unreachable')

    def _get_next_sequence_satisfying_numpy_and_sequence_constraints(self, rng: np.random.Generator) -> str:
        num_to_generate = 100
        num_sequences_total = len(self._bases_to_use()) ** self.length

        sequence = None
        while sequence is None:
            if num_to_generate >= num_sequences_total / 2:
                num_to_generate = num_sequences_total

            seqs_satisfying_numpy_constraints = \
                self._generate_random_sequences_satisfying_numpy_constraints(rng, num_to_generate)
            sequence = self._first_sequence_satisfying_sequence_constraints(seqs_satisfying_numpy_constraints)
            if sequence is not None:
                return sequence

            if num_to_generate > 10 ** 9:
                raise NotImplementedError("We've generated over 1 billion random DNA sequences of length "
                                          f"{self.length} and found none that passed the NumpyConstraints "
                                          f"and " "SequenceConstraints. Try relaxing the constraints so "
                                          "that it's more likely a random sequence satisfies the "
                                          "constraints.")
            if num_to_generate == num_sequences_total:
                raise NotImplementedError(f"We generated all possible {num_sequences_total} DNA sequences "
                                          f"of length {self.length} and found none that passed the "
                                          f"NumpyConstraints and "
                                          "SequenceConstraints. Try relaxing the constraints so that "
                                          "some sequence satisfies the constraints.")

            num_to_generate *= 2

        raise AssertionError('should be unreachable')

    def _generate_random_sequences_satisfying_numpy_constraints(self, rng: np.random.Generator,
                                                                num_to_generate: int) -> dn.DNASeqList:
        bases = self._bases_to_use()
        length = self.length
        _length_threshold_numpy = math.floor(math.log(num_to_generate, 4))
        seqs = dn.DNASeqList(length=length, alphabet=bases, shuffle=True,
                             num_random_seqs=num_to_generate, rng=rng)
        seqs_satisfying_numpy_constraints = self._filter_numpy_constraints(seqs)
        self._log_numpy_generation(length, num_to_generate, len(seqs_satisfying_numpy_constraints))
        return seqs_satisfying_numpy_constraints

    @staticmethod
    def _log_numpy_generation(length: int, num_to_generate: int, num_passed: int):
        if log_numpy_generation:
            num_decimals = len(str(num_to_generate))
            logger.debug(f'generated {num_to_generate:{num_decimals}} sequences '
                         f'of length {length:2}, '
                         f'of which {num_passed:{num_decimals}} '
                         f'passed the numpy sequence constraints')

    def _bases_to_use(self) -> Collection[str]:
        # checks explicitly for RestrictBasesConstraint
        for constraint in self.numpy_constraints:
            if isinstance(constraint, RestrictBasesConstraint):
                return constraint.bases
        return 'A', 'C', 'G', 'T'

    def _filter_numpy_constraints(self, seqs: dn.DNASeqList) -> dn.DNASeqList:
        # filter sequence not passing numpy constraints, but skip NumpyRestrictBasesConstraint since
        # that is more efficiently handled by the DNASeqList constructor to generate the sequences
        # in the first place
        for constraint in self.numpy_constraints:
            if isinstance(constraint, RestrictBasesConstraint):
                continue
            seqs = constraint.remove_violating_sequences(seqs)
        return seqs


def add_quotes(string: str) -> str:
    # adds quotes around a string
    return f'"{string}"'


def mandatory_field(ret_type: Type, json_map: Dict, main_key: str, *legacy_keys: str) -> Any:
    # should be called from function whose return type is the type being constructed from JSON, e.g.,
    # Design or Strand, given by ret_type. This helps give a useful error message
    for key in (main_key,) + legacy_keys:
        if key in json_map:
            return json_map[key]
    ret_type_name = ret_type.__name__
    msg_about_keys = f'the key "{main_key}"'
    if len(legacy_keys) > 0:
        msg_about_keys += f" (or any of the following legacy keys: {', '.join(map(add_quotes, legacy_keys))})"
    msg = f'I was looking for {msg_about_keys} in the JSON encoding of a {ret_type_name}, ' \
          f'but I did not find it.' \
          f'\n\nThis occurred when reading this JSON object:\n{json_map}'
    raise ValueError(msg)


class Part(ABC):

    def __eq__(self, other: Part) -> bool:
        return type(self) == type(other) and self.name == other.name

    # Remember to set subclass __hash__ equal to this implementation; see here:
    # https://docs.python.org/3/reference/datamodel.html#object.__hash__
    def __hash__(self) -> int:
        return hash(self.key())

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def key(self) -> str:
        # used as key in dictionary
        pass

    @staticmethod
    @abstractmethod
    def name_of_part_type(self) -> str:
        pass

    @property
    @abstractmethod
    def fixed(self) -> bool:
        pass

    @abstractmethod
    def individual_parts(self) -> Union[Tuple[Domain, ...], Tuple[Strand, ...]]:
        # if Part represents a tuple, e.g., StrandPair or DomainPair, then returns tuple of
        # individual domains/strands
        pass


DomainLabel = TypeVar('DomainLabel')


@dataclass
class DomainPair(Part, Generic[DomainLabel]):
    domain1: Domain
    domain2: Domain

    def __post_init__(self) -> None:
        # make this symmetric so make dict lookups work
        if self.domain1.name > self.domain2.name:
            self.domain1, self.domain2 = self.domain2, self.domain1

    # needed to avoid unhashable type error; see
    # https://docs.python.org/3/reference/datamodel.html#object.__hash__
    __hash__ = Part.__hash__

    @property
    def name(self) -> str:
        return f'{self.domain1.name}, {self.domain2.name}'

    def key(self) -> str:
        return f'DomainPair[{self.domain1.name}, {self.domain2.name}]'

    @staticmethod
    def name_of_part_type(self) -> str:
        return 'domain pair'

    def individual_parts(self) -> Tuple[Domain, ...]:
        return self.domain1, self.domain2

    @property
    def fixed(self) -> bool:
        return self.domain1.fixed and self.domain2.fixed


@dataclass
class Domain(Part, JSONSerializable, Generic[DomainLabel]):
    """
    Represents a contiguous substring of the DNA sequence of a :any:`Strand`, which is intended
    to be either single-stranded, or to bind fully to the Watson-Crick complement of the :any:`Domain`.

    If two domains are complementary, they are represented by the same :any:`Domain` object.
    They are distinguished only by whether the :any:`Strand` object containing them has the
    :any:`Domain` in its set :py:data:`Strand.starred_domains` or not.

    A :any:`Domain` uses only its name to compute hash and equality checks, not its sequence.
    This allows a :any:`Domain` to be used in sets and dicts while modifying the sequence assigned to it,
    and also modifying the pool (letting the pool be assigned after it is created).
    """

    _name: str
    """
    Name of the :any:`Domain`.
    This is the "unstarred" version of the name, and it cannot end in `*`.
    """

    _starred_name: str

    _pool: Optional[DomainPool] = field(init=False, default=None, compare=False, hash=False)
    """
    Each :any:`Domain` in the same :any:`DomainPool` as this one share a set of properties, such as
    length and individual :any:`DomainConstraint`'s.
    """

    # TODO: `set_sequence_recursive_up`

    #        - if parent is not none, make recursive call to set_sequence_recursive_up
    # TODO: `set_sequence_recursive_down`
    #        - iterate over children, call set_sequence
    _sequence: Optional[str] = field(init=False, repr=False, default=None, compare=False, hash=False)
    """
    DNA sequence assigned to this :any:`Domain`. This is assumed to be the sequence of the unstarred
    variant; the starred variant has the Watson-Crick complement,
    accessible via :py:data:`Domain.starred_sequence`.
    """

    weight: float = 1.0
    """
    Weight to apply before picking domain at random to change when re-assigning DNA sequences during search.
    Should only be changed for independent domains. (those with :data:`Domain.dependent` set to False)
    
    Normally a domain's probability of being changed is proportional to the total score of violations it
    causes, but that total score is first multiplied by :data:`Domain.weight`. This is useful,
    for instance, to weight a domain lower when it has many subdomains that intersect many strands,
    for example if a domain represents an M13 strand. It may be more efficient to pick such a domain
    less often since changing it will change many strands in the design and, when the design gets
    close to optimized, this will likely cause the score to go up.
    """

    fixed: bool = False
    """
    Whether this :any:`Domain`'s DNA sequence is fixed, i.e., cannot be changed by the
    search algorithm :py:meth:`search.search_for_dna_sequences`.

    Note: If a domain is fixed then all of its subdomains must also be fixed.
    """

    label: Optional[DomainLabel] = None
    """
    Optional generic "label" object to associate to this :any:`Domain`.

    Useful for associating extra information with the :any:`Domain` that will be serialized, for example,
    for DNA sequence design. It must be an object (e.g., a dict or primitive type such as str or int)
    that is naturally JSON serializable. (Calling
    `json.dumps <https://docs.python.org/3/library/json.html#json.dumps>`_
    on the object should succeed without having to specify a custom encoder.)
    """

    dependent: bool = False
    """
    Whether this :any:`Domain`'s DNA sequence is dependent on others. Usually this is not the case.
    However, domains can be subdivided hierarchically into a tree of domains by setting 
    :data:`Domain.subdomains` to describe the tree. In this case exactly
    one domain along every path from the root to any leaf must be independent, and the rest dependent:
    the dependent domains will have their sequences calculated from the indepenedent ones.
    
    A possible use case is that one strand represents a subsequence of M13 of length 300,
    of which there are 7249 possible DNA sequences to assign based on the different
    rotations of M13. If this strand is bound to several other strands, it will have
    several domains, but they cannot be set independently of each other.
    This can be done by creating a strand with a single long domain, which is subdivided into many dependent 
    child domains.
    Only the entire strand, the root domain, can be assigned at once, changing every domain at once,
    so the domains are dependent on the root domain's assigned sequence.
    """

    length: Optional[int] = None
    """
    Length of this domain. If None, then the method :meth:`Domain.length` asks :data:`Domain.pool`
    for the length. However, a :any:`Domain` with :data:`Domain.dependent` set to True has no
    :data:`Domain.pool`. For such domains, it is necessary to set a :data:`Domain.length` field directly.
    """

    _subdomains: List[Domain] = field(init=False, default_factory=list)
    """List of smaller subdomains whose concatenation is this domain. If empty, then there are no subdomains.
    """

    parent: Optional[Domain] = field(init=False, default=None)
    """Domain of which this is a subdomain. Note, this is not set manually, this is set by the library based 
    on the :data:`Domain.subdomains` of other domains in the same tree.
    """

    def __init__(self, name: str, pool: Optional[DomainPool] = None, sequence: Optional[str] = None,
                 fixed: bool = False, label: Optional[DomainLabel] = None, dependent: bool = False,
                 subdomains: Optional[List["Domain"]] = None, weight: Optional[float] = None) -> None:
        if subdomains is None:
            subdomains = []
        self._name = name
        self._starred_name = name + '*'
        self._pool = pool
        self._sequence = sequence
        self._starred_sequence = None if sequence is None else dv.wc(sequence)
        self.fixed = fixed
        self.label = label
        self.dependent = dependent
        self._subdomains = subdomains

        if self.name.endswith('*'):
            raise ValueError('Domain name cannot end with *\n'
                             f'domain name = {self.name}')

        if self.fixed:
            for sd in self._subdomains:
                if not sd.fixed:
                    raise ValueError(f'Domain is fixed, but subdomain {sd} is not fixed')
        else:
            contains_no_non_fixed_subdomains = True
            for sd in self._subdomains:
                if not sd.fixed:
                    contains_no_non_fixed_subdomains = False
                    break
            if len(self._subdomains) > 0 and contains_no_non_fixed_subdomains:
                raise ValueError(f'Domain is not fixed, but all subdomains are fixed')

        # Set parent field for all subdomains.
        for subdomain in self._subdomains:
            subdomain.parent = self

        if self.dependent and weight is not None:
            raise ValueError(f'cannot set Domain.weight when Domain.dependent is True, '
                             f'since dependent domains cannot be picked to change in the search, '
                             f'which is the probability that DOmain.weight affects')
        if weight is not None:
            self.weight = weight

    @staticmethod
    def name_of_part_type(self) -> str:
        return 'domain'

    def key(self) -> str:
        return f'Domain({self.name})'

    # needed to avoid unhashable type error; see
    # https://docs.python.org/3/reference/datamodel.html#object.__hash__
    __hash__ = Part.__hash__

    def __repr__(self) -> str:
        return self._name

    def individual_parts(self) -> Tuple[Domain, ...]:
        return self,

    def to_json_serializable(self, suppress_indent: bool = True) -> Union[NoIndent, Dict[str, Any]]:
        """
        :return:
            Dictionary ``d`` representing this :any:`Domain` that is "naturally" JSON serializable,
            by calling ``json.dumps(d)``.
        """
        dct: Dict[str, Any] = {name_key: self.name}
        if self._pool is not None:
            dct[domain_pool_name_key] = self._pool.name
        if self.has_sequence():
            dct[sequence_key] = self._sequence
            if self.fixed:
                dct[fixed_key] = True
        if self.label is not None:
            dct[label_key] = self.label
        return NoIndent(dct) if suppress_indent else dct

    @staticmethod
    def from_json_serializable(json_map: Dict[str, Any],
                               pool_with_name: Optional[Dict[str, DomainPool]],
                               label_decoder: Callable[[Any], DomainLabel] = lambda label: label) \
            -> Domain[DomainLabel]:
        """
        :param json_map:
            JSON serializable object encoding this :any:`Domain`, as returned by
            :py:meth:`Domain.to_json_serializable`.
        :param pool_with_name:
            dict mapping name to :any:`DomainPool` with that name; required to rehydrate :any:`Domain`'s.
            If None, then a DomainPool with no constraints is created with the name and domain length
            found in the JSON.
        :param label_decoder:
            Function transforming object deserialized from JSON  (e.g, dict, list, string) into an object
            of type DomainLabel.
        :return:
            :any:`Domain` represented by dict `json_map`, assuming it was created by
            :py:meth:`Domain.to_json_serializable`.
        """
        name: str = mandatory_field(Domain, json_map, name_key)
        sequence: Optional[str] = json_map.get(sequence_key)
        fixed: bool = json_map.get(fixed_key, False)

        label_json: Any = json_map.get(label_key)
        label = label_decoder(label_json)

        pool: Optional[DomainPool]
        pool_name: Optional[str] = json_map.get(domain_pool_name_key)
        if pool_name is not None:
            if pool_with_name is not None:
                pool = pool_with_name[pool_name] if pool_with_name is not None else None
            else:
                raise AssertionError()
        else:
            pool = None

        domain: Domain[DomainLabel] = Domain(
            name=name, sequence=sequence, fixed=fixed, pool=pool, label=label)
        return domain

    @property
    def name(self) -> str:
        """
        :return: :any:`DomainPool` of this :any:`Domain`
        """
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        """
        :param new_name: new name to set
        """
        self._name = new_name
        self._starred_name = new_name + '*'

    @property
    def pool(self) -> DomainPool:
        """
        :return: :any:`DomainPool` of this :any:`Domain`
        """
        if self._pool is None:
            raise ValueError(f'pool has not been set for Domain {self.name}')
        return self._pool

    @pool.setter
    def pool(self, new_pool: DomainPool) -> None:
        """
        :param new_pool: new :any:`DomainPool` to set
        :raises ValueError: if :py:data:`Domain.pool_` is not None and is not same object as `new_pool`
        """
        if self._pool is not None and new_pool is not self._pool:
            raise ValueError(f'Assigning pool {new_pool} to domain '
                             f'{self} but {self} already has domain '
                             f'pool {self._pool}')
        self._pool = new_pool

    @property
    def subdomains(self) -> List["Domain"]:
        """
        Subdomains of this :any:`Domain`.

        Used in connection with :data:`Domain.dependent` to declare that
        some :any:`Domain`'s are contained within other domains (forming a tree in general),
        and domains with :data:`Domain.dependent` set to True automatically take their sequences from
        independent domains.

        WARNING: this can be a bit tricky to determine the order when setting these.
        The subdomains should be listed in 5' to 3' order for UNSTARRED domains.
        If there is a starred domain with starred subdomains, they would be listed in
        REVERSE order.

        For example, if there is a domain `dom*`  ``[--------->`` of length 11
        with two subdomains `sub1*` ``[----->`` of length 7 and `sub2*` ``[-->`` of length 4
        (put together they look like ``[----->[-->``)
        that appear
        in that order left to right (5' to 3'), then one would assign the domain `dom` to have subdomains
        ``[sub2, sub1]``, since the UNSTARRED domains appear ``<-----]<--]``, i.e., in 5' to 3' order
        for the unstarred domains, first the length 4 domain `dom2` appears,
        then the length 7 domain `dom1`.
        """
        return self._subdomains

    @subdomains.setter
    def subdomains(self, new_subdomains: List["Domain"]) -> None:
        self._subdomains = new_subdomains
        for s in new_subdomains:
            s.parent = self

    def has_length(self) -> bool:
        """
        :return:
            True if this :any:`Domain` has a length, which means either a sequence has been assigned
            to it, or it has a :any:`DomainPool`.
        """
        return self._sequence is not None or (
                self._pool is not None and self._pool.length is not None) or self.length is not None

    def get_length(self) -> int:
        """
        :return:
            Length of this domain (delegates to pool)
        :raises ValueError:
            if no :any:`DomainPool` has been set for this :any:`Domain`
        """
        if self.length is not None:
            return self.length
        if self.fixed and self._sequence is not None:
            return len(self._sequence)
        if self._pool is None:
            raise ValueError(f'No DomainPool has been set for domain {self.name}, '
                             f'so it has no length yet.\n'
                             'Assign a DomainPool (which has a length field) to give this Domain a length.')
        if self._pool.length is not None:
            return self._pool.length
        elif self._pool.possible_sequences is not None:
            if isinstance(self._pool.possible_sequences, list):
                # if pool.length is None, then possible_sequences must be not None and nonempty,
                # so we consult its first sequence to inquire about the length
                assert len(self._pool.possible_sequences) > 0
                return len(self._pool.possible_sequences[0])
            elif isinstance(self._pool.possible_sequences, SubstringSampler):
                return self._pool.possible_sequences.substring_length
            else:
                raise ValueError('possible_sequences should be list of strings or SuperSequence but is '
                                 f'{type(self._pool.possible_sequences)}: {self._pool.possible_sequences}')

    def sequence(self) -> str:
        """
        :return: DNA sequence of this domain (unstarred version)
        :raises ValueError: If no sequence has been assigned.
        """
        if self._sequence is None or '?' in self._sequence:
            raise ValueError(f'sequence has not been set for Domain {self.name}\n'
                             f'sequence: {self._sequence}')
        return self._sequence

    def set_sequence(self, new_sequence: str) -> None:
        """
        :param new_sequence: new DNA sequence to set
        """
        if self.fixed:
            raise ValueError('cannot assign a new sequence to this Domain; its sequence is fixed as '
                             f'{self._sequence}')
        if self.has_length() and len(new_sequence) != self.get_length():
            raise ValueError(
                f'incorrect length for new_sequence={new_sequence};\n'
                f'it is length {len(new_sequence)}, but this domain is length {self.get_length()}')
        # Check that total length of subdomains (if used) adds up domain length.
        if len(self._subdomains) != 0:
            sd_total_length = 0
            for sd in self._subdomains:
                sd_total_length += sd.get_length()
            if sd_total_length != self.get_length():
                raise ValueError(
                    f'Domain {self} is length {self.get_length()} but subdomains {self._subdomains} '
                    f'have total length of {sd_total_length}')
        self._sequence = new_sequence
        self._starred_sequence = dv.wc(new_sequence)
        self._set_subdomain_sequences(new_sequence)
        self._set_parent_sequence(new_sequence)

    def _set_subdomain_sequences(self, new_sequence: str) -> None:
        """Sets sequence for all subdomains.

        :param new_sequence: Sequence assigned to this domain.
        :type new_sequence: str
        """
        sequence_idx = 0
        for sd in self._subdomains:
            sd_len = sd.get_length()
            sd_sequence = new_sequence[sequence_idx: sequence_idx + sd_len]
            sd._sequence = sd_sequence
            sd._starred_sequence = dv.wc(sd_sequence)
            sd._set_subdomain_sequences(sd_sequence)
            sequence_idx += sd_len

    def _set_parent_sequence(self, new_sequence: str) -> None:
        """Set parent sequence and propagate upwards

        :param new_sequence: new sequence
        :type new_sequence: str
        """
        parent = self.parent
        if parent is not None:
            if parent._sequence is None:
                parent._sequence = '?' * parent.get_length()
                parent._starred_sequence = '?' * parent.get_length()
            # Add up lengths of subdomains, add new_sequence
            idx = 0
            assert self in parent._subdomains
            sd: Optional[Domain] = None
            for sd in parent._subdomains:
                if sd == self:
                    break
                else:
                    idx += sd.get_length()
            assert sd is not None
            old_sequence = parent._sequence
            parent._sequence = old_sequence[:idx] + new_sequence + old_sequence[idx + sd.get_length():]
            parent._starred_sequence = dv.wc(parent._sequence)
            parent._set_parent_sequence(parent._sequence)

    def set_fixed_sequence(self, fixed_sequence: str) -> None:
        """
        Set DNA sequence and fix it so it is not changed by the dsd sequence designer.

        Since it is being fixed, there is no Domain pool, so we don't check the pool or whether it has
        a length. We also bypass the check that it is not fixed.

        :param fixed_sequence: new fixed DNA sequence to set
        """
        self._sequence = fixed_sequence
        self._starred_sequence = dv.wc(fixed_sequence)
        self._set_subdomain_sequences(fixed_sequence)
        self._set_parent_sequence(fixed_sequence)
        self.fixed = True

    @property
    def starred_name(self) -> str:
        """
        :return: The value :py:data:`Domain.name` with `*` appended to it.
        """
        return self._starred_name

    @property
    def starred_sequence(self) -> str:
        """
        :return: Watson-Crick complement of DNA sequence assigned to this :any:`Domain`.
        """
        if self._sequence is None:
            raise ValueError('no DNA sequence has been assigned to this Domain')
        # return dv.wc(self.sequence)
        return self._starred_sequence

    def get_name(self, starred: bool) -> str:
        """
        :param starred: whether to return the starred or unstarred version of the name
        :return: The value :py:data:`Domain.name` or :py:data:`Domain.starred_name`, depending on
                 the value of parameter `starred`.
        """
        return self._starred_name if starred else self._name

    def concrete_sequence(self, starred: bool) -> str:
        """
        :param starred: whether to return the starred or unstarred version of the sequence
        :return: The value :py:data:`Domain.sequence` or :py:data:`Domain.starred_sequence`, depending on
                 the value of parameter `starred`.
        :raises ValueError: if this :any:`Domain` does not have a sequence assigned
        """
        if self._sequence is None:
            raise ValueError('no DNA sequence has been assigned to this Domain')
        if self._starred_sequence is None:
            raise AssertionError('_starred_sequence should be set to non-None if _sequence is not None. '
                                 'Something went wrong in the logic of dsd.')
        return self._starred_sequence if starred else self._sequence

    def has_sequence(self) -> bool:
        """
        :return: Whether a complete DNA sequence has been assigned to this :any:`Domain`.
                 If this domain has subdomains, False if any subdomain has not been assigned
                 a sequence.
        """
        return self._sequence is not None and '?' not in self._sequence

    @staticmethod
    def complementary_domain_name(domain_name: str) -> str:
        """
        Returns the name of the domain complementary to `domain_name`. In other words, a ``*`` is either
        removed from the end of `domain_name`, or appended to it if not already there.

        :param domain_name:
            name of domain
        :return:
            name of complementary domain
        """
        return domain_name[:-1] if domain_name[-1] == '*' else domain_name + '*'

    def _is_independent(self) -> bool:
        """Return true if self is independent (not dependent or fixed).

        :return: [description]
        """
        return not self.dependent or self.fixed

    def _contains_any_independent_subdomain_recursively(self) -> bool:
        """Returns true if the subdomain graph rooted at this domain contains
        at least one independent subdomain.

        :rtype: bool
        """
        if self._is_independent():
            return True

        for sd in self._subdomains:
            if sd._contains_any_independent_subdomain_recursively():
                return True

        return False

    def _check_subdomain_graph_is_uniquely_assignable(self) -> None:
        """Checks that the subdomain graph that this domain is part of is
        uniquely assignable. Meaning that all paths from the root to the
        leaf of the subdomain graph contains exaclty one independent subdomain.
        """
        if self.parent is None:
            self._check_exactly_one_independent_subdomain_all_paths()
        else:
            self.parent._check_subdomain_graph_is_uniquely_assignable()

    def _check_exactly_one_independent_subdomain_all_paths(self) -> None:
        """Checks if all paths in the subdomains graph from the self to
        a leaf subdomain contains exactly one independent (dependent = False or
        fixed = True) subdomain (could be this one).

        :raises ValueError: if condition is not satisfied
        """
        self_independent = not self.dependent or self.fixed

        if self_independent:
            # Since this domain is independent, check that there are no more independent subdomains
            # in any children recursively
            for sd in self._subdomains:
                if sd._contains_any_independent_subdomain_recursively():
                    # Too many independent subdomains in this path
                    raise ValueError(f"Domain {self} is independent, but subdomain {sd} already contains an "
                                     f"independent subdomain in its subdomain graph")
        else:
            if len(self._subdomains) == 0:
                raise ValueError(f"Domain {self} is dependent and does not contain any subdomains.")
            # Since this domain is dependent, check that each subdomain has
            # exactly one independent subdomain in all paths.
            for sd in self._subdomains:
                try:
                    sd._check_exactly_one_independent_subdomain_all_paths()
                except ValueError as e:
                    raise ValueError(
                        f"Domain {self} is dependent and could not find exactly one independent subdomain "
                        f"in subdomain graph rooted at subdomain {sd}. The following error was found: {e}")

    def _check_acyclic_subdomain_graph(self, seen_domains: Optional[Set["Domain"]] = None) -> None:
        """Check to see if domain's subdomain graph contains a cycle.

        :param seen_domains: All the domains seen so far (used by implementation)
        :type seen_domains: Optional[Set["Domain"]]
        :raises ValueError: Cycle found.
        """
        if len(self._subdomains) > 0:
            if seen_domains is None:
                seen_domains = set()

            if self in seen_domains:
                raise ValueError(f"Domain {self} found twice in DFS")
            else:
                seen_domains.add(self)

            for sd in self._subdomains:
                try:
                    sd._check_acyclic_subdomain_graph(seen_domains)
                except ValueError as e:
                    raise ValueError(f"Cycle found in subdomain graph rooted at {self}. "
                                     f"Propogated from subdomain {sd}: {e}"
                                     )

    def all_domains_in_tree(self) -> List["Domain"]:
        """
        :return:
            list of all domains in the same subdomain tree as this domain (including itself)
        """
        domains = self._get_all_domains_from_parent()
        domains.extend(self._get_all_domains_from_this_subtree())
        return domains

    def all_domains_intersecting(self) -> List["Domain"]:
        """
        :return:
            list of all domains intersecting this one, meaning those domains in the subtree rooted
            at this domain (including itself), plus any ancestors of this domain.
        """
        domains = self.ancestors()
        domains.extend(self._get_all_domains_from_this_subtree())
        return domains

    def ancestors(self) -> List["Domain"]:
        """
        :return:
            list of all domains that are ancestors of this one, NOT including this domain
        """
        ancestor = self.parent
        all_ancestors = []
        while ancestor is not None:
            all_ancestors.append(ancestor)
            ancestor = ancestor.parent
        return all_ancestors

    def _get_all_domains_from_parent(self) -> List["Domain"]:
        # note that this gets "sibling/cousin" domains as well
        # call _ancestors to get only ancestors
        domains = []

        parent = self.parent
        if parent is not None:
            parent_domains = parent._get_all_domains_from_this_subtree(excluded_subdomain=self)
            domains.extend(parent_domains)
            domains.extend(parent._get_all_domains_from_parent())

        return domains

    def _get_all_domains_from_this_subtree(self, excluded_subdomain: Optional['Domain'] = None) \
            -> List["Domain"]:
        # includes itself
        domains = [self]
        for sd in self._subdomains:
            if sd != excluded_subdomain:
                domains.extend(sd._get_all_domains_from_this_subtree())
        return domains

    def has_pool(self) -> bool:
        """
        :return:
            whether a :any:`DomainPool` has been assigned to this :any:`Domain`
        """
        return self._pool is not None

    def contains_in_subtree(self, other: Domain) -> bool:
        """
        :param other:
            another :any:`Domain`
        :return:
            True if `self` contains `other` in its subtree of subdomains
        """
        # base case
        if self is other:
            return True

        # recursive case
        for subdomain in self._subdomains:
            if subdomain is other:
                return True
            if subdomain.contains_in_subtree(other):
                return True

        return False

    def independent_source(self) -> Domain:
        """
        Like :meth:`independent_ancestor_or_descendent`,
        but returns this Domain if it is already independent.

        :return:
            the independent :any:`Domain` that this domain depends on,
            which is *itself* if it is already independent
        """
        if self.dependent:
            return self.independent_ancestor_or_descendent()
        else:
            return self

    def independent_ancestor_or_descendent(self) -> Domain:
        """
        Find the independent ancestor or descendent of this dependent :any:`Domain`.
        Raises exception if this is not a dependent :any:`Domain`.

        :return:
            The independent ancestor or descendent of this :any:`Domain`.
        """
        if not self.dependent:
            raise ValueError('cannot call independent_ancestor_or_descendent on non-dependent Domain'
                             f' {self.name}')

        # first try ancestors
        domain = self
        while domain.parent is not None:
            domain = domain.parent
            if not domain.dependent:
                return domain

        # then try descendents
        independent_descendent = self._independent_descendent()
        if independent_descendent is None:
            raise ValueError(f'could not find an independent ancestor or descendent of domain {self.name}')
        return independent_descendent

    def _independent_descendent(self) -> Optional[Domain]:
        if not self.dependent:
            return self

        if len(self.subdomains) > 0:
            for subdomain in self.subdomains:
                independent_descendent = subdomain._independent_descendent()
                if independent_descendent is not None:
                    return independent_descendent

        return None


def domains_not_substrings_of_each_other_constraint(
        check_complements: bool = True, short_description: str = 'dom neq', weight: float = 1.0,
        min_length: int = 0,
        pairs: Optional[Iterable[Tuple[Domain, Domain]]] = None) -> DomainPairConstraint:
    """
    Returns constraint ensuring no two domains are substrings of each other.
    Note that this ensures that no two :any:`Domain`'s are equal if they are the same length.

    :param check_complements:
        whether to also ensure the check for Watson-Crick complements of the sequences
    :param short_description:
        short description of constraint suitable for logging to stdout
    :param weight:
        weight to assign to constraint
    :param min_length:
        minimum length substring to check.
        For instance if `min_length` is 4, then having two domains with sequences AAAA and CAAAAC would
        violate this constraint, but domains with sequences AAA and CAAAC would not.
    :param pairs:
        pairs of domains to check.
        By default all pairs of unequal domains are compared unless both are fixed.
    :return:
        a :any:`DomainPairConstraint` ensuring no two domain sequences contain each other as a substring
        (in particular, if they are equal length, then they are not the same domain)
    """

    # def evaluate(s1: str, s2: str, domain1: Optional[Domain], domain2: Optional[Domain]) -> float:
    def evaluate(seqs: Tuple[str, ...],
                 domains: Optional[Tuple[Domain, Domain]]) -> Tuple[float, str]:  # noqa
        s1, s2 = seqs
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        summary = ''
        score = 0.0
        passed = True
        if len(s1) >= min_length and s1 in s2:
            score = 1.0
            summary = f'{s1} is a length->={min_length} substring of {s2}'
            passed = False
        if check_complements:
            # by symmetry, only need to check c1 versus s2 for WC complement, since
            # (s1 not in s2 <==> c1 in c2) and (c1 in s2 <==> s1 in c2)
            c1 = dv.wc(s1)
            if len(c1) >= min_length and c1 in s2:
                msg = f'{c1} is a length->={min_length} substring of {s2}'
                if not passed:
                    summary += f'; {msg}'
                else:
                    summary = msg
                score += 1.0
        return score, summary

    return DomainPairConstraint(description='domains not substrings of each other',
                                short_description=short_description,
                                weight=weight,
                                check_domain_against_itself=False,
                                pairs=pairs,
                                evaluate=evaluate)


@dataclass
class IDTFields(JSONSerializable):
    """Data required when ordering DNA strands from the synthesis company
    `IDT (Integrated DNA Technologies) <https://www.idtdna.com/>`_.
    This data is used when automatically generating files used to order DNA from IDT.

    When exporting to IDT files via :py:meth:`Design.write_idt_plate_excel_file`
    or :py:meth:`Design.write_idt_bulk_input_file`, the field :py:data:`Strand.name` is used for the
    name if it exists, otherwise a reasonable default is chosen."""

    scale: str = default_idt_scale
    """Synthesis scale at which to synthesize the strand (third field in IDT bulk input:
    https://www.idtdna.com/site/order/oligoentry).
    Choices supplied by IDT at the time this was written: 
    ``"25nm"``, ``"100nm"``, ``"250nm"``, ``"1um"``, ``"5um"``, 
    ``"10um"``, ``"4nmU"``, ``"20nmU"``, ``"PU"``, ``"25nmS"``.
    """

    purification: str = default_idt_purification
    """Purification options (fourth field in IDT bulk input:
    https://www.idtdna.com/site/order/oligoentry). 
    Choices supplied by IDT at the time this was written: 
    ``"STD"``, ``"PAGE"``, ``"HPLC"``, ``"IEHPLC"``, ``"RNASE"``, ``"DUALHPLC"``, ``"PAGEHPLC"``.
    """

    plate: Optional[str] = None
    """Name of plate in case this strand will be ordered on a 96-well or 384-well plate.

    Optional field, but non-optional if :data:`IDTFields.well` is not ``None``.
    """

    well: Optional[str] = None
    """Well position on plate in case this strand will be ordered on a 96-well or 384-well plate.

    Optional field, but non-optional if :data:`IDTFields.plate` is not ``None``.
    """

    def __post_init__(self) -> None:
        _check_idt_string_not_none_or_empty(self.scale, 'scale')
        _check_idt_string_not_none_or_empty(self.purification, 'purification')
        if self.plate is None and self.well is not None:
            raise ValueError(f'IDTFields.plate cannot be None if IDTFields.well is not None\n'
                             f'IDTFields.well = {self.well}')
        if self.plate is not None and self.well is None:
            raise ValueError(f'IDTFields.well cannot be None if IDTFields.plate is not None\n'
                             f'IDTFields.plate = {self.plate}')

    def to_json_serializable(self, suppress_indent: bool = True,
                             **kwargs: Any) -> Union[NoIndent, Dict[str, Any]]:
        dct: Dict[str, Any] = dict(self.__dict__)
        if self.plate is None:
            del dct['plate']
        if self.well is None:
            del dct['well']
        return NoIndent(dct) if suppress_indent else dct

    @staticmethod
    def from_json_serializable(json_map: Dict[str, Any]) -> IDTFields:
        scale = mandatory_field(IDTFields, json_map, idt_scale_key)
        purification = mandatory_field(IDTFields, json_map, idt_purification_key)
        plate = json_map.get(idt_plate_key)
        well = json_map.get(idt_well_key)
        return IDTFields(scale=scale, purification=purification, plate=plate, well=well)

    def clone(self) -> IDTFields:
        return IDTFields(scale=self.scale, purification=self.purification,
                         plate=self.plate, well=self.well)

    def to_scadnano_idt(self) -> sc.IDTFields:
        return sc.IDTFields(scale=self.scale, purification=self.purification,
                            plate=self.plate, well=self.well)


def _check_idt_string_not_none_or_empty(value: str, field_name: str) -> None:
    if value is None:
        raise ValueError(f'field {field_name} in IDTFields cannot be None')
    if len(value) == 0:
        raise ValueError(f'field {field_name} in IDTFields cannot be empty')


default_strand_group = 'default_strand_group'

StrandLabel = TypeVar('StrandLabel')


@dataclass
class StrandPair(Part, Generic[StrandLabel, DomainLabel]):
    strand1: Strand
    strand2: Strand

    def __post_init__(self) -> None:
        # make this symmetric so make dict lookups work
        if self.strand1.name > self.strand2.name:
            self.strand1, self.strand2 = self.strand2, self.strand1

    # needed to avoid unhashable type error; see
    # https://docs.python.org/3/reference/datamodel.html#object.__hash__
    __hash__ = Part.__hash__

    @property
    def name(self) -> str:
        return f'{self.strand1.name}, {self.strand2.name}'

    def key(self) -> str:
        return f'StrandPair[{self.strand1.name}, {self.strand2.name}]'

    @staticmethod
    def name_of_part_type(self) -> str:
        return 'strand pair'

    def individual_parts(self) -> Tuple[Strand, ...]:
        return self.strand1, self.strand2

    @property
    def fixed(self) -> bool:
        return self.strand1.fixed and self.strand2.fixed


@dataclass
class Complex(Part, Generic[StrandLabel, DomainLabel]):
    strands: Tuple[Strand, ...]
    """The strands in this complex."""

    def __init__(self, *args: Strand) -> None:
        """
        Creates a complex of strands given as arguments, e.g., ``Complex(strand1, strand2)`` creates
        a 2-strand complex.
        """
        for strand in args:
            if not isinstance(strand, Strand):
                raise TypeError(f'must pass Strands to constructor for complex, not {strand}')
        self.strands = tuple(args)

    # needed to avoid unhashable type error; see
    # https://docs.python.org/3/reference/datamodel.html#object.__hash__
    __hash__ = Part.__hash__

    @property
    def name(self) -> str:
        strand_names = ', '.join(strand.name for strand in self.strands)
        return f'Complex[{strand_names}]'

    def key(self) -> str:
        return f'Complex[{self.name}]'

    @staticmethod
    def name_of_part_type(self) -> str:
        return 'complex'

    def individual_parts(self) -> Tuple[Strand, ...]:
        return self.strands

    def __iter__(self) -> Iterator[Strand]:
        return iter(self.strands)

    def __len__(self) -> int:
        return len(self.strands)

    def __getitem__(self, i: int) -> Strand:
        return self.strands[i]

    @property
    def fixed(self) -> bool:
        return all(strand.fixed for strand in self.strands)


@dataclass
class Strand(Part, JSONSerializable, Generic[StrandLabel, DomainLabel]):
    """Represents a DNA strand, made of several :any:`Domain`'s. """

    domains: List[Domain[DomainLabel]]
    """The :any:`Domain`'s on this :any:`Strand`, in order from 5' end to 3' end."""

    starred_domain_indices: FrozenSet[int]
    """Set of positions of :any:`Domain`'s in :py:data:`Strand.domains`
    on this :any:`Strand` that are starred."""

    group: str
    """Optional "group" field to describe strands that share similar properties."""

    _domain_names_concatenated: str
    """Concatenation of domain names; cached for efficiency since these are used in calculating 
    hash values."""

    _hash_domain_names_concatenated: int
    """Hash value of _domain_names_concatenated; cached for efficiency."""

    idt: Optional[IDTFields] = None
    """Fields used when ordering strands from the synthesis company IDT 
    (Integrated DNA Technologies, Coralville, IA). If present (i.e., not equal to :const:`None`)
    then the method :py:meth:`Design.write_idt_bulk_input_file` can be called to automatically
    generate an text file for ordering strands in test tubes: 
    https://www.idtdna.com/site/order/oligoentry,
    as can the method :py:meth:`Design.write_idt_plate_excel_file` for writing a Microsoft Excel 
    file that can be uploaded to IDT's website for describing DNA sequences to be ordered in 96-well
    or 384-well plates."""

    _name: Optional[str] = None
    """Optional name of strand."""

    modification_5p: Optional[dm.Modification5Prime] = None
    """
    5' modification; None if there is no 5' modification. 
    """

    modification_3p: Optional[dm.Modification3Prime] = None
    """
    3' modification; None if there is no 3' modification. 
    """

    modifications_int: Dict[int, dm.ModificationInternal] = field(default_factory=dict)
    """
    :any:`modifications.Modification`'s to the DNA sequence (e.g., biotin, Cy3/Cy5 fluorphores). 
    
    Maps index within DNA sequence to modification. If the internal modification is attached to a base 
    (e.g., internal biotin, /iBiodT/ from IDT), 
    then the index is that of the base.
    If it goes between two bases 
    (e.g., internal Cy3, /iCy3/ from IDT),
    then the index is that of the previous base, 
    e.g., to put a Cy3 between bases at indices 3 and 4, the index should be 3. 
    So for an internal modified base on a sequence of length n, the allowed indices are 0,...,n-1,
    and for an internal modification that goes between bases, the allowed indices are 0,...,n-2.
    """

    label: Optional[StrandLabel] = None
    """
    Optional generic "label" object to associate to this :any:`Strand`.

    Useful for associating extra information with the :any:`Strand` that will be serialized, for example,
    for DNA sequence design. It must be an object (e.g., a dict or primitive type such as str or int)
    that is naturally JSON serializable. (Calling
    `json.dumps <https://docs.python.org/3/library/json.html#json.dumps>`_
    on the object should succeed without having to specify a custom encoder.)
    """

    def __init__(self,
                 domains: Optional[List[Domain[DomainLabel]]] = None,
                 starred_domain_indices: Optional[Iterable[int]] = None,
                 group: str = default_strand_group,
                 name: Optional[str] = None,
                 label: Optional[StrandLabel] = None,
                 idt: Optional[IDTFields] = None,
                 ) -> None:
        """
        A :any:`Strand` can be created only by listing explicit :any:`Domain` objects
        via parameter `domains`. To specify a :any:`Strand` by giving domain *names*, see the method
        :meth:`Design.add_strand`.

        :param domains:
            list of :any:`Domain`'s on this :any:`Strand`
        :param starred_domain_indices:
            Indices of :any:`Domain`'s in `domains` that are starred.
        :param group:
            name of group of this :any:`Strand`.
        :param name:
            Name of this :any:`Strand`.
        :param label:
            Label to associate with this :any:`Strand`.
        :param idt:
            :any:`IDTFields` object to associate with this :any:`Strand`; needed to call
            methods for exporting to IDT formats (e.g., :meth:`Strand.write_idt_bulk_input_file`)
        """
        self._all_intersecting_domains = None
        self.group = group
        self._name = name

        # XXX: moved this check to Design constructor to allow subdomain graphs to be
        # constructed gradually while building up the design
        # Check that each base in the sequence is assigned by exactly one
        # independent subdomain.
        # for d in cast(List[Domain], domains):
        #     d._check_acyclic_subdomain_graph()  # noqa
        #     d._check_subdomain_graph_is_uniquely_assignable()  # noqa

        self.domains = list(domains)  # type: ignore
        self.starred_domain_indices = frozenset(starred_domain_indices)  # type: ignore
        self.label = label
        self.idt = idt

        # don't know why we have to do this, but we get a missing attribute error otherwise
        # https://stackoverflow.com/questions/70986725/python-dataclass-attribute-missing-when-using-explicit-init-constructor-and
        self.modifications_int = {}

        self.compute_derived_fields()

    @staticmethod
    def name_of_part_type(self) -> str:
        return 'strand'

    def key(self) -> str:
        return f'Strand({self._hash_domain_names_concatenated})'

    # needed to avoid unhashable type error; see
    # https://docs.python.org/3/reference/datamodel.html#object.__hash__
    __hash__ = Part.__hash__

    def individual_parts(self) -> Tuple[Strand, ...]:
        return self,

    def clone(self, name: Optional[str]) -> Strand:
        """
        Returns a copy of this :any:`Strand`. The copy is "shallow" in that the :any:`Domain`'s are shared.
        This is useful for creating multiple versions of each :any:`Strand`, e.g., for having a
        variant with an extension.

        WARNING: the :data:`Strand.label` will be shared between them. If it should be copied,
        this must be done manually. A shallow copy of it can be made by setting

        :param name:
            new name to give this Strand
        :return:
            A copy of this :any:`Strand`.
        """
        domains = list(self.domains)
        starred_domain_indices = list(self.starred_domain_indices)
        name = name if name is not None else self.name
        idt = None if self.idt is None else self.idt.clone()
        return Strand(domains=domains, starred_domain_indices=starred_domain_indices, name=name,
                      group=self.group, label=self.label, idt=idt)

    def compute_derived_fields(self):
        """
        Re-computes derived fields of this :any:`Strand`. Should be called after modifications to the
        Strand. (Done automatically at the start of :meth:`search.search_for_dna_sequences`.)
        """
        self._domain_names_concatenated = '-'.join(self.domain_names_tuple())
        self._hash_domain_names_concatenated = hash(self._domain_names_concatenated)
        self._compute_all_intersecting_domains()

    def all_intersecting_domains(self) -> List[Domain]:
        if self._all_intersecting_domains is None:
            self._compute_all_intersecting_domains()
        return self._all_intersecting_domains

    def _compute_all_intersecting_domains(self) -> None:
        # Check that each base in the sequence is assigned by exactly one independent subdomain.
        # We normally wait until the Design constructor to check for this to raise an exception,
        # but here we just check to see whether to bother computing self._all_intersecting_domains.
        for d in cast(List[Domain], self.domains):
            try:
                d._check_acyclic_subdomain_graph()  # noqa
                d._check_subdomain_graph_is_uniquely_assignable()  # noqa
            except ValueError:
                return

        self._all_intersecting_domains = []
        for direct_domain in self.domains:
            for domain_in_tree in direct_domain.all_domains_intersecting():
                if domain_in_tree not in self._all_intersecting_domains:
                    self._all_intersecting_domains.append(domain_in_tree)

    def intersects_domain(self, domain: Domain) -> bool:
        """
        :param domain:
            domain to test for intersection
        :return:
            whether this strand intersects `domain`, which is true if either `domain` is in the list
            :data:`Strand.domains`, or if any of those domains have `domain` in their hierarchical tree
            as a subdomain or an ancestor
        """
        return domain in self.all_intersecting_domains()

    def length(self) -> int:
        """
        :return:
            Sum of lengths of :any:`Domain`'s in this :any:`Strand`.
            Each :any:`Domain` must have a :any:`DomainPool` assigned so that the length is defined.
        """
        return sum(domain.get_length() for domain in self.domains)

    def domain_names_concatenated(self, delim: str = '-') -> str:
        """
        :param delim:
            Delimiter to put between domain names.
        :return:
            names of :any:`Domain`'s in this :any:`Strand`, concatenated with `delim` in between.
        """
        return delim.join(self.domain_names_tuple())

    def domain_names_tuple(self) -> Tuple[str, ...]:
        """
        :return: tuple of names of :any:`Domain`'s in this :any:`Strand`.
        """
        domain_names: List[str] = []
        for idx, domain in enumerate(self.domains):
            is_starred = idx in self.starred_domain_indices
            domain_names.append(domain.get_name(is_starred))
        return tuple(domain_names)

    def idt_dna_sequence(self) -> str:
        """
        :return: DNA sequence as it needs to be typed to order from IDT, with
            :py:data:`Modification5Prime`'s,
            :py:data:`Modification3Prime`'s,
            and
            :py:data:`ModificationInternal`'s represented with text codes, e.g., "/5Biosg/ACGT" for sequence
            ACGT with a 5' biotin modification.
        """
        self._ensure_modifications_legal(check_offsets_legal=True)

        ret_list: List[str] = []
        if self.modification_5p is not None and self.modification_5p.idt_text is not None:
            ret_list.append(self.modification_5p.idt_text)

        for offset, base in enumerate(self.sequence(delimiter='')):
            ret_list.append(base)
            if offset in self.modifications_int:  # if internal mod attached to base, replace base
                mod = self.modifications_int[offset]
                if mod.idt_text is not None:
                    if mod.allowed_bases is not None:
                        if base not in mod.allowed_bases:
                            msg = f'internal modification {mod} can only replace one of these bases: ' \
                                  f'{",".join(mod.allowed_bases)}, but the base at offset {offset} is {base}'
                            raise ValueError(msg)
                        ret_list[-1] = mod.idt_text  # replace base with modified base
                    else:
                        ret_list.append(mod.idt_text)  # append modification between two bases

        if self.modification_3p is not None and self.modification_3p.idt_text is not None:
            ret_list.append(self.modification_3p.idt_text)

        return ''.join(ret_list)

    def _ensure_modifications_legal(self, check_offsets_legal: bool = False) -> None:
        if check_offsets_legal:
            mod_i_offsets_list = list(self.modifications_int.keys())
            min_offset = min(mod_i_offsets_list) if len(mod_i_offsets_list) > 0 else None
            max_offset = max(mod_i_offsets_list) if len(mod_i_offsets_list) > 0 else None
            if min_offset is not None and min_offset < 0:
                raise ValueError(f"smallest offset is {min_offset} but must be nonnegative: "
                                 f"{self.modifications_int}")
            if max_offset is not None and max_offset > len(self.sequence(delimiter='')):
                raise ValueError(f"largest offset is {max_offset} but must be at most "
                                 f"{len(self.sequence(delimiter=''))}: "
                                 f"{self.modifications_int}")

    def to_json_serializable(self, suppress_indent: bool = True) -> Union[NoIndent, Dict[str, Any]]:
        """
        :return:
            Dictionary ``d`` representing this :any:`Strand` that is "naturally" JSON serializable,
            by calling ``json.dumps(d)``.
        """
        dct: Dict[str, Any] = {name_key: self.name, group_key: self.group}

        domains_list = [domain.name for domain in self.domains]
        dct[domain_names_key] = NoIndent(domains_list) if suppress_indent else domains_list

        starred_domain_indices_list = sorted(list(self.starred_domain_indices))
        dct[starred_domain_indices_key] = NoIndent(starred_domain_indices_list) if suppress_indent \
            else starred_domain_indices_list

        if self.label is not None:
            dct[label_key] = NoIndent(self.label) if suppress_indent else self.label

        if self.idt is not None:
            dct[idt_key] = self.idt.to_json_serializable(suppress_indent)

        if self.modification_5p is not None:
            dct[dm.modification_5p_key] = self.modification_5p.id

        if self.modification_3p is not None:
            dct[dm.modification_3p_key] = self.modification_3p.id

        if len(self.modifications_int) > 0:
            mods_dict = {}
            for offset, mod in self.modifications_int.items():
                mods_dict[f"{offset}"] = mod.id
            dct[dm.modifications_int_key] = NoIndent(mods_dict) if suppress_indent else mods_dict

        return dct

    @staticmethod
    def from_json_serializable(json_map: Dict[str, Any],
                               domain_with_name: Dict[str, Domain[DomainLabel]],
                               label_decoder: Callable[[Any], StrandLabel] = (lambda label: label),
                               ) -> Strand[StrandLabel, DomainLabel]:
        """
        :return:
            :any:`Strand` represented by dict `json_map`, assuming it was created by
            :py:meth:`Strand.to_json_serializable`.
        """
        name: str = mandatory_field(Strand, json_map, name_key)
        domain_names_json = mandatory_field(Strand, json_map, domain_names_key)
        domains: List[Domain[DomainLabel]] = [domain_with_name[name] for name in domain_names_json]
        starred_domain_indices = mandatory_field(Strand, json_map, starred_domain_indices_key)

        group = json_map.get(group_key, default_strand_group)

        label_json = json_map.get(label_key)
        label = label_decoder(label_json)

        strand: Strand[StrandLabel, DomainLabel] = Strand(
            domains=domains, starred_domain_indices=starred_domain_indices,
            group=group, name=name, label=label)
        return strand

    def __repr__(self) -> str:
        return self.name

    def unstarred_domains(self) -> List[Domain[DomainLabel]]:
        """
        :return: list of unstarred :any:`Domain`'s in this :any:`Strand`, in order they appear in
                 :py:data:`Strand.domains`
        """
        return [domain for idx, domain in enumerate(self.domains) if idx not in self.starred_domain_indices]

    def starred_domains(self) -> List[Domain[DomainLabel]]:
        """
        :return: list of starred :any:`Domain`'s in this :any:`Strand`, in order they appear in
                 :py:data:`Strand.domains`
        """
        return [domain for idx, domain in enumerate(self.domains) if idx in self.starred_domain_indices]

    def unstarred_domains_set(self) -> OrderedSet[Domain[DomainLabel]]:
        """
        :return: set of unstarred :any:`Domain`'s in this :any:`Strand`
        """
        return OrderedSet(self.unstarred_domains())

    def starred_domains_set(self) -> OrderedSet[Domain[DomainLabel]]:
        """
        :return: set of starred :any:`Domain`'s in this :any:`Strand`
        """
        return OrderedSet(self.starred_domains())

    def sequence(self, delimiter: str = '') -> str:
        """
        :param delimiter:
            Delimiter string to place between sequences of each :any:`Domain` in this :any:`Strand`.
            For instance, if `delimiter` = ``'--'``, then it will return a string such as
            ``ACGTAGCTGA--CGCTAGCTGA--CGATCGATC--GCGATCGAT``
        :return:
            DNA sequence assigned to this :any:`Strand`, calculated by concatenating all sequences
            assigned to its :any:`Domain`'s.
        :raises ValueError:
            if any :any:`Domain` of this :any:`Strand` does not have a sequence assigned
        """
        seqs = []
        for idx, domain in enumerate(self.domains):
            starred = idx in self.starred_domain_indices
            seqs.append(domain.concrete_sequence(starred))
        return delimiter.join(seqs)

    def assign_dna(self, sequence: str) -> None:
        """
        :param sequence:
            DNA sequence to assign to this :any:`Strand`.
            Must have length = :py:meth:`Strand.length`.
        """
        if not self.length() == len(sequence):
            raise ValueError(f'Strand {self.name} has length {self.length()}, but DNA sequence '
                             f'{sequence} has length {len(sequence)}')
        start = 0
        for domain in self.domains:
            end = start + domain.get_length()
            domain_sequence = sequence[start:end]
            domain.set_sequence(domain_sequence)
            start = end

    @property
    def fixed(self) -> bool:
        """True if every :any:`Domain` on this :any:`Strand` has a fixed DNA sequence."""
        return all(domain.fixed for domain in self.domains)

    def unfixed_domains(self) -> List[Domain[DomainLabel]]:
        """
        :return: all :any:`Domain`'s in this :any:`Strand` where :py:data:`Domain.fixed` is False
        """
        return [domain for domain in self.domains if not domain.fixed]

    @property
    def name(self) -> str:
        """
        :return: name of this :any:`Strand` if it was assigned one, otherwise :any:`Domain` names are
                 concatenated with '-' joining them
        """
        return self.domain_names_concatenated() if self._name is None else self._name

    @name.setter
    def name(self, new_name: str) -> None:
        """
        Sets name of this :any:`Strand`.
        """
        self._name = new_name

    # def complementary_domains(self, other: Strand) -> List[Domain]:
    #     """
    #     :param other: another :any:`Strand`
    #     :return: list of :any:`Domain`'s that are complementary between this :any:`Strand` and `other`,
    #              in the order they appear in this :any:`Strand`.
    #     """

    def address_of_domain(self, domain_idx: int) -> 'StrandDomainAddress':
        """Returns :any:`StrandDomainAddress` of the domain located at domain_idx

        :rparam domain_idx: Index of domain
        """
        return StrandDomainAddress(self, domain_idx)

    def address_of_nth_domain_occurence(self, domain_name: str, n: int,
                                        forward=True) -> 'StrandDomainAddress':
        """
        Returns :any:`StrandDomainAddress` of the `n`'th occurence of domain named `domain_name`.

        :param domain_name:
            name of :any:`Domain` to find address of
        :param n:
            which occurrence (in order on the :any:`Strand`)
            of :any:`Domain` with name `domain_name` to find address of.
        :param forward:
            if True, starts searching from 5' end, otherwise starts searching from 3' end.
        :return:
            :any:`StrandDomainAddress` of the `n`'th occurence of domain named `domain_name`.
        """
        if n < 1:
            raise ValueError(f'n needs to be at least 1')
        domain_names = self.domain_names_tuple()
        idx = -1
        occurences = 0

        itr = range(0, len(domain_names)) if forward else range(len(domain_names) - 1, -1, -1)

        for i in itr:
            if domain_names[i] == domain_name:
                occurences += 1
                if occurences == n:
                    idx = i
                    break
        if idx == -1:
            raise ValueError(f'{self} contained less than {n} occurrences of domain {domain_name}')

        return StrandDomainAddress(self, idx)

    def address_of_first_domain_occurence(self, domain_name: str) -> 'StrandDomainAddress':
        """
        Returns :any:`StrandDomainAddress` of the first occurrence of domain named domain_name
        starting from the 5' end.
        """
        return self.address_of_nth_domain_occurence(domain_name, 1)

    def address_of_last_domain_occurence(self, domain_name: str) -> 'StrandDomainAddress':
        """
        Returns :any:`StrandDomainAddress` of the nth occurrence of domain named domain_name
        starting from the 3' end.
        """
        return self.address_of_nth_domain_occurence(domain_name, 1, forward=False)

    def append_domain(self, domain: Domain, starred: bool = False) -> None:
        """
        Appends `domain` to 3' end of this :any:`Strand`.

        :param domain:
            :any:`Domain` to append
        :param starred:
            whether `domain` is starred
        """
        self.insert_domain(len(self.domains), domain, starred)

    def prepend_domain(self, domain: Domain, starred: bool = False) -> None:
        """
        Prepends `domain` to 5' end of this :any:`Strand` (i.e., the beginning of the :any:`Strand`).

        :param domain:
            :any:`Domain` to prepend
        :param starred:
            whether `domain` is starred
        """
        self.insert_domain(0, domain, starred)

    def insert_domain(self, idx: int, domain: Domain, starred: bool = False) -> None:
        """
        Inserts `domain` at index `idx` of this :any:`Strand`, with same semantics as Python's List.insert.
        For example, ``strand.insert(0, domain)`` is equivalent to ``strand.prepend_domain(domain)``
        and ``strand.insert(len(strand.domains), domain)`` is equivalent to ``strand.append_domain(domain)``.

        :param idx:
            index at which to insert `domain` into this :any:`Strand`
        :param domain:
            :any:`Domain` to append
        :param starred:
            whether `domain` is starred
        """
        self.domains.insert(idx, domain)

        new_starred_idx = frozenset([idx]) if starred else frozenset()

        # increment all starred indices >= idx
        starred_domain_indices_at_least_idx = frozenset([idx_
                                                         for idx_ in self.starred_domain_indices
                                                         if idx_ >= idx])
        starred_domain_indices_at_least_idx_inc = frozenset([idx_ + 1
                                                             for idx_ in starred_domain_indices_at_least_idx
                                                             if idx_ >= idx])
        # remove old starred indices >= idx, union in their increments,
        # and if new domain is starred, union it in also
        self.starred_domain_indices = self.starred_domain_indices.difference(
            starred_domain_indices_at_least_idx).union(starred_domain_indices_at_least_idx_inc).union(
            new_starred_idx)


def remove_duplicates(lst: Iterable[T]) -> List[T]:
    """
    :param lst:
        an Iterable of objects
    :return:
        a List consisting of elements of `lst` with duplicates removed,
        while preserving iteration order of `lst`
        (naive approach using Python set would not preserve order,
        since iteration order of Python sets is not specified)
    """
    # XXX: be careful; original version used set to remove duplicates, but that has unspecified
    # insertion order, even though Python 3.7 dicts preserve insertion order:
    # https://softwaremaniacs.org/blog/2020/02/05/dicts-ordered/
    seen: Set[T] = set()
    seen_add = seen.add
    return [x for x in lst if not (x in seen or seen_add(x))]


def _export_dummy_scadnano_design_for_idt_export(strands: Iterable[Strand]) -> sc.Design:
    """
    Exports a dummy scadnano design from this dsd :any:`Design`.
    Useful for reusing scadnano methods such as to_idt_bulk_input_format.

    :param strands:
        strands to export
    :return:
        a "dummy" scadnano design, where domains are positioned arbitrarily on helices,
        with the only goal to make the scadnano Design legal
    """
    helices = [sc.Helix(max_offset=strand.length()) for strand in strands]
    sc_strands = []
    for helix_idx, strand in enumerate(strands):
        idt_export = strand.idt.to_scadnano_idt() if strand.idt is not None else None
        sc_domains = []
        prev_end = 0
        for domain in strand.domains:
            sc_domain = sc.Domain(helix=helix_idx, forward=True,
                                  start=prev_end, end=prev_end + domain.get_length())
            prev_end = sc_domain.end
            sc_domains.append(sc_domain)
        sc_strand = sc.Strand(domains=sc_domains, idt=idt_export,
                              dna_sequence=strand.sequence(), name=strand.name)

        # handle modifications
        if strand.modification_5p is not None:
            mod = strand.modification_5p
            sc_mod = sc.Modification5Prime(idt_text=mod.idt_text, id=mod.id, display_text=mod.idt_text)
            sc_strand.modification_5p = sc_mod
        if strand.modification_3p is not None:
            mod = strand.modification_3p
            sc_mod = sc.Modification3Prime(idt_text=mod.idt_text, id=mod.id, display_text=mod.idt_text)
            sc_strand.modification_3p = sc_mod
        if len(strand.modifications_int) > 0:
            for offset, mod in strand.modifications_int.items():
                sc_mod = sc.ModificationInternal(idt_text=mod.idt_text, id=mod.id, display_text=mod.idt_text,
                                                 allowed_bases=mod.allowed_bases)
                sc_strand.modifications_int[offset] = sc_mod

        sc_strands.append(sc_strand)
    design = sc.Design(helices=helices, strands=sc_strands, grid=sc.square)
    return design


_96WELL_PLATE_ROWS: List[str] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
_96WELL_PLATE_COLS: List[int] = list(range(1, 13))

_384WELL_PLATE_ROWS: List[str] = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
_384WELL_PLATE_COLS: List[int] = list(range(1, 25))


@unique
class PlateType(int, Enum):
    """Represents two different types of plates in which DNA sequences can be ordered."""

    wells96 = 96
    """96-well plate."""

    wells384 = 384
    """384-well plate."""

    def rows(self) -> List[str]:
        return _96WELL_PLATE_ROWS if self is PlateType.wells96 else _384WELL_PLATE_ROWS

    def cols(self) -> List[int]:
        return _96WELL_PLATE_COLS if self is PlateType.wells96 else _384WELL_PLATE_COLS

    def num_wells_per_plate(self) -> int:
        """
        :return:
            number of wells in this plate type
        """
        if self is PlateType.wells96:
            return 96
        elif self is PlateType.wells384:
            return 384
        else:
            raise AssertionError('unreachable')

    def min_wells_per_plate(self) -> int:
        """
        :return:
            minimum number of wells in this plate type to avoid extra charge by IDT
        """
        if self is PlateType.wells96:
            return 24
        elif self is PlateType.wells384:
            return 96
        else:
            raise AssertionError('unreachable')


@dataclass
class Design(Generic[StrandLabel, DomainLabel], JSONSerializable):
    """
    Represents a complete design, i.e., a set of DNA :any:`Strand`'s with domains,
    and :any:`Constraint`'s on the sequences
    to assign to them via :py:meth:`search.search_for_dna_sequences`.
    """

    strands: List[Strand[StrandLabel, DomainLabel]]
    """List of all :any:`Strand`'s in this :any:`Design`."""

    _domains_interned: Dict[str, Domain]

    #################################################
    # derived fields, so not specified in constructor

    domains: List[Domain[DomainLabel]] = field(init=False)
    """
    List of all :any:`Domain`'s in this :any:`Design`. (without repetitions)

    Computed from :py:data:`Design.strands`, so not specified in constructor.
    """

    strands_by_group_name: Dict[str, List[Strand[StrandLabel, DomainLabel]]] = field(init=False)
    """
    Dict mapping each group name to a list of the :any:`Strand`'s in this :any:`Design` in the group.

    Computed from :py:data:`Design.strands`, so not specified in constructor.
    """

    domain_pools_to_domain_map: Dict[DomainPool, List[Domain]] = field(init=False)
    """
    Dict mapping each :any:`DomainPool` to a list of the :any:`Domain`'s in this :any:`Design` in the pool.

    Computed from :py:data:`Design.strands`, so not specified in constructor.
    """

    domains_by_name: Dict[str, Domain] = field(init=False)
    """
    Dict mapping each name of a :any:`Domain` to the :any:`Domain`'s in this :any:`Design`.

    Computed from :py:data:`Design.strands`, so not specified in constructor.
    """

    def __init__(self, strands: Iterable[Strand] = ()) -> None:
        """
        :param strands:
            the :any:`Strand`'s in this :any:`Design`
        """
        self.strands = strands if isinstance(strands, list) else list(strands)
        self.check_all_subdomain_graphs_acyclic()
        self.check_all_subdomain_graphs_uniquely_assignable()
        self.compute_derived_fields()
        self._domains_interned = {}

    def compute_derived_fields(self) -> None:
        """
        Computes derived fields of this :any:`Design`. Used to ensure that all fields are valid in case
        the :any:`Design` was manually modified after being created, before running
        :meth:`search.search_for_dna_sequences`.
        """
        # Get domains not explicitly listed on strands that are part of domain tree.
        # Also set up quick access to domain by name, and ensure each domain name unique.
        self.domains_by_name = {}
        domains = []
        for strand in self.strands:
            for domain_in_strand in strand.domains:
                domains_in_tree = domain_in_strand.all_domains_in_tree()
                domains.extend(domains_in_tree)
                for domain_in_tree in domains_in_tree:
                    name = domain_in_tree.name
                    if name in self.domains_by_name and domain_in_tree is not self.domains_by_name[name]:
                        raise ValueError(f'domain names must be unique, '
                                         f'but I found two different domains with name {domain_in_tree.name}')
                    self.domains_by_name[domain_in_tree.name] = domain_in_tree

        self.domains = remove_duplicates(domains)

        self.strands_by_group_name = defaultdict(list)
        for strand in self.strands:
            self.strands_by_group_name[strand.group].append(strand)

        self.store_domain_pools()

        for strand in self.strands:
            strand.compute_derived_fields()

    def to_json(self) -> str:
        """
        :return:
            JSON string representing this :any:`Design`.
        """
        self.store_domain_pools()
        # XXX: disabled the indent suppression because it takes a LONG time for a large Design to
        # convert the NoIndent instances. Since people tend not to look at design.json files that much
        # (unlike with scadnano scripting, for instance), hopefully this doesn't matter.
        # return json_encode(self, suppress_indent=True)
        return json_encode(self, suppress_indent=False)

    def to_json_serializable(self, suppress_indent: bool = True) -> Dict[str, Any]:
        """
        :param suppress_indent:
            Whether to suppress indentation of some objects using the NoIndent object.
        :return:
            Dictionary ``d`` representing this :any:`Design` that is "naturally" JSON serializable,
            by calling ``json.dumps(d)``.
        """

        dct = {
            strands_key: [strand.to_json_serializable(suppress_indent) for strand in self.strands],
            domains_key: [domain.to_json_serializable(suppress_indent) for domain in self.domains],
            domain_pools_key: [pool.to_json_serializable(suppress_indent) for pool in self.domain_pools()]
        }

        # modifications
        mods = self.modifications()
        if len(mods) > 0:
            mods_dict = {}
            for mod in mods:
                if mod.id not in mods_dict:
                    mods_dict[mod.id] = mod.to_json_serializable(suppress_indent)
            dct[dm.design_modifications_key] = mods_dict

        return dct

    @staticmethod
    def from_design_file(filename: str,
                         strand_label_decoder: Callable[[Any], StrandLabel] = lambda label: label,
                         domain_label_decoder: Callable[[Any], DomainLabel] = lambda label: label,
                         ) -> Design[StrandLabel, DomainLabel]:
        """
        :param filename:
            name of JSON file describing the :any:`Design`
        :param domain_label_decoder:
            Function that transforms JSON representation of :py:data:`Domain.label` into the proper type.
        :param strand_label_decoder:
            Function that transforms JSON representation of :py:data:`Strand.label` into the proper type.
        :return:
            :any:`Design` described by the JSON file with name `filename`, assuming it was created using
            :py:meth`Design.to_json`.
        """
        with open(filename, 'r') as f:
            json_str = f.read()
        return Design.from_json(json_str, strand_label_decoder, domain_label_decoder)

    @staticmethod
    def from_json(json_str: str,
                  strand_label_decoder: Callable[[Any], StrandLabel] = lambda label: label,
                  domain_label_decoder: Callable[[Any], DomainLabel] = lambda label: label,
                  ) -> Design[StrandLabel, DomainLabel]:
        """
        :param json_str:
            The string representing the :any:`Design` as a JSON object.
        :param domain_label_decoder:
            Function that transforms JSON representation of :py:data:`Domain.label` into the proper type.
        :param strand_label_decoder:
            Function that transforms JSON representation of :py:data:`Strand.label` into the proper type.
        :return:
            :any:`Design` described by this JSON string, assuming it was created using
            :py:meth`Design.to_json`.
        """
        json_map = json.loads(json_str)
        design: Design[StrandLabel, DomainLabel] = Design.from_json_serializable(
            json_map, domain_label_decoder=domain_label_decoder, strand_label_decoder=strand_label_decoder)
        return design

    @staticmethod
    def from_json_serializable(json_map: Dict[str, Any],
                               domain_label_decoder: Callable[[Any], DomainLabel] = lambda label: label,
                               strand_label_decoder: Callable[[Any], StrandLabel] = lambda label: label,
                               ) -> 'Design[StrandLabel, DomainLabel]':
        """
        :param json_map:
            JSON serializable object encoding this :any:`Design`, as returned by
            :py:meth:`Design.to_json_serializable`.
        :param domain_label_decoder:
            Function that transforms JSON representation of :py:data:`Domain.label` into the proper type.
        :param strand_label_decoder:
            Function that transforms JSON representation of :py:data:`Strand.label` into the proper type.
        :return:
            :any:`Design` represented by dict `json_map`, assuming it was created by
            :py:meth:`Design.to_json_serializable`. No constraints are populated.
        """
        pools_json = mandatory_field(Design, json_map, domain_pools_key)
        pools: List[DomainPool] = [DomainPool.from_json_serializable(pool_json) for pool_json in pools_json]
        pool_with_name: Dict[str, DomainPool] = {pool.name: pool for pool in pools}

        domains_json = mandatory_field(Design, json_map, domains_key)
        domains: List[Domain] = [
            Domain.from_json_serializable(domain_json, pool_with_name=pool_with_name,
                                          label_decoder=domain_label_decoder)
            for domain_json in domains_json]
        domain_with_name = {domain.name: domain for domain in domains}

        strands_json = mandatory_field(Design, json_map, strands_key)
        strands = [Strand.from_json_serializable(
            json_map=strand_json, domain_with_name=domain_with_name,
            label_decoder=strand_label_decoder)
            for strand_json in strands_json]

        # modifications in whole design
        if dm.design_modifications_key in json_map:
            all_mods_json = json_map[dm.design_modifications_key]
            all_mods = {}
            for mod_key, mod_json in all_mods_json.items():
                mod = dm.Modification.from_json(mod_json)
                mod = dataclasses.replace(mod, id=mod_key)
                all_mods[mod_key] = mod
            Design.assign_modifications_to_strands(strands, strands_json, all_mods)

        return Design(strands=strands)

    def add_strand(self,
                   domain_names: Optional[List[str]] = None,
                   domains: Optional[List[Domain[DomainLabel]]] = None,
                   starred_domain_indices: Optional[Iterable[int]] = None,
                   group: str = default_strand_group,
                   name: Optional[str] = None,
                   label: Optional[StrandLabel] = None,
                   idt: Optional[IDTFields] = None,
                   ) -> Strand:
        """
        This is an alternative way to create strands instead of calling the :any:`Strand` constructor
        explicitly. It behaves similarly to the :any:`Strand` constructor, but it has an option
        to specify :any:`Domain`'s simply by giving a name.

        A :any:`Strand` can be created either by listing explicit :any:`Domain` objects via parameter
        `domains` (as in the :any:`Strand` constructor), or by giving names via parameter `domain_names`.
        If `domain_names` is specified, then by convention those that end with a ``*`` are
        assumed to be starred. Also, :any:`Domain`'s created in this way are "interned" as variables
        in a cache stored in the :any:`Design` object;
        no two :any:`Domain`'s with the same name in this design will be created,
        and subsequent uses of the same name will refer to the same :any:`Domain` object.

        :param domain_names:
            Names of the :any:`Domain`'s on this :any:`Strand`.
            Mutually exclusive with :py:data:`Strand.domains` and :py:data:`Strand.starred_domain_indices`.
        :param domains:
            list of :any:`Domain`'s on this :any:`Strand`.
            Mutually exclusive with :py:data:`Strand.domain_names`, and must be specified jointly with
            :py:data:`Strand.starred_domain_indices`.
        :param starred_domain_indices:
            Indices of :any:`Domain`'s in `domains` that are starred.
            Mutually exclusive with :py:data:`Strand.domain_names`, and must be specified jointly with
            :py:data:`Strand.domains`.
        :param group:
            name of group of this :any:`Strand`.
        :param name:
            Name of this :any:`Strand`.
        :param label:
            Label to associate with this :any:`Strand`.
        :param idt:
            :any:`IDTFields` object to associate with this :any:`Strand`; needed to call
            methods for exporting to IDT formats (e.g., :meth:`Strand.write_idt_bulk_input_file`)
        :return:
            the :any:`Strand` that is created
        """
        if (domain_names is not None and not (domains is None and starred_domain_indices is None)) or \
                (domain_names is None and not (domains is not None and starred_domain_indices is not None)):
            raise ValueError('exactly one of domain_names or '
                             'domains and starred_domain_indices must be non-None\n'
                             f'domain_names: {domain_names}\n'
                             f'domains: {domains}\n'
                             f'starred_domain_indices: {starred_domain_indices}')

        elif domain_names is not None:
            domains = []
            starred_domain_indices = OrderedSet()
            for idx, domain_name in enumerate(domain_names):
                is_starred = domain_name.endswith('*')
                if is_starred:
                    domain_name = domain_name[:-1]

                # domain = Domain(name) if name not in _domains_interned else _domains_interned[name]
                domain: Domain
                if domain_name not in self._domains_interned:
                    domain = Domain(name=domain_name)
                    self._domains_interned[domain_name] = domain
                else:
                    domain = self._domains_interned[domain_name]

                domains.append(domain)
                if is_starred:
                    starred_domain_indices.add(idx)

        domains_of_strand = list(domains)  # type: ignore
        strand = Strand(domains=domains_of_strand,
                        starred_domain_indices=starred_domain_indices,
                        group=group,
                        name=name,
                        label=label,
                        idt=idt)

        for existing_strand in self.strands:
            if strand.name == existing_strand.name:
                raise ValueError(f'strand name {strand.name} already exists for this strand:\n'
                                 f'  {existing_strand}\n'
                                 f'so it cannot be used for the new strand\n'
                                 f'  {strand}')
        self.strands.append(strand)

        for domain_in_strand in strand.domains:
            domains_in_tree = domain_in_strand.all_domains_in_tree()
            for domain in domains_in_tree:
                if domain not in self.domains:
                    self.domains.append(domain)
                name = domain.name
                if name in self.domains_by_name and domain is not self.domains_by_name[name]:
                    raise ValueError(f'domain names must be unique, '
                                     f'but I found two different domains with name {domain.name}')
                self.domains_by_name[domain.name] = domain

        return strand

    @staticmethod
    def assign_modifications_to_strands(strands: List[Strand], strand_jsons: List[dict],
                                        all_mods: Dict[str, dm.Modification]) -> None:
        for strand, strand_json in zip(strands, strand_jsons):
            if dm.modification_5p_key in strand_json:
                mod_name = strand_json[dm.modification_5p_key]
                strand.modification_5p = cast(dm.Modification5Prime, all_mods[mod_name])
            if dm.modification_3p_key in strand_json:
                mod_name = strand_json[dm.modification_3p_key]
                strand.modification_3p = cast(dm.Modification3Prime, all_mods[mod_name])
            if dm.modifications_int_key in strand_json:
                mod_names_by_offset = strand_json[dm.modifications_int_key]
                for offset_str, mod_name in mod_names_by_offset.items():
                    offset = int(offset_str)
                    strand.modifications_int[offset] = cast(dm.ModificationInternal, all_mods[mod_name])

    def modifications(self, mod_type: Optional[dm.ModificationType] = None) -> Set[dm.Modification]:
        """
        Returns either set of all :any:`modifications.Modification`'s in this :any:`Design`,
        or set of all modifications of a given type (5', 3', or internal).

        :param mod_type:
            type of modifications (5', 3', or internal); if not specified, all three types are returned
        :return:
            Set of all modifications in this :any:`Design` (possibly of a given type).
        """
        if mod_type is None:
            mods_5p = {strand.modification_5p for strand in self.strands if
                       strand.modification_5p is not None}
            mods_3p = {strand.modification_3p for strand in self.strands if
                       strand.modification_3p is not None}
            mods_int = {mod for strand in self.strands for mod in strand.modifications_int.values()}

            all_mods = mods_5p | mods_3p | mods_int

        elif mod_type is dm.ModificationType.five_prime:
            all_mods = {strand.modification_5p for strand in self.strands if
                        strand.modification_5p is not None}

        elif mod_type is dm.ModificationType.three_prime:
            all_mods = {strand.modification_3p for strand in self.strands if
                        strand.modification_3p is not None}

        elif mod_type is dm.ModificationType.internal:
            all_mods = {mod for strand in self.strands for mod in strand.modifications_int.values()}

        else:
            raise AssertionError('should be unreachable')

        self._ensure_mods_unique_names(all_mods)

        return all_mods

    @staticmethod
    def _ensure_mods_unique_names(all_mods: Set[dm.Modification]) -> None:
        mods_dict = {}
        for mod in all_mods:
            if mod.id not in mods_dict:
                mods_dict[mod.id] = mod
            else:
                other_mod = mods_dict[mod.id]
                raise ValueError(f'two different modifications share the id {mod.id}; '
                                 f'one is\n  {mod}\nand the other is\n  {other_mod}')

    def to_idt_bulk_input_format(self,
                                 delimiter: str = ',',
                                 key: Optional[KeyFunction[Strand]] = None,
                                 warn_duplicate_name: bool = False,
                                 only_strands_with_idt: bool = False,
                                 strands: Optional[Iterable[Strand]] = None) -> str:
        """Called by :meth:`Design.write_idt_bulk_input_file` to determine what string to write to
        the file. This function can be used to get the string directly without creating a file.

        Parameters have the same meaning as in :meth:`Design.write_idt_bulk_input_file`.

        :return:
            string that is written to the file in the method :meth:`Design.write_idt_bulk_input_file`.
        """
        if strands is None:
            strands = self.strands
        sc_design = _export_dummy_scadnano_design_for_idt_export(strands)
        return sc_design.to_idt_bulk_input_format(delimiter, key, warn_duplicate_name, only_strands_with_idt)

    def write_idt_bulk_input_file(self, *, directory: str = '.', filename: str = None,
                                  key: Optional[KeyFunction[Strand]] = None,
                                  extension: Optional[str] = None,
                                  delimiter: str = ',',
                                  warn_duplicate_name: bool = True,
                                  only_strands_with_idt: bool = False,
                                  strands: Optional[Iterable[Strand]] = None) -> None:
        """Write ``.idt`` text file encoding the strands of this :any:`Design` with the field
        :any:`Strand.idt`, suitable for pasting into the "Bulk Input" field of IDT
        (Integrated DNA Technologies, Coralville, IA, https://www.idtdna.com/),
        with the output file having the same name as the running script but with ``.py`` changed to ``.idt``,
        unless `filename` is explicitly specified.
        For instance, if the script is named ``my_origami.py``,
        then the sequences will be written to ``my_origami.idt``.
        If `filename` is not specified but `extension` is, then that extension is used instead of ``idt``.
        At least one of `filename` or `extension` must be ``None``.

        The string written is that returned by :meth:`Design.to_idt_bulk_input_format`.

        :param directory:
            specifies a directory in which to place the file, either absolute or relative to
            the current working directory. Default is the current working directory.
        :param filename:
            optional custom filename to use (instead of currently running script)
        :param key:
            `key function <https://docs.python.org/3/howto/sorting.html#key-functions>`_ used to determine
            order in which to output strand sequences. Some useful defaults are provided by
            :meth:`strand_order_key_function`
        :param extension:
            alternate filename extension to use (instead of idt)
        :param delimiter:
            is the symbol to delimit the four IDT fields name,sequence,scale,purification.
        :param warn_duplicate_name:
            if ``True`` prints a warning when two different :any:`Strand`'s have the same
            :data:`IDTFields.name` and the same :meth:`Strand.sequence`. A ValueError
            is raised (regardless of the value of this parameter)
            if two different :any:`Strand`'s have the same name but different sequences, IDT scales, or IDT
            purifications.
        :param only_strands_with_idt:
            If False (the default), all non-scaffold sequences are output, with reasonable default values
            chosen if the field :data:`Strand.idt` is missing.
            If True, then strands lacking the field :data:`Strand.idt` will not be exported.
        :param strands:
            strands to export; if not specified, all strands in design are exported.
            NOTE: it is not checked that each :any:`Strand` in `strands` is actually contained in this
            any:`Design`
        """
        contents = self.to_idt_bulk_input_format(delimiter=delimiter,
                                                 key=key,
                                                 warn_duplicate_name=warn_duplicate_name,
                                                 only_strands_with_idt=only_strands_with_idt,
                                                 strands=strands)
        if extension is None:
            extension = 'idt'
        sc.write_file_same_name_as_running_python_script(contents, extension, directory, filename)

    def write_idt_plate_excel_file(self, *, directory: str = '.', filename: str = None,
                                   key: Optional[KeyFunction[Strand]] = None,
                                   warn_duplicate_name: bool = False,
                                   only_strands_with_idt: bool = False,
                                   use_default_plates: bool = True, warn_using_default_plates: bool = True,
                                   plate_type: PlateType = PlateType.wells96,
                                   strands: Optional[Iterable[Strand]] = None) -> None:
        """
        Write ``.xls`` (Microsoft Excel) file encoding the strands of this :any:`Design` with the field
        :py:data:`Strand.idt`, suitable for uploading to IDT
        (Integrated DNA Technologies, Coralville, IA, https://www.idtdna.com/)
        to describe a 96-well or 384-well plate
        (https://www.idtdna.com/site/order/plate/index/dna/),
        with the output file having the same name as the running script but with ``.py`` changed to ``.xls``,
        unless `filename` is explicitly specified.
        For instance, if the script is named ``my_origami.py``,
        then the sequences will be written to ``my_origami.xls``.

        If the last plate as fewer than 24 strands for a 96-well plate, or fewer than 96 strands for a
        384-well plate, then the last two plates are rebalanced to ensure that each plate has at least
        that number of strands, because IDT charges extra for a plate with too few strands:
        https://www.idtdna.com/pages/products/custom-dna-rna/dna-oligos/custom-dna-oligos

        :param directory:
            specifies a directory in which to place the file, either absolute or relative to
            the current working directory. Default is the current working directory.
        :param filename:
            custom filename if default (explained above) is not desired
        :param key:
            `key function <https://docs.python.org/3/howto/sorting.html#key-functions>`_ used to determine
            order in which to output strand sequences. Some useful defaults are provided by
            :meth:`strand_order_key_function`
        :param warn_duplicate_name:
            if ``True`` prints a warning when two different :any:`Strand`'s have the same
            :data:`IDTFields.name` and the same :meth:`Strand.sequence`. A ValueError is
            raised (regardless of the value of this parameter)
            if two different :any:`Strand`'s have the same name but different sequences, IDT scales, or IDT
            purifications.
        :param only_strands_with_idt:
            If False (the default), all non-scaffold sequences are output, with reasonable default values
            chosen if the field :data:`Strand.idt` is missing.
            (though scaffold is included if `export_scaffold` is True).
            If True, then strands lacking the field :data:`Strand.idt` will not be exported.
            If False, then `use_default_plates` must be True.
        :param use_default_plates:
            Use default values for plate and well (ignoring those in idt fields, which may be None).
            If False, each Strand to export must have the field :data:`Strand.idt`, so in particular
            the parameter `only_strands_with_idt` must be True.
        :param warn_using_default_plates:
            specifies whether, if `use_default_plates` is True, to print a warning for strands whose
            :data:`Strand.idt` has the fields :py:data:`IDTFields.plate` and :py:data:`IDTFields.well`,
            since `use_default_plates` directs these fields to be ignored.
        :param plate_type:
            a :any:`PlateType` specifying whether to use a 96-well plate or a 384-well plate
            if the `use_default_plates` parameter is ``True``.
            Ignored if `use_default_plates` is ``False``, because in that case the wells are explicitly set
            by the user, who is free to use coordinates for either plate type.
        :param strands:
            strands to export; if not specified, all strands in design are exported.
            NOTE: it is not checked that each :any:`Strand` in `strands` is actually contained in this
            any:`Design`
        """
        if strands is None:
            strands = self.strands
        sc_design = _export_dummy_scadnano_design_for_idt_export(strands)
        sc_design.write_idt_plate_excel_file(directory=directory,
                                             filename=filename,
                                             key=key,
                                             warn_duplicate_name=warn_duplicate_name,
                                             only_strands_with_idt=only_strands_with_idt,
                                             use_default_plates=use_default_plates,
                                             warn_using_default_plates=warn_using_default_plates,
                                             plate_type=plate_type)

    def store_domain_pools(self) -> None:
        self.domain_pools_to_domain_map = defaultdict(list)
        for domain in self.domains:
            if domain._pool is not None:  # noqa
                self.domain_pools_to_domain_map[domain.pool].append(domain)

    def domain_pools(self) -> List[DomainPool]:
        """
        :return:
            list of all :any:`DomainPool`'s in this :any:`Design`
        """
        return list(self.domain_pools_to_domain_map.keys())

    def domains_by_pool_name(self, domain_pool_name: str) -> List[Domain[DomainLabel]]:
        """
        :param domain_pool_name: name of a :any:`DomainPool`
        :return: the :any:`Domain`'s in `domain_pool`
        """
        domains_in_pool: List[Domain] = []
        for domain in self.domains:
            if domain.pool.name == domain_pool_name:
                domains_in_pool.append(domain)
        return domains_in_pool

    @staticmethod
    def from_scadnano_file(sc_filename: str,
                           fix_assigned_sequences: bool = True,
                           ignored_strands: Optional[Iterable] = None) -> Design[StrandLabel, DomainLabel]:
        """
        Converts a scadnano Design stored in file named `sc_filename` to a a :any:`Design` for doing
        DNA sequence design.
        Each Strand name and Domain name from the scadnano Design are assigned as
        :py:data:`Strand.name` and :py:data:`Domain.name` in the obvious way.
        Assumes each Strand label is a string describing the strand group.

        The scadnano package must be importable.

        Also assigns sequences from domains in sc_design to those of the returned :any:`Design`.
        If `fix_assigned_sequences` is true, then these DNA sequences are fixed; otherwise not.

        :param sc_filename:
            Name of file containing scadnano Design.
        :param fix_assigned_sequences:
            Whether to fix the sequences that are assigned from those found in `sc_design`.
        :param ignored_strands:
            Strands to ignore
        :return:
            An equivalent :any:`Design`, ready to be given constraints for DNA sequence design.
        :raises TypeError:
            If any scadnano strand label is not a string.
        """
        sc_design = sc.Design.from_scadnano_file(sc_filename)
        return Design.from_scadnano_design(sc_design, fix_assigned_sequences, ignored_strands)

    @staticmethod
    def from_scadnano_design(sc_design: sc.Design[StrandLabel, DomainLabel],
                             fix_assigned_sequences: bool = True,
                             ignored_strands: Optional[Iterable] = None,
                             warn_existing_domain_labels: bool = True) -> Design[StrandLabel, DomainLabel]:
        """
        Converts a scadnano Design `sc_design` to a a :any:`Design` for doing DNA sequence design.
        Each Strand name and Domain name from the scadnano Design are assigned as
        :py:data:`Strand.name` and :py:data:`Domain.name` in the obvious way.
        Assumes each Strand label is a string describing the strand group.

        The scadnano package must be importable.

        Also assigns sequences from domains in sc_design to those of the returned :any:`Design`.
        If `fix_assigned_sequences` is true, then these DNA sequences are fixed; otherwise not.

        :param sc_design:
            Instance of scadnano.Design from the scadnano Python scripting library.
        :param fix_assigned_sequences:
            Whether to fix the sequences that are assigned from those found in `sc_design`.
        :param ignored_strands:
            Strands to ignore; none are ignore if not specified.
        :param warn_existing_domain_labels:
            If True, logs warning when dsd :any:`Domain` already has a label and so does scadnano domain,
            since scadnano label will not be assigned to the dsd :any:`Domain`.
        :return:
            An equivalent :any:`Design`, ready to be given constraints for DNA sequence design.
        :raises TypeError:
            If any scadnano strand label is not a string.
        """

        # check types
        if not isinstance(sc_design, sc.Design):
            raise TypeError(f'sc_design must be an instance of scadnano.Design, but it is {type(sc_design)}')
        if ignored_strands is not None:
            for ignored_strand in ignored_strands:
                if not isinstance(ignored_strand, sc.Strand):
                    raise TypeError('each ignored strand must be an instance of scadnano.Strand, but one is '
                                    f'{type(ignored_strand)}: {ignored_strand}')

        # filter out ignored strands
        strands_to_include = [strand for strand in sc_design.strands if strand not in ignored_strands] \
            if ignored_strands is not None else sc_design.strands

        # warn if not labels are dicts containing group_name_key on strands
        for sc_strand in strands_to_include:
            if (isinstance(sc_strand.label, dict) and group_key not in sc_strand.label) or \
                    (not isinstance(sc_strand.label, dict) and not hasattr(sc_strand.label, group_key)):
                logger.warning(f'Strand label {sc_strand.label} should be an object with attribute named '
                               f'"{group_key}" (for instance a dict or namedtuple).\n'
                               f'  The label is type {type(sc_strand.label)}. '
                               f'In order to auto-populate StrandGroups, ensure the label has attribute '
                               f'named "{group_key}" with associated value of type str.')
            else:
                label_value = Design.get_group_name_from_strand_label(sc_strand)
                if not isinstance(label_value, str):
                    logger.warning(f'Strand label {sc_strand.label} has attribute named '
                                   f'"{group_key}", but its associated value is not a string.\n'
                                   f'The value is type {type(label_value)}. '
                                   f'In order to auto-populate StrandGroups, ensure the label has attribute '
                                   f'named "{group_key}" with associated value of type str.')

                # raise TypeError(f'strand label {sc_strand.label} must be a dict, '
                #                 f'but instead is type {type(sc_strand.label)}')

        # groups scadnano strands by strand labels
        sc_strand_groups: DefaultDict[str, List[sc.Strand]] = defaultdict(list)
        for sc_strand in strands_to_include:
            assigned = False
            if hasattr(sc_strand.label, group_key) or (
                    isinstance(sc_strand.label, dict) and group_key in sc_strand.label):
                group = Design.get_group_name_from_strand_label(sc_strand)
                if isinstance(group, str):
                    sc_strand_groups[group].append(sc_strand)
                    assigned = True
            if not assigned:
                sc_strand_groups[default_strand_group].append(sc_strand)

        # make dsd StrandGroups, taking names from Strands and Domains,
        # and assign (and maybe fix) DNA sequences
        strand_names: Set[str] = set()
        design: Design[StrandLabel, DomainLabel] = Design()
        for group, sc_strands in sc_strand_groups.items():
            for sc_strand in sc_strands:
                # do not include strands with the same name more than once
                if sc_strand.name in strand_names:
                    logger.debug('In scadnano design, found duplicate instance of strand with name '
                                 f'{sc_strand.name}; skipping all but the first when creating dsd design. '
                                 f'Please ensure that this strand really is supposed to have the same name.')
                    continue

                domain_names: List[str] = [domain.name for domain in sc_strand.domains]
                sequence = sc_strand.dna_sequence
                nuad_strand: Strand[StrandLabel, DomainLabel] = design.add_strand(domain_names=domain_names,
                                                                                  group=group,
                                                                                  name=sc_strand.name,
                                                                                  label=sc_strand.label)
                # assign sequence
                if sequence is not None:
                    for dsd_domain, sc_domain in zip(nuad_strand.domains, sc_strand.domains):
                        domain_sequence = sc_domain.dna_sequence
                        # if this is a starred domain,
                        # take the WC complement first so the dsd Domain stores the "canonical" sequence
                        if sc_domain.name[-1] == '*':
                            domain_sequence = dv.wc(domain_sequence)
                        if sc.DNA_base_wildcard not in domain_sequence:
                            if fix_assigned_sequences:
                                dsd_domain.set_fixed_sequence(domain_sequence)
                            else:
                                dsd_domain.set_sequence(domain_sequence)

                # set domain labels
                for dsd_domain, sc_domain in zip(nuad_strand.domains, sc_strand.domains):
                    if dsd_domain.label is None:
                        dsd_domain.label = sc_domain.label
                    elif sc_domain.label is not None and warn_existing_domain_labels:
                        logger.warning(f'warning; dsd domain already has label {dsd_domain.label}; '
                                       f'skipping assignment of scadnano label {sc_domain.label}')

                strand_names.add(nuad_strand.name)

        design.compute_derived_fields()

        return design

    @staticmethod
    def get_group_name_from_strand_label(sc_strand: Strand) -> Any:
        if hasattr(sc_strand.label, group_key):
            return getattr(sc_strand.label, group_key)
        elif isinstance(sc_strand.label, dict) and group_key in sc_strand.label:
            return sc_strand.label[group_key]
        else:
            raise AssertionError(f'label does not have either an attribute or a dict key "{group_key}"')

    def assign_sequences_to_scadnano_design(self, sc_design: sc.Design[StrandLabel, DomainLabel],
                                            ignored_strands: Iterable[Strand] = ()) -> None:
        """
        Assigns sequences from this :any:`Design` into `sc_design`.

        Assumes that each domain name in domains in `sc_design` is a :py:data:`Domain.name` of a
        :any:`Domain` in this :any:`Design`.

        If multiple strands in `sc_design` share the same name, then all of them are assigned the
        DNA sequence of the dsd :any:`Strand` with that name.

        :param sc_design:
            a scadnano design.
        :param ignored_strands:
            strands in the scadnano design that are to be ignored by the sequence designer.
        """

        # filter out ignored strands
        sc_strands_to_include = [strand for strand in sc_design.strands if strand not in ignored_strands]

        # check types
        if not isinstance(sc_design, sc.Design):
            raise TypeError(f'sc_design must be an instance of scadnano.Design, but it is {type(sc_design)}')

        # dict mapping tuples of domain names to strands that have those domains in that order
        # sc_domain_name_tuples = {strand.domain_names_tuple(): strand for strand in self.strands}
        sc_domain_name_tuples: Dict[Tuple[str, ...], Strand] = {}
        for strand in self.strands:
            domain_names_tuple = strand.domain_names_tuple()
            sc_domain_name_tuples[domain_names_tuple] = strand

        for sc_strand in sc_strands_to_include:
            domain_names = [domain.name for domain in sc_strand.domains]
            if sc_strand.dna_sequence is None:
                assert None not in domain_names
                self._assign_to_strand_with_no_sequence(sc_strand, sc_design)
            elif None not in domain_names:
                self._assign_to_strand_with_partial_sequence(sc_strand, sc_design, sc_domain_name_tuples)
            else:
                logger.warning('Skipping assignment of DNA sequence to scadnano strand with sequence '
                               f'{sc_strand.dna_sequence}, since it has at least one domain name '
                               f'that is None.\n'
                               f'Make sure that this is a strand you intended to leave out of the '
                               f'sequence design process')

    def assign_idt_fields_to_scadnano_design(self, sc_design: sc.Design[StrandLabel, DomainLabel],
                                             ignored_strands: Iterable[Strand] = ()) -> None:
        """
        Assigns :any:`IDTFields` from this :any:`Design` into `sc_design`.

        If multiple strands in `sc_design` share the same name, then all of them are assigned the
        IDT fields of the dsd :any:`Strand` with that name.

        :param sc_design:
            a scadnano design.
        :param ignored_strands:
            strands in the scadnano design that are to be not assigned.
        :raises ValueError:
            if scadnano strand already has any modifications assigned
        """
        # filter out ignored strands
        sc_strands_to_include = [strand for strand in sc_design.strands if strand not in ignored_strands]

        nuad_strands_by_name = {strand.name: strand for strand in self.strands}

        for sc_strand in sc_strands_to_include:
            nuad_strand = nuad_strands_by_name[sc_strand.name]
            if nuad_strand.idt is not None:
                if sc_strand.idt is not None:
                    raise ValueError(f'Cannot assign IDT fields from dsd strand to scadnano strand '
                                     f'{sc_strand.name} because the scadnano strand already has IDT fields '
                                     f'assigned:\n{sc_strand.idt}')
                sc_strand.idt = nuad_strand.idt.to_scadnano_idt()

    def assign_modifications_to_scadnano_design(self, sc_design: sc.Design[StrandLabel, DomainLabel],
                                                ignored_strands: Iterable[Strand] = ()) -> None:
        """
        Assigns :any:`modifications.Modification`'s from this :any:`Design` into `sc_design`.

        If multiple strands in `sc_design` share the same name, then all of them are assigned the
        modifications of the dsd :any:`Strand` with that name.

        :param sc_design:
            a scadnano design.
        :param ignored_strands:
            strands in the scadnano design that are to be not assigned.
        :raises ValueError:
            if scadnano strand already has any modifications assigned
        """
        print('WARNING: the method assign_modifications_to_scadnano_design has not been tested yet '
              'and may have errors')
        # filter out ignored strands
        sc_strands_to_include = [strand for strand in sc_design.strands if strand not in ignored_strands]

        dsd_strands_by_name = {strand.name: strand for strand in self.strands}
        for sc_strand in sc_strands_to_include:
            dsd_strand: Strand = dsd_strands_by_name[sc_strand.name]
            if dsd_strand.modification_5p is not None:
                if sc_strand.modification_5p is not None:
                    raise ValueError(f'Cannot assign 5\' modification from dsd strand to scadnano strand '
                                     f'{sc_strand.name} because the scadnano strand already has a 5\''
                                     f'modification assigned:\n{sc_strand.modification_5p}')
                sc_strand.modification_5p = dsd_strand.modification_5p.to_scadnano_modification()

            if dsd_strand.modification_3p is not None:
                if sc_strand.modification_3p is not None:
                    raise ValueError(f'Cannot assign 3\' modification from dsd strand to scadnano strand '
                                     f'{sc_strand.name} because the scadnano strand already has a 3\''
                                     f'modification assigned:\n{sc_strand.modification_3p}')
                sc_strand.modification_3p = dsd_strand.modification_3p.to_scadnano_modification()

            for offset, mod_int in dsd_strand.modifications_int.items():
                if offset in sc_strand.modifications_int is not None:
                    raise ValueError(f'Cannot assign internal modification from dsd strand to '
                                     f'scadnano strand {sc_strand.name} at offset {offset} '
                                     f'because the scadnano strand already has an internal '
                                     f'modification assigned at that offset:\n'
                                     f'{sc_strand.modifications_int[offset]}')
                sc_strand.modifications_int[offset] = mod_int.to_scadnano_modification()

    def _assign_to_strand_with_no_sequence(self,
                                           sc_strand: sc.Strand[StrandLabel, DomainLabel],
                                           sc_design: sc.Design[StrandLabel, DomainLabel]) -> None:
        # check types
        if not isinstance(sc_design, sc.Design):
            raise TypeError(f'sc_design must be an instance of scadnano.Design, but it is {type(sc_design)}')
        if not isinstance(sc_strand, sc.Strand):
            raise TypeError(f'sc_strand must be an instance of scadnano.Strand, but it is {type(sc_strand)}')

        sequence_list: List[str] = []
        for sc_domain in sc_strand.domains:
            domain_name = sc_domain.name
            if domain_name is None:
                raise AssertionError('did not expect domain_name to be None')
            starred = domain_name[-1] == '*'
            if starred:
                domain_name = domain_name[:-1]
            dsd_domain = self.domains_by_name.get(domain_name)
            if dsd_domain is None:
                raise AssertionError(f'expected domain_name {domain_name} to be a key in domains_by_name '
                                     f'{list(self.domains_by_name.keys())}')
            domain_sequence = dsd_domain.concrete_sequence(starred)
            sequence_list.append(domain_sequence)
        strand_sequence = ''.join(sequence_list)
        sc_design.assign_dna(strand=sc_strand, sequence=strand_sequence, assign_complement=False,
                             check_length=True)

    @staticmethod
    def _assign_to_strand_with_partial_sequence(sc_strand: sc.Strand[StrandLabel, DomainLabel],
                                                sc_design: sc.Design[StrandLabel, DomainLabel],
                                                sc_domain_name_tuples: Dict[Tuple[str, ...], Strand]) -> None:

        # check types
        if not isinstance(sc_design, sc.Design):
            raise TypeError(f'sc_design must be an instance of scadnano.Design, but it is {type(sc_design)}')
        if not isinstance(sc_strand, sc.Strand):
            raise TypeError(f'sc_strand must be an instance of scadnano.Strand, but it is {type(sc_strand)}')

        # sigh: we don't have a great way to track which strand in sc_design corresponds to the same
        # strand in dsd_design (self), so we collect list of domain names in sc_strand and see if there's
        # a strand in dsd_design with the same domain names in the same order. If not we assume the strand
        # was not part of dsd_design
        domain_name_list: List[str] = []
        for sc_domain in sc_strand.domains:
            domain_name = sc_domain.name
            if domain_name is None:
                raise AssertionError('did not expect domain_name to be None')
            domain_name_list.append(domain_name)

        domain_names = tuple(domain_name_list)
        dsd_strand = sc_domain_name_tuples.get(domain_names)
        if dsd_strand is None:
            logger.warning('Skipping assignment of DNA sequence to scadnano strand with domains '
                           f'{"-".join(domain_names)}.\n'
                           f'Make sure that this is a strand you intended to leave out of the '
                           f'sequence design process')
            return

        wildcard: str = sc.DNA_base_wildcard

        sequence_list: List[str] = []
        for sc_domain, dsd_domain, domain_name in zip(sc_strand.domains, dsd_strand.domains, domain_names):
            starred = domain_name[-1] == '*'
            sc_domain_sequence = sc_domain.dna_sequence

            # if we're in this method, then domains of sc_strand should have a partial assignment
            assert sc_domain_sequence is not None
            # now we detect whether this domain was assigned or not
            if wildcard in sc_domain_sequence:
                # if there are any '?' wildcards, then all of them should be wildcards
                assert sc_domain_sequence == wildcard * len(sc_domain_sequence)
                # if not assigned in sc_strand, we assign from dsd
                domain_sequence = dsd_domain.concrete_sequence(starred)
            else:
                # otherwise we stick with the sequence that was already assigned in sc_domain
                domain_sequence = sc_domain_sequence
                # but let's make sure dsd didn't actually change that sequence; it should have been fixed
                dsd_domain_sequence = dsd_domain.concrete_sequence(starred)
                if domain_sequence != dsd_domain_sequence:
                    raise AssertionError(f'\n    domain_sequence = {domain_sequence} is unequal to\n'
                                         f'dsd_domain_sequence = {dsd_domain_sequence}')
            sequence_list.append(domain_sequence)
        strand_sequence = ''.join(sequence_list)
        sc_design.assign_dna(strand=sc_strand, sequence=strand_sequence, assign_complement=False,
                             check_length=True)

    def copy_sequences_from(self, other: Design) -> None:
        """
        Assuming every :any:`Domain` in this :any:`Design` is has a matching (same name) :any:`Domain` in
        `other`, copies sequences from `other` into this :any:`Design`.

        :param other:
            other :any:`Design` from which to copy sequences
        """
        # see if self.domains needs to be initialized
        computed_derived_fields = False
        if self.domains is None:
            self.compute_derived_fields()
            computed_derived_fields = True

        # copy sequences
        for domain in self.domains:
            other_domain = other.domains_by_name[domain.name]
            if other_domain.fixed:
                domain.set_fixed_sequence(other_domain.sequence())
            elif other_domain.has_sequence():
                domain.set_sequence(other_domain.sequence())

        # no need to compute_derived_fields if we already called it above,
        # since new sequences won't change derived fields
        if not computed_derived_fields:
            self.compute_derived_fields()

    def check_all_subdomain_graphs_acyclic(self) -> None:
        """
        Check that all domain graphs (if subdomains are used) are acyclic.
        """
        for strand in self.strands:
            # Check that each base in the sequence is assigned by exactly one
            # independent subdomain.
            for d in cast(List[Domain], strand.domains):
                d._check_acyclic_subdomain_graph()  # noqa

    def check_all_subdomain_graphs_uniquely_assignable(self) -> None:
        """
        Check that subdomain graphs are consistent and raise error if not.
        """
        for strand in self.strands:
            # Check that each base in the sequence is assigned by exactly one
            # independent subdomain.
            for d in cast(List[Domain], strand.domains):
                d._check_acyclic_subdomain_graph()  # noqa
                d._check_subdomain_graph_is_uniquely_assignable()  # noqa

    def check_names_unique(self) -> None:
        # domain names already checked in compute_derived_fields()
        self.check_strand_names_unique()
        self.check_domain_pool_names_unique()

    def check_strand_names_unique(self) -> None:
        strands_by_name = {}
        for strand in self.strands:
            name = strand.name
            if name in strands_by_name:
                raise ValueError(f'found two strands with name {name}:\n'
                                 f'  {strand}\n'
                                 f'and\n'
                                 f'  {strands_by_name[name]}')

    def check_domain_pool_names_unique(self) -> None:
        # self.domain_pools() already computed by compute_derived_fields()
        domain_pools_by_name = {}
        for pool in self.domain_pools():
            name = pool.name
            if name in domain_pools_by_name:
                raise ValueError(f'found two DomainPools with name {name}:\n'
                                 f'  {pool}\n'
                                 f'and\n'
                                 f'  {domain_pools_by_name[name]}')
            else:
                domain_pools_by_name[pool.name] = pool


# represents a "Design Part", e.g., Strand, Tuple[Domain, Domain], etc... whatever portion of the Design
# is checked by the constraint
# NOTE: this is needed in addition to the abstract base class Part, because it allows mypy type checking
# of the various different types of evaluate and evaluate_bulk functions. Otherwise they have more
# abstract type signatures, and we can't write something like evaluate(strand: Strand)
# Maybe if we eventually get rid of the parts and only pass in the sequences, this will not be needed.
DesignPart = TypeVar('DesignPart',
                     Domain,
                     Strand,
                     DomainPair,
                     StrandPair,
                     Complex,
                     Design)


@dataclass(frozen=True, eq=False)
class Constraint(Generic[DesignPart], ABC):
    """
    Abstract base class of all "soft" constraints to apply when running
    :meth:`search.search_for_dna_sequences`.
    Unlike a :any:`NumpyConstraint` or a :any:`SequenceConstraint`, which disallow certain DNA sequences
    from ever being assigned to a :any:`Domain`, a :any:`Constraint` can be violated during the search.
    The goal of the search is to reduce the number of violated :any:`Constraint`'s.
    See :meth:`search.search_for_dna_sequences` for a more detailed description of how the search algorithm
    interacts with the constraints.

    You will not use this class directly, but instead its concrete subclasses
    :any:`DomainConstraint`,
    :any:`StrandConstraint`,
    :any:`DomainPairConstraint`,
    :any:`StrandPairConstraint`,
    :any:`ComplexConstraint`,
    which are subclasses of :any:`SingularConstraint`,
    :any:`DomainsConstraint`,
    :any:`StrandsConstraint`,
    :any:`DomainPairsConstraint`,
    :any:`StrandPairsConstraint`,
    which are subclasses of :any:`BulkConstraint`,
    or
    :any:`DesignConstraint`.
    """

    description: str
    """
    Description of the constraint, e.g., 'strand has secondary structure exceeding -2.0 kcal/mol'.
    """

    short_description: str = ''
    """
    Very short description of the constraint suitable for compactly logging to the screen, e.g., 'strand_ss'
    """

    weight: float = 1.0
    """
    Constant multiplier Weight of the problem; the higher the total weight of all the :any:`Constraint`'s 
    a :any:`Domain` has caused, the greater likelihood its sequence is changed when stochastically searching 
    for sequences to satisfy all constraints.
    """

    score_transfer_function: Callable[[float], float] = default_score_transfer_function
    """
    Score transfer function to use. When a constraint is violated, the constraint returns a nonnegative
    float (the score) indicating the "severity" of the violation. For example, if a :any:`Strand` has 
    secondary structure energy exceeding a threshold, the score returned is the difference between 
    the energy and the threshold.
    
    The score is then passed through the `score_transfer_function`.
    The default is the squared ReLU function: f(x) = max(0, x^2).
    This "punishes" more severe violations more, for example, it would
    bring down the total score of violations more to reduce a violation 3 kcal/mol in excess of its
    threshold to 2 kcal/mol excess,
    than to reduce a violation only 1 kcal/mol in excess of its threshold down to 0.
    """

    @staticmethod
    @abstractmethod
    def part_name() -> str:
        """
        Returns name of the :any:`Part` that this :any:`Constraint` tests.

        :return:
            name of the :any:`Part` that this :any:`Constraint` tests
            (e.g., "domain", "strand pair")
        """
        raise NotImplementedError()


def _raise_unreachable():
    raise AssertionError('This should be unreachable')


@dataclass(frozen=True, eq=False)
class SingularConstraint(Constraint[DesignPart], Generic[DesignPart], ABC):
    evaluate: Callable[[Tuple[str, ...], Optional[DesignPart]],
                       Tuple[float, str]] = lambda _: _raise_unreachable()
    """
    Essentially a wrapper for a function that evaluates the :any:`Constraint`. 
    It takes as input a tuple of DNA sequences 
    (Python strings) and an optional :any:`Part`, where :any:`Part` is one of 
    :any:`Domain`, :any:`Strand`, :any:`DomainPair`, :any:`StrandPair`, or :any:`Complex`
    (the latter being an alias for arbitrary-length tuple of :any:`Strand`'s).

    The second argument will be None if :data:`Constraint.parallel` is True (since it's more expensive
    to serialize the :any:`Domain` and :any:`Strand` objects than strings for passing data to processes
    executing in parallel). Thus, if the :any:`Constraint` needs to use more data about the :any:`Part`
    than just its DNA sequence, by accessing the second argument, :data:`Constraint.parallel` should be set
    to False.
    """

    parallel: bool = False
    """
    Whether or not to use parallelization across multiple processes to take advantage of multiple
    processors/cores, by calling :data:`SingularConstraint.evaluate` on different DesignParts
    in separate processes.
    """

    def __post_init__(self) -> None:
        if self.evaluate is None:
            raise ValueError(f'_evaluate must be set for a {self.__class__.__name__}')

        if len(self.short_description) == 0:
            # self.short_description = self.description
            object.__setattr__(self, 'short_description', self.description)

        if self.weight <= 0:
            raise ValueError(f'weight must be positive but it is {self.weight}')

    def call_evaluate(self, seqs: Tuple[str, ...], part: Optional[DesignPart]) -> Tuple[float, str]:
        """
        Evaluates this :any:`Constraint` using function :data:`SingularConstraint.evaluate`
        supplied in constructor.

        :param seqs:
            sequence(s) of relevant :any:`Part`, e.g., if `part` is a pair of :any:`Strand`'s,
            then `seqs` is a pair of strings
        :param part:
            the :any:`Part` to be evaluated. Might be None if parallelization is being used,
            since it is cheaper to serialize only the sequence(s) than the entire :any:`Part`
            for passing to other processes to evaluate in parallel.
        :return:
            pair (`excess`, `summary`), where `excess` is a float indicating how much the constraint
            was violated (0.0 if satisfied) and `summary` is a string summarizing the violation (or lack
            thereof), suitable for printing into a line of a report. For example, if measuring
            complex free energy -2.5 kcal/mol of a strand and comparing against a threshold -1.0 kcal/mol,
            `excess` might be the difference 1.5 between the energy and the threshold,
            and `summary` might be the string "-2.5 kcal/mol".
        """
        excess, summary = (self.evaluate)(seqs, part)  # noqa
        if excess < 0.0:
            excess = 0.0
        score = self.weight * self.score_transfer_function(excess)
        return score, summary


@dataclass(frozen=True, eq=False)
class BulkConstraint(Constraint[DesignPart], Generic[DesignPart], ABC):
    evaluate_bulk: Callable[[Sequence[DesignPart]],
                            List[Tuple[DesignPart, float, str]]] = lambda _: _raise_unreachable()

    def call_evaluate_bulk(self, parts: Sequence[DesignPart]) -> List[Tuple[DesignPart, float, str]]:
        results: List[Tuple[DesignPart, float, str]] = (self.evaluate_bulk)(parts)  # noqa
        results_with_scores_transferred_and_weighted: List[Tuple[DesignPart, float, str]] = []
        for part, excess, summary in results:
            if excess < 0.0:
                excess = 0.0
            score = self.weight * self.score_transfer_function(excess)
            results_with_scores_transferred_and_weighted.append((part, score, summary))
        return results_with_scores_transferred_and_weighted


_no_summary_string = "No summary for this constraint. " \
                     "To generate one, pass a function as the parameter named " \
                     '"summary" when creating the Constraint.'


@dataclass(frozen=True, eq=False)
class ConstraintWithDomains(Generic[DesignPart]):  # noqa
    domains: Optional[Tuple[Domain, ...]] = None
    """
    Tuple of :any:`Domain`'s to check; if not specified, all :any:`Domain`'s in :any:`Design` are checked.
    """


@dataclass(frozen=True, eq=False)
class ConstraintWithStrands(Generic[DesignPart]):  # noqa
    strands: Optional[Tuple[Strand, ...]] = None
    """
    Tuple of :any:`Strand`'s to check; if not specified, all :any:`Strand`'s in :any:`Design` are checked.
    """


@dataclass(frozen=True, eq=False)  # type: ignore
class DomainConstraint(ConstraintWithDomains[Domain], SingularConstraint[Domain]):
    """Constraint that applies to a single :any:`Domain`."""

    def __post_init__(self) -> None:
        if self.evaluate is None:
            raise ValueError('_evaluate must be specified for a DomainConstraint')
        super().__post_init__()

    @staticmethod
    def part_name() -> str:
        return 'domain'


@dataclass(frozen=True, eq=False)  # type: ignore
class StrandConstraint(ConstraintWithStrands[Strand], SingularConstraint[Strand]):
    """Constraint that applies to a single :any:`Strand`."""

    def __post_init__(self) -> None:
        if self.evaluate is None:
            raise ValueError('_evaluate must be specified for a StrandConstraint')
        super().__post_init__()

    @staticmethod
    def part_name() -> str:
        return 'strand'


@dataclass(frozen=True, eq=False)
class ConstraintWithDomainPairs(Constraint[DesignPart], Generic[DesignPart]):  # noqa
    pairs: Optional[Tuple[Tuple[Domain, Domain], ...]] = None
    """
    List of pairs of :any:`Domain`'s to check; if not specified, all pairs in :any:`Design` are checked.
    """

    check_domain_against_itself: bool = True
    """
    Whether to check a domain against itself when checking all pairs of :any:`Domain`'s in the :any:`Design`. 
    Only used if :data:`ConstraintWithDomainPairs.pairs` is not specified, otherwise it is ignored.
    """


@dataclass(frozen=True, eq=False)
class ConstraintWithStrandPairs(Constraint[DesignPart], Generic[DesignPart]):  # noqa
    pairs: Optional[Tuple[Tuple[Strand, Strand], ...]] = None
    """
    List of pairs of :any:`Strand`'s to check; if not specified, all pairs in :any:`Design` are checked.
    """

    check_strand_against_itself: bool = True
    """
    Whether to check a strand against itself when checking all pairs of :any:`Strand`'s in the :any:`Design`. 
    Only used if :data:`ConstraintWithStrandPairs.pairs` is not specified, otherwise it is ignored.
    """


@dataclass(frozen=True, eq=False)  # type: ignore
class DomainPairConstraint(ConstraintWithDomainPairs[DomainPair],
                           SingularConstraint[DomainPair]):
    """Constraint that applies to a pair of :any:`Domain`'s.

    These should be symmetric, meaning that the constraint will give the same evaluation whether its
    evaluate method is given the pair (domain1, domain2), or the pair (domain2, domain1)."""

    @staticmethod
    def part_name() -> str:
        return 'domain pair'


@dataclass(frozen=True, eq=False)  # type: ignore
class StrandPairConstraint(ConstraintWithStrandPairs[StrandPair],
                           SingularConstraint[StrandPair]):
    """Constraint that applies to a pair of :any:`Strand`'s.

    These should be symmetric, meaning that the constraint will give the same evaluation whether its
    evaluate method is given the pair (strand1, strand2), or the pair (strand2, strand1)."""

    @staticmethod
    def part_name() -> str:
        return 'strand pair'


@dataclass(frozen=True, eq=False)  # type: ignore
class DomainsConstraint(ConstraintWithDomains[Domain], BulkConstraint[Domain]):
    """
    Constraint that applies to a several :any:`Domain`'s.

    The difference with :any:`DomainConstraint` is that
    the caller may want to process all :any:`Domain`'s at once, e.g., by giving many of them to a third-party
    program such as ViennaRNA, which may be more efficient than repeatedly calling a Python function.

    It *is* assumed that the constraint works by checking one :any:`Domain` at a time. After computing
    initial violations of constraints, subsequent calls to this constraint only give the domain that was
    mutated, not the entire of :any:`Domain`'s in the whole :any:`Design`.
    Use :any:`DesignConstraint` for constraints that require every :any:`Domain` in the :any:`Design`.
    """

    @staticmethod
    def part_name() -> str:
        return 'domain'


@dataclass(frozen=True, eq=False)  # type: ignore
class StrandsConstraint(ConstraintWithStrands[Strand], BulkConstraint[Strand]):
    """
    Constraint that applies to a several :any:`Strand`'s.

    The difference with :any:`StrandConstraint` is that
    the caller may want to process all :any:`Strand`'s at once, e.g., by giving many of them to a third-party
    program such as ViennaRNA.

    It *is* assumed that the constraint works by checking one :any:`Strand` at a time. After computing
    initial violations of constraints, subsequent calls to this constraint only give strands containing
    the domain that was mutated, not the entire of :any:`Strand`'s in the whole :any:`Design`.
    Use :any:`DesignConstraint` for constraints that require every :any:`Strand` in the :any:`Design`.
    """

    @staticmethod
    def part_name() -> str:
        return 'strand'


@dataclass(frozen=True, eq=False)  # type: ignore
class DomainPairsConstraint(ConstraintWithDomainPairs[DomainPair], BulkConstraint[DomainPair]):
    """
    Similar to :any:`DomainsConstraint` but operates on a specified list of pairs of :any:`Domain`'s.
    """

    @staticmethod
    def part_name() -> str:
        return 'domain pair'


@dataclass(frozen=True, eq=False)  # type: ignore
class StrandPairsConstraint(ConstraintWithStrandPairs[StrandPair], BulkConstraint[StrandPair]):
    """
    Similar to :any:`StrandsConstraint` but operates on a specified list of pairs of :any:`Strand`'s.
    """

    @staticmethod
    def part_name() -> str:
        return 'strand pair'


@dataclass(frozen=True, eq=False)  # type: ignore
class DesignConstraint(Constraint[Design]):
    """
    Constraint that applies to the entire :any:`Design`. This is used for any :any:`Constraint` that
    does not naturally fit the structure of the other types of constraints.

    Unlike other constraints, which specify either :data:`Constraint._evaluate` or
    :data:`Constraint._evaluate_bulk`, a :any:`DesignConstraint` leaves both of these unspecified and
    specifies :data:`DesignConstraint._evaluate_design` instead.
    """

    evaluate_design: Callable[[Design, Iterable[Domain]],
                              List[Tuple[DesignPart, float, str]]] = lambda _: _raise_unreachable()
    """
    Evaluates the :any:`Design` (first argument), possibly taking into account which :any:`Domain`'s have
    changed in the last iteration (second argument).
    
    Returns a list of tuples (`part`, `score`, `summary`), 
    one tuple per violation of the :any:`DesignConstraint`.
    
    `part` is the part of the :any:`Design` that caused the violation.
    It must be one of :any:`Domain`, :any:`Strand`, pair of `Domain`'s, or tuple of :any:`Strand`'s.
    
    `score` is the score of the violation.
    
    `summary` is a 1-line summary of the violation to put into the generated reports.
    """

    def __post_init__(self) -> None:
        if self.evaluate_design is None:
            raise ValueError('_evaluate_design should be specified in a DesignConstraint')

    def call_evaluate_design(self, design: Design, domains_changed: Iterable[Domain]) \
            -> List[Tuple[DesignPart, float, str]]:
        results = (self._evaluate_bulk)(design, domains_changed)  # noqa
        results_with_scores_transferred_and_weighted = []
        for part, excess, summary in results:
            if excess < 0.0:
                excess = 0.0
            weighted_score = self.weight * self.score_transfer_function(excess)
            results_with_scores_transferred_and_weighted.append((part, weighted_score, summary))
        return results_with_scores_transferred_and_weighted

    @staticmethod
    def part_name() -> str:
        return 'whole design'


def verify_designs_match(design1: Design, design2: Design, check_fixed: bool = True) -> None:
    """
    Verifies that two designs match, other than their constraints. This is useful when loading a
    design that has been saved in the middle of searching for DNA sequences, to verify that it matches
    a design created before the DNA sequence search started.

    :param design1:
        A :any:`Design`.
    :param design2:
        Another :any:`Design`.
    :param check_fixed:
        Whether to check for fixed sequences equal between the two (may want to not check in case these
        are set later).
    :raises ValueError:
        If the designs do not match.
        Here is what is checked:
        - strand names and group names appear in the same order
        - domain names and pool names appear in the same order in strands with the same name
        - :py:data:`Domain.fixed` matches between :any:`Domain`'s
    """
    for idx, (strand1, strand2) in enumerate(zip(design1.strands, design2.strands)):
        if strand1.name != strand2.name:
            raise ValueError(f'strand names at position {idx} don\'t match: '
                             f'{strand1.name} and {strand2.name}')
        if (strand1.group is not None
                and strand2.group is not None
                and strand1.group != strand2.group):  # noqa
            raise ValueError(f'strand {strand2.name} group name does not match:'
                             f'design1 strand {strand1.name} group = {strand1.group},\n'
                             f'design2 strand {strand2.name} group = {strand2.group}')
        for domain1, domain2 in zip(strand1.domains, strand2.domains):
            if domain1.name != domain2.name:
                raise ValueError(f'domain of strand {strand2.name} don\'t match: '
                                 f'{strand1.domains} and {strand2.domains}')
            if check_fixed and domain1.fixed != domain2.fixed:
                raise ValueError(f'domain {domain2.name} is fixed in one but not the other:\n'
                                 f'design1 domain {domain1.name} fixed = {domain1.fixed},\n'
                                 f'design2 domain {domain2.name} fixed = {domain2.fixed}')
            if (domain1.has_pool()
                    and domain2.has_pool()
                    and domain1.pool.name != domain2.pool.name):
                raise ValueError(f'domain {domain2.name} pool name does not match:'
                                 f'design1 domain {domain1.name} pool = {domain1.pool.name},\n'
                                 f'design2 domain {domain2.name} pool = {domain2.pool.name}')


def convert_threshold(threshold: Union[float, Dict[T, float]], key: T) -> float:
    """
    :param threshold: either a single float, or a dictionary mapping instances of T to floats
    :param key: instance of T
    :return: threshold for key
    """
    threshold_value: float
    if isinstance(threshold, float):
        threshold_value = threshold
    elif isinstance(threshold, dict):
        threshold_value = threshold[key]
    else:
        raise ValueError(f'threshold = {threshold} must be one of float or dict, '
                         f'but it is {type(threshold)}')
    return threshold_value


def _check_nupack_installed() -> None:
    """
     Raises ImportError if nupack module is not installed.
    """
    try:
        import nupack
    except ModuleNotFoundError:
        raise ImportError(
            'NUPACK 4 must be installed to create a constraint that uses NUPACK. '
            'Installation instructions can be found at '
            'https://github.com/UC-Davis-molecular-computing/dsd#installation and '
            'https://piercelab-caltech.github.io/nupack-docs/start/')


def nupack_domain_complex_free_energy_constraint(
        threshold: float,
        temperature: float = dv.default_temperature,
        sodium: float = dv.default_sodium,
        magnesium: float = dv.default_magnesium,
        weight: float = 1.0,
        score_transfer_function: Callable[[float], float] = default_score_transfer_function,
        parallel: bool = False,
        description: Optional[str] = None,
        short_description: str = 'strand_ss_nupack',
        domains: Optional[Iterable[Domain]] = None) -> DomainConstraint:
    """
    Returns constraint that checks individual :any:`Domain`'s for excessive interaction using
    NUPACK's pfunc.

    NUPACK 4 must be installed. Installation instructions can be found at
    https://piercelab-caltech.github.io/nupack-docs/start/.

    :param threshold:
        energy threshold in kcal/mol
    :param temperature:
        temperature in Celsius
    :param sodium:
        molarity of sodium (more generally, monovalent ions such as Na+, K+, NH4+)
        in moles per liter
    :param magnesium:
        molarity of magnesium (Mg++) in moles per liter
    :param weight:
        how much to weigh this :any:`Constraint`
    :param score_transfer_function:
        See :py:data:`Constraint.score_transfer_function`.
    :param parallel:
        Whether to use parallelization by running constraint evaluation in separate processes
        to take advantage of multiple cores.
    :param domains:
        :any:`Domain`'s to check; if not specified, all domains are checked.
    :param description:
        detailed description of constraint suitable for putting in report; if not specified
        a reasonable default is chosen
    :param short_description:
        short description of constraint suitable for logging to stdout
    :return:
        the constraint
    """
    _check_nupack_installed()

    def evaluate(seqs: Tuple[str], _: Optional[Domain]) -> Tuple[float, str]:
        sequence = seqs[0]
        energy = dv.complex_free_energy_single_strand(sequence, temperature, sodium, magnesium)
        excess = threshold - energy
        return max(0.0, excess), f'{energy:6.2f} kcal/mol'

    if description is None:
        description = f'NUPACK secondary structure of domain exceeds {threshold} kcal/mol'

    if domains is not None:
        domains = tuple(domains)

    return DomainConstraint(description=description,
                            short_description=short_description,
                            weight=weight,
                            score_transfer_function=score_transfer_function,
                            evaluate=evaluate,
                            parallel=parallel,
                            domains=domains)


def nupack_strand_complex_free_energy_constraint(
        threshold: float,
        temperature: float = dv.default_temperature,
        sodium: float = dv.default_sodium,
        magnesium: float = dv.default_magnesium,
        weight: float = 1.0,
        score_transfer_function: Callable[[float], float] = default_score_transfer_function,
        parallel: bool = False,
        description: Optional[str] = None,
        short_description: str = 'strand_ss_nupack',
        strands: Optional[Iterable[Strand]] = None) -> StrandConstraint:
    """
    Returns constraint that checks individual :any:`Strand`'s for excessive interaction using
    NUPACK's pfunc.

    NUPACK 4 must be installed. Installation instructions can be found at
    https://piercelab-caltech.github.io/nupack-docs/start/.

    :param threshold:
        energy threshold in kcal/mol
    :param temperature:
        temperature in Celsius
    :param sodium:
        molarity of sodium (more generally, monovalent ions such as Na+, K+, NH4+)
        in moles per liter
    :param magnesium:
        molarity of magnesium (Mg++) in moles per liter
    :param weight:
        how much to weigh this :any:`Constraint`
    :param score_transfer_function:
        See :py:data:`Constraint.score_transfer_function`.
    :param parallel:
        Whether to use parallelization by running constraint evaluation in separate processes
        to take advantage of multiple cores.
    :param strands:
        Strands to check; if not specified, all strands are checked.
    :param description:
        detailed description of constraint suitable for putting in report; if not specified
        a reasonable default is chosen
    :param short_description:
        short description of constraint suitable for logging to stdout
    :return:
        the constraint
    """
    _check_nupack_installed()

    def evaluate(seqs: Tuple[str], _: Optional[Strand]) -> Tuple[float, str]:
        sequence = seqs[0]
        energy = dv.complex_free_energy_single_strand(sequence, temperature, sodium, magnesium)
        excess = threshold - energy
        return max(0.0, excess), f'{energy:6.2f} kcal/mol'

    if description is None:
        description = f'NUPACK secondary structure of strand exceeds {threshold} kcal/mol'

    if strands is not None:
        strands = tuple(strands)

    return StrandConstraint(description=description,
                            short_description=short_description,
                            weight=weight,
                            score_transfer_function=score_transfer_function,
                            evaluate=evaluate,
                            parallel=parallel,
                            strands=strands)


def nupack_domain_pair_constraint(
        threshold: float,
        temperature: float = dv.default_temperature,
        sodium: float = dv.default_sodium,
        magnesium: float = dv.default_magnesium,
        parallel: bool = False,
        weight: float = 1.0,
        score_transfer_function: Callable[[float], float] = default_score_transfer_function,
        description: Optional[str] = None,
        short_description: str = 'dom_pair_nupack',
        pairs: Optional[Iterable[Tuple[Domain, Domain]]] = None,
) -> DomainPairConstraint:
    """
    Returns constraint that checks given pairs of :any:`Domain`'s for excessive interaction using
    NUPACK's pfunc executable. Each of the four combinations of seq1, seq2 and their Watson-Crick complements
    are compared.

    :param threshold:
        Energy threshold in kcal/mol.
    :param temperature:
        Temperature in Celsius
    :param sodium:
        molarity of sodium (more generally, monovalent ions such as Na+, K+, NH4+)
        in moles per liter
    :param magnesium:
        molarity of magnesium (Mg++) in moles per liter
    :param parallel:
        Whether to test the each pair of :any:`Domain`'s in parallel (i.e., sets field
        :py:data:`Constraint.parallel`)
    :param weight:
        How much to weigh this :any:`Constraint`.
    :param score_transfer_function:
        See :py:data:`Constraint.score_transfer_function`.
    :param description:
        Detailed description of constraint suitable for summary report.
    :param short_description:
        Short description of constraint suitable for logging to stdout.
    :param pairs:
        Pairs of :any:`Domain`'s to compare; if not specified, checks all pairs (including a
        :any:`Domain` against itself).
    :return:
        The :any:`DomainPairConstraint`.
    """
    _check_nupack_installed()

    if description is None:
        if isinstance(threshold, Number):
            description = f'NUPACK energy of domain pair exceeds {threshold} kcal/mol'
        elif isinstance(threshold, dict):
            domain_pool_name_pair_to_threshold = {(domain_pool1.name, domain_pool2.name): value
                                                  for (domain_pool1, domain_pool2), value in
                                                  threshold.items()}
            description = f'NUPACK energy of domain pair exceeds threshold defined by their DomainPools ' \
                          f'as follows:\n{domain_pool_name_pair_to_threshold}'
        else:
            raise ValueError(f'threshold = {threshold} must be one of float or dict, '
                             f'but it is {type(threshold)}')

    def binding_closure(seq_pair: Tuple[str, str]) -> float:
        return dv.binding(seq_pair[0], seq_pair[1], temperature=temperature,
                          sodium=sodium, magnesium=magnesium)

    # def evaluate(seq1: str, seq2: str, domain1: Optional[Domain], domain2: Optional[Domain]) -> float:
    def evaluate(seqs: Tuple[str, ...], domain_pair: Optional[DomainPair]) -> Tuple[float, str]:
        seq1, seq2 = seqs
        name_pairs = [(None, None)] * 4
        if domain_pair is not None:
            seq_pairs, name_pairs, _ = _all_pairs_domain_sequences_complements_names_from_domains(
                [domain_pair])
        else:
            # If seq1==seq2, don't check d-d* or d*-d in this case, but do check d-d and d*-d*
            seq_pairs = [
                (seq1, seq2),
                (dv.wc(seq1), dv.wc(seq2)),
            ]
            if seq1 != seq2:
                # only check these if domains are not the same
                seq_pairs.extend([
                    (seq1, dv.wc(seq2)),
                    (dv.wc(seq1), seq2),
                ])

        energies: List[float]
        energies = []
        for seq_pair in seq_pairs:
            energy = binding_closure(seq_pair)
            energies.append(energy)

        excesses: List[float] = []
        for energy, (name1, name2) in zip(energies, name_pairs):
            if name1 is not None and name2 is not None:
                logger.debug(
                    f'domain pair threshold: {threshold:6.2f} '
                    f'binding({name1}, {name2}, {temperature}) = {energy:6.2f} ')
            excess = max(0.0, (threshold - energy))
            excesses.append(excess)

        max_excess = max(excesses)

        max_name_length = max(len(name) for name in _flatten(name_pairs))
        lines_and_energies = [(f'{name1:{max_name_length}}, '
                               f'{name2:{max_name_length}}: '
                               f' {energy:6.2f} kcal/mol', energy)
                              for (name1, name2), energy in zip(name_pairs, energies)]
        lines_and_energies.sort(key=lambda line_and_energy: line_and_energy[1])
        lines = [line for line, _ in lines_and_energies]
        msg = '\n  ' + '\n  '.join(lines)

        return max(0.0, max_excess), msg

    if pairs is not None:
        pairs = tuple(pairs)

    return DomainPairConstraint(description=description,
                                short_description=short_description,
                                weight=weight,
                                score_transfer_function=score_transfer_function,
                                evaluate=evaluate,
                                parallel=parallel,
                                pairs=pairs)


def nupack_strand_pairs_constraint(
        threshold: float,
        temperature: float = dv.default_temperature,
        sodium: float = dv.default_sodium,
        magnesium: float = dv.default_magnesium,
        weight: float = 1.0,
        score_transfer_function: Callable[[float], float] = default_score_transfer_function,
        description: Optional[str] = None,
        short_description: str = 'strand_pair_nupack',
        parallel: bool = False,
        pairs: Optional[Iterable[Tuple[Strand, Strand]]] = None,
) -> StrandPairConstraint:
    """
    Returns constraint that checks given pairs of :any:`Strand`'s for excessive interaction using
    NUPACK's pfunc executable.

    NUPACK 4 must be installed. Installation instructions can be found at
    https://piercelab-caltech.github.io/nupack-docs/start/.

    :param threshold:
        Energy threshold in kcal/mol
    :param temperature:
        Temperature in Celsius
    :param sodium:
        concentration of Na+ in molar
    :param magnesium:
        concentration of Mg++ in molar
    :param weight:
        How much to weigh this :any:`Constraint`.
    :param score_transfer_function:
        See :py:data:`Constraint.score_transfer_function`.
    :param parallel:
        Whether to use parallelization by running constraint evaluation in separate processes
        to take advantage of multiple cores.
    :param description:
        Detailed description of constraint suitable for report.
    :param short_description:
        Short description of constraint suitable for logging to stdout.
    :param pairs:
        Pairs of :any:`Strand`'s to compare; if not specified, checks all pairs (including a
        :any:`Strand` against itself).
    :return:
        The :any:`StrandPairConstraint`.
    """
    _check_nupack_installed()

    if description is None:
        description = f'NUPACK binding energy of strand pair exceeds {threshold} kcal/mol'

    def evaluate(seqs: Tuple[str, ...], _: Optional[StrandPair]) -> Tuple[float, str]:
        seq1, seq2 = seqs
        energy = dv.binding(seq1, seq2, temperature=temperature, sodium=sodium, magnesium=magnesium)
        excess = threshold - energy
        return max(0.0, excess), f'{energy:6.2f} kcal/mol'

    if pairs is not None:
        pairs = tuple(pairs)

    return StrandPairConstraint(description=description,
                                short_description=short_description,
                                weight=weight,
                                score_transfer_function=score_transfer_function,
                                parallel=parallel,
                                pairs=pairs,
                                evaluate=evaluate,
                                # summary=summary
                                )


def chunker(sequence: Sequence[T], chunk_length: Optional[int] = None, num_chunks: Optional[int] = None) -> \
        List[List[T]]:
    """
    Collect data into fixed-length chunks or blocks, e.g., chunker('ABCDEFG', 3) --> ABC DEF G

    :param sequence:
        Sequence (list or tuple) of items.
    :param chunk_length:
        Length of each chunk. Mutually exclusive with `num_chunks`.
    :param num_chunks:
        Number of chunks. Mutually exclusive with `chunk_length`.
    :return:
        List of `num_chunks` lists, each list of length `chunk_length` (one of `num_chunks` or
        `chunk_length` will be calculated from the other).
    """
    if chunk_length is None and num_chunks is None or chunk_length is not None and num_chunks is not None:
        raise ValueError('exactly one of chunk_length or num_chunks must be None')

    if chunk_length is None:
        if num_chunks is None:
            raise ValueError('exactly one of chunk_length or num_chunks must be None')
        if num_chunks < 1:
            raise ValueError('num_chunks must be positive')
        num_items = len(sequence)
        chunk_length, remainder = divmod(num_items, num_chunks)
        if remainder > 0:
            chunk_length += 1

    args = [iter(sequence)] * chunk_length
    chunks = list(itertools.zip_longest(*args, fillvalue=None))
    for i, chunk in enumerate(chunks):
        chunks[i] = [item for item in chunks[i] if item is not None]
    return chunks


def cpu_count(logical: bool = False) -> int:
    """
    Counts the number of physical CPUs (cores). For greatest accuracy, requires the 3rd party
    `psutil <https://pypi.org/project/psutil/>`_
    package to be installed.

    :param logical:
        Whether to count number of logical processors or physical CPU cores.
    :return:
        Number of physical CPU cores if logical is False and package psutils is installed;
        otherwise, the number of logical processors.
    """
    count: Optional[int]
    try:
        import psutil  # type: ignore
        count = psutil.cpu_count(logical=logical)
    except ModuleNotFoundError:
        logger.warning('''\
psutil package not installed. Using os package to determine number of cores.
WARNING: this will count the number of logical cores, but the number of
physical cores is a more effective number to use. It is recommended to
install the package psutil to help determine the number of physical cores
and make parallel processing more efficient:
  https://pypi.org/project/psutil/''')
        count = os.cpu_count()
    if count is None:
        logger.warning('could not determine number of physical CPU cores; defaulting to 1')
        count = 1
    return count


def _check_vienna_rna_installed() -> None:
    try:
        dv.rna_duplex_multiple([("ACGT", "TGCA")])
    except FileNotFoundError:
        raise ImportError('''
Vienna RNA is not installed correctly. Please install it and ensure that 
executables such as RNAduplex can be called from the command line. 
Installation instructions can be found at 
https://github.com/UC-Davis-molecular-computing/dsd#installation and 
https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/install.html''')


def rna_duplex_domain_pairs_constraint(
        threshold: float,
        temperature: float = dv.default_temperature,
        weight: float = 1.0,
        score_transfer_function: Callable[[float], float] = lambda x: x,
        description: Optional[str] = None,
        short_description: str = 'rna_dup_dom_pairs',
        pairs: Optional[Iterable[Tuple[Domain, Domain]]] = None,
        parameters_filename: str = dv.default_vienna_rna_parameter_filename) \
        -> DomainPairsConstraint:
    """
    Returns constraint that checks given pairs of :any:`Domain`'s for excessive interaction using
    Vienna RNA's RNAduplex executable.

    :param threshold:
        energy threshold
    :param temperature:
        temperature in Celsius
    :param weight:
        how much to weigh this :any:`Constraint`
    :param score_transfer_function:
        See :py:data:`Constraint.score_transfer_function`.
    :param description:
        long description of constraint suitable for printing in report file
    :param short_description:
        short description of constraint suitable for logging to stdout
    :param pairs:
        pairs of :any:`Domain`'s to compare; if not specified, checks all pairs
    :param parameters_filename:
        name of parameters file for ViennaRNA; default is
        same as :py:meth:`vienna_nupack.rna_duplex_multiple`
    :return:
        constraint
    """
    _check_vienna_rna_installed()

    if description is None:
        description = f'RNAduplex energy for some domain pairs exceeds {threshold} kcal/mol'

    def evaluate_bulk(domain_pairs: Iterable[DomainPair]) -> List[Tuple[DomainPair, float, str]]:
        sequence_pairs, _, _ = _all_pairs_domain_sequences_complements_names_from_domains(domain_pairs)
        pairs_scores_summaries: List[Tuple[DomainPair, float, str]] = []
        energies = dv.rna_duplex_multiple(sequence_pairs, logger, temperature, parameters_filename)

        for pair, energy in zip(domain_pairs, energies):
            excess = threshold - energy
            summary = f'{energy:6.2f} kcal/mol'
            pair_score_summary = (pair, excess, summary)
            pairs_scores_summaries.append(pair_score_summary)
        return pairs_scores_summaries

    pairs_tuple = None
    if pairs is not None:
        pairs_tuple = tuple(pairs)

    return DomainPairsConstraint(description=description,
                                 short_description=short_description,
                                 weight=weight,
                                 score_transfer_function=score_transfer_function,
                                 evaluate_bulk=evaluate_bulk,
                                 pairs=pairs_tuple)


# def _normalize_threshold_as_dict(threshold: Union[float, Dict[int, float]],
#                                  keys: Iterable[int]) -> Dict[int, float]:
#     # normalize threshold to be a dict with keys `keys` if it is not already, mapping all of the keys
#     # to its float value if it is a float
#     if isinstance(threshold, float):
#         thresholds: Dict[int, float] = {}
#         for num_comp_domains in keys:
#             thresholds[num_comp_domains] = threshold
#         return thresholds
#     else:
#         return threshold


def _populate_strand_list_and_pairs(strands: Optional[Iterable[Strand]],
                                    pairs: Optional[Iterable[Tuple[Strand, Strand]]]) \
        -> Tuple[List[Strand], List[Tuple[Strand, Strand]]]:
    # assert exactly one of strands or pairs is None, then populate the other since both are used below
    # also normalize both to be a list instead of iterable
    if strands is None and pairs is None:
        raise ValueError('exactly one of strands or pairs must be specified, but neither is')
    elif strands is not None and pairs is not None:
        raise ValueError('exactly one of strands or pairs must be specified, but both are')
    elif strands is not None:
        assert pairs is None
        if not isinstance(strands, list):
            strands = list(strands)
        pairs = list(itertools.combinations_with_replacement(strands, 2))
    elif pairs is not None:
        assert strands is None
        if not isinstance(pairs, list):
            pairs = list(pairs)
        strand_names: Set[str] = set()
        strands = []
        for s1, s2 in pairs:
            for strand in [s1, s2]:
                if strand.name not in strand_names:
                    strand_names.add(strand.name)
                    strands.append(strand)

    return strands, pairs


def strand_pairs_by_lengths(strands: Iterable[Strand]) -> Dict[Tuple[int, int], List[Tuple[Strand, Strand]]]:
    """
    Separates pairs of strands in `strands` by lengths. If there are n different strand lengths
    in `strands`, then there are ((n+1) choose 2) keys in the returned dict, one for each pair of
    lengths ``(len1, len2)``, including pairs where ``len1 == len2``. This key maps to a list of all pairs of
    strands in `strands` where the first strand has length ``len1`` and the second has length ``len2``.

    Args:
        strands: strands to check

    Returns:
        dict mapping pairs of lengths to pairs of strands from `strands` having those respective lengths
    """
    pairs: Dict[Tuple[int, int], List[Tuple[Strand, Strand]]] = defaultdict(list)
    for s1, s2 in itertools.combinations_with_replacement(strands, 2):
        len1, len2 = s1.length(), s2.length()
        pairs[(len1, len2)].append((s1, s2))
    return pairs


def strand_pairs_by_number_matching_domains(*, strands: Optional[Iterable[Strand]] = None,
                                            pairs: Optional[Iterable[Tuple[Strand, Strand]]] = None) \
        -> Dict[int, List[Tuple[Strand, Strand]]]:
    """
    Utility function for calculating number of complementary domains betweeen several pairs of strands.

    Args:
        strands: list of :any:`Strand`'s in which to find pairs. Mutually exclusive with `pairs`.
        pairs: list of pairs of strands. Mutually exclusive with `strands`.

    Returns:
        dict mapping integer (number of complementary :any:`Domain`'s) to the list of pairs of strands
        in `strands` with that number of complementary domains
    """
    strands, pairs = _populate_strand_list_and_pairs(strands, pairs)

    # This reduces the number of times we have to create these sets from quadratic to linear
    unstarred_domains_sets = {}
    starred_domains_sets = {}
    for strand in strands:
        unstarred_domains_sets[strand.name] = strand.unstarred_domains_set()
        starred_domains_sets[strand.name] = strand.starred_domains_set()

    # determine which pairs of strands have each number of complementary domains
    strand_pairs: Dict[int, List[Tuple[Strand, Strand]]] = defaultdict(list)
    for strand1, strand2 in pairs:
        domains1_unstarred = unstarred_domains_sets[strand1.name]
        domains2_unstarred = unstarred_domains_sets[strand2.name]
        domains1_starred = starred_domains_sets[strand1.name]
        domains2_starred = starred_domains_sets[strand2.name]

        complementary_domains = (domains1_unstarred & domains2_starred) | \
                                (domains2_unstarred & domains1_starred)
        complementary_domain_names = [domain.name for domain in complementary_domains]
        num_complementary_domains = len(complementary_domain_names)

        strand_pairs[num_complementary_domains].append((strand1, strand2))

    return strand_pairs


class _StrandPairsConstraintCreator(Protocol):
    # Used to specify type of function that
    #   rna_duplex_strand_pairs_constraints_by_number_matching_domains
    # and
    #   rna_cofold_strand_pairs_constraints_by_number_matching_domains
    # both are. See https://mypy.readthedocs.io/en/stable/protocols.html#callback-protocols
    # The Protocol class seems to be available in the typing module, even though the above
    # documentation seems to indicate it is only in typing_extensions?
    def __call__(self,
                 threshold: float,
                 temperature: float = dv.default_temperature,
                 weight: float = 1.0,
                 score_transfer_function: Callable[[float], float] = default_score_transfer_function,
                 description: Optional[str] = None,
                 short_description: str = 'rna_dup_strand_pairs',
                 parallel: bool = False,
                 pairs: Optional[Iterable[Tuple[Strand, Strand]]] = None,
                 parameters_filename: str = dv.default_vienna_rna_parameter_filename,
                 ) -> StrandPairsConstraint: ...


def _strand_pairs_constraints_by_number_matching_domains(
        constraint_creator: _StrandPairsConstraintCreator,
        thresholds: Dict[int, float],
        temperature: float = dv.default_temperature,
        weight: float = 1.0,
        score_transfer_function: Callable[[float], float] = default_score_transfer_function,
        descriptions: Optional[Dict[int, str]] = None,
        short_descriptions: Optional[Dict[int, str]] = None,
        parallel: bool = False,
        strands: Optional[Iterable[Strand]] = None,
        pairs: Optional[Iterable[Tuple[Strand, Strand]]] = None,
        parameters_filename: str = dv.default_vienna_rna_parameter_filename,
):
    # function to share common code between
    #   rna_duplex_strand_pairs_constraints_by_number_matching_domains
    # and
    #   rna_cofold_strand_pairs_constraints_by_number_matching_domains
    if strands is None and pairs is None:
        raise ValueError('exactly one of strands or pairs must be specified, but neither is')
    elif strands is not None and pairs is not None:
        raise ValueError('exactly one of strands or pairs must be specified, but both are')

    if strands is not None:
        assert pairs is None
        pairs = itertools.combinations_with_replacement(strands, 2)

    pairs_by_matching_domains = strand_pairs_by_number_matching_domains(pairs=pairs)
    keys = set(pairs_by_matching_domains.keys())
    if thresholds is not None:
        thres_keys = set(thresholds.keys())
        if keys != thres_keys:
            raise ValueError(f'The keys of parameter thresholds must be exactly {sorted(list(keys))}, '
                             'which is the set of integers representing the number of matching domains '
                             'across all pairs of Strands in the parameter pairs, '
                             f'but instead the thresholds.keys() is {sorted(list(thres_keys))}')

    constraints: List[StrandPairsConstraint] = []

    for num_matching_domains, threshold in thresholds.items():
        pairs_with_matching_domains = pairs_by_matching_domains[num_matching_domains]

        if descriptions is None or num_matching_domains not in descriptions:
            description = f'RNAduplex energy for pairs strand with {num_matching_domains} ' \
                          f'matching domains exceeds {threshold} kcal/mol at {temperature} C'
        else:
            description = descriptions[num_matching_domains]

        if short_descriptions is None or num_matching_domains not in short_descriptions:
            short_description = f'RNADup{num_matching_domains}Comp'
        else:
            short_description = short_descriptions[num_matching_domains]

        constraint = constraint_creator(
            threshold=threshold,
            temperature=temperature,
            weight=weight,
            score_transfer_function=score_transfer_function,
            description=description,
            short_description=short_description,
            parallel=parallel,
            pairs=pairs_with_matching_domains,
            parameters_filename=parameters_filename,
        )
        constraints.append(constraint)

    return constraints


def rna_cofold_strand_pairs_constraints_by_number_matching_domains(
        thresholds: Dict[int, float],
        temperature: float = dv.default_temperature,
        weight: float = 1.0,
        score_transfer_function: Callable[[float], float] = default_score_transfer_function,
        descriptions: Optional[Dict[int, str]] = None,
        short_descriptions: Optional[Dict[int, str]] = None,
        parallel: bool = False,
        strands: Optional[Iterable[Strand]] = None,
        pairs: Optional[Iterable[Tuple[Strand, Strand]]] = None,
        parameters_filename: str = dv.default_vienna_rna_parameter_filename) \
        -> List[StrandPairsConstraint]:
    """
    Similar to :meth:`rna_duplex_strand_pairs_constraints_by_number_matching_domains`
    but creates constraints as returned by :meth:`rna_cofold_strand_pairs_constraint`.
    """
    return _strand_pairs_constraints_by_number_matching_domains(
        rna_cofold_strand_pairs_constraint,
        thresholds=thresholds,
        temperature=temperature,
        weight=weight,
        score_transfer_function=score_transfer_function,
        descriptions=descriptions,
        short_descriptions=short_descriptions,
        parallel=parallel,
        strands=strands,
        pairs=pairs,
        parameters_filename=parameters_filename,
    )


def rna_duplex_strand_pairs_constraints_by_number_matching_domains(
        thresholds: Dict[int, float],
        temperature: float = dv.default_temperature,
        weight: float = 1.0,
        score_transfer_function: Callable[[float], float] = default_score_transfer_function,
        descriptions: Optional[Dict[int, str]] = None,
        short_descriptions: Optional[Dict[int, str]] = None,
        parallel: bool = False,
        strands: Optional[Iterable[Strand]] = None,
        pairs: Optional[Iterable[Tuple[Strand, Strand]]] = None,
        parameters_filename: str = dv.default_vienna_rna_parameter_filename) \
        -> List[StrandPairsConstraint]:
    """
    Convenience function for creating many constraints as returned by
    :meth:`rna_duplex_strand_pairs_constraint`, one for each threshold specified in parameter `thresholds`,
    based on number of matching (complementary) domains between pairs of strands.

    Optional parameters `description` and `short_description` are also dicts keyed by the same keys.

    Exactly one of `strands` or `pairs` must be specified. If `strands`, then all pairs of strands
    (including a strand with itself) will be checked; otherwise only those pairs in `pairs` will be checked.

    It is also common to set different thresholds according to the lengths of the strands.
    This can be done by calling :meth:`strand_pairs_by_lengths` to separate first by lengths
    in a dict mapping length pairs to strand pairs,
    then calling this function once for each (key, value) in that dict, giving the value
    (which is a list of pairs of strands) as the `pairs` parameter to this function.

    Args:
        thresholds: Energy thresholds in kcal/mol. If `k` domains are complementary between the strands,
                    then use threshold `thresholds[k]`.
        temperature: Temperature in Celsius.
        weight: How much to weigh this :any:`Constraint`.
        score_transfer_function: See :py:data:`Constraint.score_transfer_function`.
        descriptions: Long descriptions of constraint suitable for putting into constraint report.
        short_descriptions: Short descriptions of constraint suitable for logging to stdout.
        parallel: Whether to test the each pair of :any:`Strand`'s in parallel.
        strands: Pairs of :any:`Strand`'s to compare; if not specified, checks all pairs in `pairs`.
                 Mutually exclusive with `pairs`.
        pairs: Pairs of :any:`Strand`'s to compare; if not specified, checks all pairs in `strands`,
               including each strand with itself.
               Mutually exclusive with `strands`.
        parameters_filename: Name of parameters file for ViennaRNA;
                             default is same as :py:meth:`vienna_nupack.rna_duplex_multiple`

    Returns:
        list of constraints, one per threshold in `thresholds`
    """
    return _strand_pairs_constraints_by_number_matching_domains(
        rna_duplex_strand_pairs_constraint,
        thresholds=thresholds,
        temperature=temperature,
        weight=weight,
        score_transfer_function=score_transfer_function,
        descriptions=descriptions,
        short_descriptions=short_descriptions,
        parallel=parallel,
        strands=strands,
        pairs=pairs,
        parameters_filename=parameters_filename,
    )


def rna_duplex_strand_pairs_constraint(
        threshold: float,
        temperature: float = dv.default_temperature,
        weight: float = 1.0,
        score_transfer_function: Callable[[float], float] = default_score_transfer_function,
        description: Optional[str] = None,
        short_description: str = 'rna_dup_strand_pairs',
        parallel: bool = False,
        pairs: Optional[Iterable[Tuple[Strand, Strand]]] = None,
        parameters_filename: str = dv.default_vienna_rna_parameter_filename) \
        -> StrandPairsConstraint:
    """
    Returns constraint that checks given pairs of :any:`Strand`'s for excessive interaction using
    Vienna RNA's RNAduplex executable.

    Often one wishes to let the threshold depend on how many domains match between a pair of strands.
    The function :meth:`rna_duplex_strand_pairs_constraints_by_number_matching_domains` is useful
    for this purpose, returning a list of :any:`StrandPairsConstraint`'s such as those returned by this
    function, one for each possible number of matching domains.

    :param threshold:
        Energy threshold in kcal/mol. If a float, this is used for all pairs of strands.
        If a dict[int, float], interpreted to mean that
    :param temperature:
        Temperature in Celsius.
    :param weight:
        How much to weigh this :any:`Constraint`.
    :param score_transfer_function:
        See :py:data:`Constraint.score_transfer_function`.
    :param description:
        Long description of constraint suitable for putting into constraint report.
    :param short_description:
        Short description of constraint suitable for logging to stdout.
    :param parallel:
        Whether to test the each pair of :any:`Strand`'s in parallel.
    :param pairs:
        Pairs of :any:`Strand`'s to compare; if not specified, checks all pairs in design.
    :param parameters_filename:
        Name of parameters file for ViennaRNA;
        default is same as :py:meth:`vienna_nupack.rna_duplex_multiple`
    :return:
        The :any:`StrandPairsConstraint`.
    """
    _check_vienna_rna_installed()

    if description is None:
        description = f'RNAduplex energy for some strand pairs exceeds ' \
                      f'{threshold} kcal/mol at {temperature} C'

    num_threads = max(cpu_count() - 1, 1)  # this seems to be slightly faster than using all cores

    # we use ThreadPool instead of pathos because we're farming this out to processes through
    # subprocess module anyway, no need for pathos to boot up separate processes or serialize through dill
    thread_pool = ThreadPool(processes=num_threads)

    def calculate_energies_unparallel(sequence_pairs: Sequence[Tuple[str, str]]) -> List[float]:
        return dv.rna_duplex_multiple(sequence_pairs, logger, temperature, parameters_filename)

    def calculate_energies(sequence_pairs: Sequence[Tuple[str, str]]) -> List[float]:
        if parallel and len(sequence_pairs) > 1:
            lists_of_sequence_pairs = chunker(sequence_pairs, num_chunks=num_threads)
            lists_of_energies = thread_pool.map(calculate_energies_unparallel, lists_of_sequence_pairs)
            energies = _flatten(lists_of_energies)
        else:
            energies = calculate_energies_unparallel(sequence_pairs)
        return energies

    def evaluate(strand_pairs: Iterable[StrandPair]) -> List[Tuple[StrandPair, float, str]]:
        sequence_pairs = [(pair.strand1.sequence(), pair.strand2.sequence()) for pair in strand_pairs]
        energies = calculate_energies(sequence_pairs)
        pairs_scores_summaries: List[Tuple[StrandPair, float, str]] = []

        for pair, energy in zip(strand_pairs, energies):
            excess = threshold - energy
            summary = f'{energy:6.2f} kcal/mol'
            pair_score_summary = (pair, excess, summary)
            pairs_scores_summaries.append(pair_score_summary)
        return pairs_scores_summaries

    pairs_tuple = None
    if pairs is not None:
        pairs_tuple = tuple(pairs)

    return StrandPairsConstraint(description=description,
                                 short_description=short_description,
                                 weight=weight,
                                 score_transfer_function=score_transfer_function,
                                 evaluate_bulk=evaluate,
                                 pairs=pairs_tuple)


def energy_excess(energy: float, threshold: float) -> float:
    excess = threshold - energy
    return excess


def energy_excess_domains(energy: float,
                          threshold: Union[float, Dict[Tuple[DomainPool, DomainPool], float]],
                          domain1: Domain, domain2: Domain) -> float:
    threshold_value = 0.0  # noqa; warns that variable isn't used even though it clearly is
    if isinstance(threshold, Number):
        threshold_value = threshold
    elif isinstance(threshold, dict):
        threshold_value = threshold[(domain1.pool, domain2.pool)]
    excess = threshold_value - energy
    return excess


def rna_cofold_strand_pairs_constraint(
        threshold: float,
        temperature: float = dv.default_temperature,
        weight: float = 1.0,
        score_transfer_function: Callable[[float], float] = default_score_transfer_function,
        description: Optional[str] = None,
        short_description: str = 'rna_dup_strand_pairs',
        parallel: bool = False,
        pairs: Optional[Iterable[Tuple[Strand, Strand]]] = None,
        parameters_filename: str = dv.default_vienna_rna_parameter_filename) \
        -> StrandPairsConstraint:
    """
    Returns constraint that checks given pairs of :any:`Strand`'s for excessive interaction using
    Vienna RNA's RNAduplex executable.

    :param threshold:
        Energy threshold in kcal/mol
    :param temperature:
        Temperature in Celsius.
    :param weight:
        How much to weigh this :any:`Constraint`.
    :param score_transfer_function:
        See :py:data:`Constraint.score_transfer_function`.
    :param description:
        Long description of constraint suitable for putting into constraint report.
    :param short_description:
        Short description of constraint suitable for logging to stdout.
    :param parallel:
        Whether to test the each pair of :any:`Strand`'s in parallel.
    :param pairs:
        Pairs of :any:`Strand`'s to compare; if not specified, checks all pairs.
    :param parameters_filename:
        Name of parameters file for ViennaRNA;
        default is same as :py:meth:`vienna_nupack.rna_duplex_multiple`
    :return:
        The :any:`StrandPairsConstraint`.
    """
    _check_vienna_rna_installed()

    if description is None:
        description = f'RNAcofold energy for some strand pairs exceeds {threshold} kcal/mol'

    num_threads = max(cpu_count() - 1, 1)  # this seems to be slightly faster than using all cores

    # we use ThreadPool instead of pathos because we're farming this out to processes through
    # subprocess module anyway, no need for pathos to boot up separate processes or serialize through dill
    thread_pool = ThreadPool(processes=num_threads)

    def calculate_energies_unparallel(sequence_pairs: Sequence[Tuple[str, str]]) -> List[float]:
        return dv.rna_cofold_multiple(sequence_pairs, logger, temperature, parameters_filename)

    def calculate_energies(sequence_pairs: Sequence[Tuple[str, str]]) -> List[float]:
        if parallel and len(sequence_pairs) > 1:
            lists_of_sequence_pairs = chunker(sequence_pairs, num_chunks=num_threads)
            lists_of_energies = thread_pool.map(calculate_energies_unparallel, lists_of_sequence_pairs)
            energies = _flatten(lists_of_energies)
        else:
            energies = calculate_energies_unparallel(sequence_pairs)
        return energies

    def evaluate(strand_pairs: Iterable[StrandPair]) -> List[Tuple[StrandPair, float, str]]:
        sequence_pairs = [(pair.strand1.sequence(), pair.strand2.sequence()) for pair in strand_pairs]
        energies = calculate_energies(sequence_pairs)
        pairs_scores_summaries: List[Tuple[StrandPair, float, str]] = []

        for pair, energy in zip(strand_pairs, energies):
            excess = threshold - energy
            summary = f'{energy:6.2f} kcal/mol'
            pair_score_summary = (pair, excess, summary)
            pairs_scores_summaries.append(pair_score_summary)
        return pairs_scores_summaries

    pairs_tuple = None
    if pairs is not None:
        pairs_tuple = tuple(pairs)

    return StrandPairsConstraint(description=description,
                                 short_description=short_description,
                                 weight=weight,
                                 score_transfer_function=score_transfer_function,
                                 evaluate_bulk=evaluate,
                                 pairs=pairs_tuple)


def _all_pairs_domain_sequences_complements_names_from_domains(
        domain_pairs: Iterable[DomainPair]) \
        -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[Domain, Domain]]]:
    """
    :param domain_pairs:
        Domain pairs.
    :return:
        pair consisting of two lists, each of length 4 times as long as `domain_pairs`.
        Each pair in `domain_pairs` is associated to the 4 combinations of WC complementing (or not)
        the sequences of each Domain.
        - sequence_pairs: the sequences (appropriated complemented or not)
        - names: the names (appropriately *'d or not)
    """
    sequence_pairs: List[Tuple[str, str]] = []
    names: List[Tuple[str, str]] = []
    domains: List[Tuple[Domain, Domain]] = []
    for pair in domain_pairs:
        d1, d2 = pair.individual_parts()
        if d1 == d2:
            # don't check d-d* or d*-d in this case, but do check d-d and d*-d*
            starred_each = [(False, False), (True, True)]
        else:
            starred_each = [(False, False), (True, True), (False, True), (True, False)]
        for starred1, starred2 in starred_each:
            seq1 = d1.concrete_sequence(starred1)
            seq2 = d2.concrete_sequence(starred2)
            name1 = d1.get_name(starred1)
            name2 = d2.get_name(starred2)
            sequence_pairs.append((seq1, seq2))
            names.append((name1, name2))
            domains.append((d1, d2))
    return sequence_pairs, names, domains


def _flatten(list_of_lists: Iterable[Iterable[T]]) -> List[T]:
    #  Flatten one level of nesting
    return list(itertools.chain.from_iterable(list_of_lists))


#########################################################################################
# ComplexConstraints defined below here
#########################################################################################


@dataclass(frozen=True, eq=False)
class ConstraintWithComplexes(Constraint[DesignPart], Generic[DesignPart]):  # noqa
    complexes: Tuple[Complex, ...] = ()
    """
    List of complexes (tuples of :any:`Strand`'s) to check.
    """


@dataclass(frozen=True, eq=False)  # type: ignore
class ComplexConstraint(ConstraintWithComplexes[Complex], SingularConstraint[Complex]):
    """
    Constraint that applies to a complex (tuple of :any:`Strand`'s).

    Specify :data:`Constraint._evaluate` in the constructor.

    Unlike other types of :any:`Constraint`'s such as :any:`StrandConstraint` or :any:`StrandPairConstraint`,
    there is no default list of :any:`Complex`'s that a :any:`ComplexConstraint` is applied to. The list of
    :any:`Complex`'s must be specified manually in the constructor.
    """

    def part_name(self) -> str:
        return 'complex'


def _alter_scores_by_transfer(sets_excesses: List[Tuple[OrderedSet[Domain], float]],
                              transfer_callback: Callable[[float], float]) \
        -> List[Tuple[OrderedSet[Domain], float]]:
    sets_weights: List[Tuple[OrderedSet[Domain], float]] = []
    for set_, excess in sets_excesses:
        if excess < 0:
            weight = 0.0
        else:
            weight = transfer_callback(excess)
        sets_weights.append((set_, weight))
    return sets_weights


@dataclass(frozen=True, eq=False)  # type: ignore
class ComplexesConstraint(ConstraintWithComplexes[Iterable[Complex]], BulkConstraint[Complex]):
    """
    Similar to :any:`ComplexConstraint` but operates on a specified list of complexes
    (tuples of :any:`Strand`'s).
    """

    def part_name(self) -> str:
        return 'complex'


class _AdjacentDuplexType(Enum):
    # Refer to comments under BaseTypePair for notation reference
    #
    #
    # Domain definitions:
    #   All AdjacentDuplexType are with reference to domain c.
    #   AdjacentDuplexType is agnostic to the ends of domain c.
    #   Hence, c is written as #-----# in each of the AdjacentDuplexType
    #   variants.
    #
    #   c* - complement of c   (must exist)
    #   d* - 5' neighbor of c* (does not neccessarily exist)
    #   d  - complement of d*  (does not neccessarily exist)
    #   e  - 5' neighbor of d  (does not neccessarily exist)
    #   e* - complement of e   (does not neccessarily exist)
    #
    #                          # #
    #                          |-|
    #                       e* |-| e (bound)
    #                          |-|
    #                          # #
    #                       c  ? # d
    #                    #-----# #-----#
    #                     |||||   |||||
    #                    #-----###-----#
    #                       c*    d*
    #
    #   Note: if "?" was a "#" (meaning e* is adjacent to c), then then
    #   it would be a three arm junction
    #
    ###########################################################################

    # d* does not exist
    #                       c
    #                    #-----#
    #                     |||||
    #                    #-----]
    #                       c*
    BOTTOM_RIGHT_EMPTY = auto()  # type: ignore

    # d* exist, but d does not exist
    #                       c
    #                    #-----#
    #                     |||||
    #                    #-----##----#
    #                       c*    d*
    BOTTOM_RIGHT_DANGLE = auto()  # type: ignore

    # d* and d exist, but e does not exist
    # d is is the 5' end of the strand
    #                       c     d
    #                    #-----#[----#
    #                     |||||  ||||
    #                    #-----##----#
    #                       c*    d*
    TOP_RIGHT_5P = auto()  # type: ignore

    # d* and d and e exist, but e* does not exist
    #                           #
    #                           |
    #                           | e (unbound)
    #                           |
    #                           #
    #                       c   | d
    #                    #-----#+---#
    #                     ||||| ||||
    #                    #-----#----#
    #                       c*    d*
    TOP_RIGHT_OVERHANG = auto()  # type: ignore

    # d* and d and e and e* exist
    #
    # ? on 3p end of c domain because this case is agnostic to
    # whether c and e* are connected
    #
    #                          # #
    #                          |-|
    #                       e* |-| e (bound)
    #                          |-|
    #                          # #
    #                       c  ? # d
    #                    #-----# #---#
    #                     |||||  ||||
    #                    #-----###---#
    #                       c*    d*
    TOP_RIGHT_BOUND_OVERHANG = auto()  # type: ignore


default_interior_to_strand_probability = 0.98
"""Default probability threshold for :py:attr:`BasePairType.INTERIOR_TO_STRAND`"""
default_adjacent_to_exterior_base_pair = 0.95
"""Default probability threshold for :py:attr:`BasePairType.ADJACENT_TO_EXTERIOR_BASE_PAIR`"""

default_blunt_end_probability = 0.33
"""Default probability threshold for :py:attr:`BasePairType.BLUNT_END`"""

default_nick_3p_probability = 0.79
"""Default probability threshold for :py:attr:`BasePairType.NICK_3P`"""

default_nick_5p_probability = 0.73
"""Default probability threshold for :py:attr:`BasePairType.NICK_5P`"""

default_dangle_3p_probability = 0.51
"""Default probability threshold for :py:attr:`BasePairType.DANGLE_3P`"""

default_dangle_5p_probability = 0.57
"""Default probability threshold for :py:attr:`BasePairType.DANGLE_5P`"""

default_dangle_5p_3p_probability = 0.73
"""Default probability threshold for :py:attr:`BasePairType.DANGLE_5P_3P`"""

default_overhang_on_this_strand_3p_probability = 0.82
"""Default probability threshold for :py:attr:`BasePairType.OVERHANG_ON_THIS_STRAND_3P`"""

default_overhang_on_this_strand_5p_probability = 0.79
"""Default probability threshold for :py:attr:`BasePairType.OVERHANG_ON_THIS_STRAND_5P`"""

default_overhang_on_adjacent_strand_3p_probability = 0.55
"""Default probability threshold for :py:attr:`BasePairType.OVERHANG_ON_ADJACENT_STRAND_3P`"""

default_overhang_on_adjacent_strand_5p_probability = 0.49
"""Default probability threshold for :py:attr:`BasePairType.OVERHANG_ON_ADJACENT_STRAND_5P`"""

default_overhang_on_both_strand_3p_probability = 0.61
"""Default probability threshold for :py:attr:`BasePairType.OVERHANG_ON_BOTH_STRANDS_3P`"""

default_overhang_on_both_strand_5p_probability = 0.55
"""Default probability threshold for :py:attr:`BasePairType.OVERHANG_ON_BOTH_STRANDS_5P`"""

default_three_arm_junction_probability = 0.69
"""Default probability threshold for :py:attr:`BasePairType.THREE_ARM_JUNCTION`"""

default_four_arm_junction_probability = 0.84
"""Default probability threshold for :py:attr:`BasePairType.FOUR_ARM_JUNCTION`"""

default_five_arm_junction_probability = 0.77
"""Default probability threshold for :py:attr:`BasePairType.FIVE_ARM_JUNCTION`"""

default_mismatch_probability = 0.76
"""Default probability threshold for :py:attr:`BasePairType.MISMATCH`"""

default_bulge_loop_3p_probability = 0.69
"""Default probability threshold for :py:attr:`BasePairType.BULGE_LOOP_3P`"""

default_bulge_loop_5p_probability = 0.65
"""Default probability threshold for :py:attr:`BasePairType.BULGE_LOOP_5P`"""

default_unpaired_probability = 0.95
"""Default probability threshold for :py:attr:`BasePairType.UNPAIRED`"""

default_other_probability = 0.70
"""Default probability threshold for :py:attr:`BasePairType.OTHER`"""


class BasePairType(Enum):
    """
    Represents different configurations for a base pair and it's immediate
    neighboring base pairs (or lack thereof).

    **Notation**:

    * "#" indicates denotes the ends of a domain.
      They can either be the end of a strand or they could be connected to another domain.
    * "]" and "[" indicates 5' ends of strand
    * ">" and "<" indicates 3' ends of a strand
    * "-" indicates a base (number of these are not important).
    * "|" indicates a bases are bound (forming a base pair).
      Any "-" not connected by "|" is unbound

    **Domain Example**:

    The following represents an unbound domain of length 5

    .. code-block:: none

        #-----#

    The following represents bound domains of length 5

    .. code-block:: none

        #-----#
         |||||
        #-----#


    Ocassionally, domains will be vertical in the case of overhangs.
    In this case, "-" and "|" have opposite meanings

    **Vertical Domain Example**:

    .. code-block:: none

        # #
        |-|
        |-|
        |-|
        |-|
        |-|
        # #

    **Formatting**:

    * Top strands have 5' end on left side and 3' end on right side
    * Bottom strand have 3' end on left side and 5' end on right side

    **Strand Example**:

    .. code-block:: none

      strand0: a-b-c-d
      strand1: d*-b*-c*-a*

                  a      b      c      d
      strand0  [-----##-----##-----##----->
                |||||  |||||  |||||  |||||
      strand1  <-----##-----##-----##-----]
                  a*     b*     c*     d*

    **Consecutive "#"**:

    In some cases, extra "#" are needed to to make space for ascii art.
    We consider any consecutive "#"s to be equivalent "##".
    The following is consider equivalent to the example above

    .. code-block:: none

                  a       b        c      d
      strand0  [-----###-----####-----##----->
                |||||   |||||    |||||  |||||
      strand1  <-----###-----####-----##-----]
                  a*      b*       c*     d*

    Note that only consecutive "#"s is consider equivalent to "##".
    The following example is not equivalent to the strands above because
    the "#  #" between b and c are seperated by spaces, so they are
    not equivalent to "##", meaning that b and c neednot be adjacent.
    Note that while b and c need not be adjacent, b* and c* are still
    adjacent because they are seperated by consecutive "#"s with no
    spaces in between.

    .. code-block:: none

                  a       b        c      d
      strand0  [-----###-----#  #-----##----->
                |||||   |||||    |||||  |||||
      strand1  <-----###-----####-----##-----]
                  a*      b*       c*     d*
    """

    INTERIOR_TO_STRAND = auto()  # type: ignore
    """
    Base pair is located inside of a strand but not next
    to a base pair that resides on the end of a strand.

    Similar base-pairing probability compared to :py:attr:`ADJACENT_TO_EXTERIOR_BASE_PAIR` 
    but usually breathes less.

    .. code-block:: none

        #-----##-----#
         |||||  |||||
        #-----##-----#
             ^
             |
         base pair

    """

    ADJACENT_TO_EXTERIOR_BASE_PAIR = auto()  # type: ignore
    """
    Base pair is located inside of a strand and next
    to a base pair that resides on the end of a strand.

    Similar base-pairing probability compared to :py:attr:`INTERIOR_TO_STRAND` but usually breathes more.

    .. code-block:: none

        #-----#
         |||||
        #-----]
            ^
            |
        base pair

    or

    .. code-block:: none


        #----->
         |||||
        #-----#
            ^
            |
        base pair
    """

    BLUNT_END = auto()  # type: ignore
    """
    Base pair is located at the end of both strands.

    .. code-block:: none

        #----->
         |||||
        #-----]
             ^
             |
         base pair
    """

    NICK_3P = auto()  # type: ignore
    """
    Base pair is located at a nick involving the 3' end of the strand.

    .. code-block:: none

        #----->[-----#
         |||||  |||||
        #-----##-----#
             ^
             |
         base pair

    """

    NICK_5P = auto()  # type: ignore
    """
    Base pair is located at a nick involving the 3' end of the strand.

    .. code-block:: none

        #-----##-----#
         |||||  |||||
        #-----]<-----#
             ^
             |
         base pair
    """

    DANGLE_3P = auto()  # type: ignore
    """
    Base pair is located at the end of a strand with a dangle on the
    3' end.

    .. code-block:: none

        #-----##----#
         |||||
        #-----]
             ^
             |
         base pair
    """

    DANGLE_5P = auto()  # type: ignore
    """
    Base pair is located at the end of a strand with a dangle on the
    5' end.

    .. code-block:: none

        #----->
         |||||
        #-----##----#
             ^
             |
         base pair
    """

    DANGLE_5P_3P = auto()  # type: ignore
    """
    Base pair is located with dangle at both the 3' and 5' end.

    .. code-block:: none

        #-----##----#
         |||||
        #-----##----#
             ^
             |
         base pair
    """

    OVERHANG_ON_THIS_STRAND_3P = auto()  # type: ignore
    """
    Base pair is next to a overhang on the 3' end.

    .. code-block:: none

              #
              |
              |
              |
              #
        #-----# #-----#
         |||||   |||||
        #-----###-----#
             ^
             |
         base pair
    """

    OVERHANG_ON_THIS_STRAND_5P = auto()  # type: ignore
    """
    Base pair is next to a overhang on the 5' end.

    .. code-block:: none

         base pair
             |
             v
        #-----###-----#
         |||||   |||||
        #-----# #-----#
              #
              |
              |
              |
              #
    """

    OVERHANG_ON_ADJACENT_STRAND_3P = auto()  # type: ignore
    """
    Base pair 3' end interfaces with an overhang.

    The adjacent base pair type is :py:attr:`OVERHANG_ON_THIS_STRAND_5P`

    .. code-block:: none

                #
                |
                |
                |
                #
        #-----# #---#
         |||||   |||
        #-----###---#
             ^
             |
         base pair
    """

    OVERHANG_ON_ADJACENT_STRAND_5P = auto()  # type: ignore
    """
    Base pair 5' end interfaces with an overhang.

    The adjacent base pair type is :py:attr:`OVERHANG_ON_THIS_STRAND_3P`

    .. code-block:: none

         base pair
             |
             v
        #-----###-----#
         |||||   |||||
        #-----# #-----#
                #
                |
                |
                |
                #
    """

    OVERHANG_ON_BOTH_STRANDS_3P = auto()  # type: ignore
    """
    Base pair's 3' end is an overhang and adjacent strand also has an overhang.

    .. code-block:: none

              # #
              | |
              | |
              | |
              # #
        #-----# #---#
         |||||  ||||
        #-----###---#
             ^
             |
         base pair
    """

    OVERHANG_ON_BOTH_STRANDS_5P = auto()  # type: ignore
    """
    Base pair's 5' end is an overhang and adjacent strand also has an overhang.

    .. code-block:: none

         base pair
             |
             v
        #-----###-----#
         |||||   |||||
        #-----# #-----#
              # #
              | |
              | |
              | |
              # #
    """

    THREE_ARM_JUNCTION = auto()  # type: ignore
    """
    Base pair is located next to a three-arm-junction.

    .. code-block:: none


              # #
              |-|
              |-|
              |-|
              # #
        #-----# #---#
         |||||  ||||
        #-----###---#
             ^
             |
         base pair
    """

    FOUR_ARM_JUNCTION = auto()  # type: ignore
    """
    TODO: Currently, this case isn't actually detected (considered as :py:attr:`OTHER`).

    Base pair is located next to a four-arm-junction (e.g. Holliday junction).

    .. code-block:: none

              # #
              |-|
              |-|
              |-|
              # #
        #-----# #-----#
         |||||   |||||
        #-----# #-----#
              # #
              |-|
              |-|
              |-|
              # #
    """

    FIVE_ARM_JUNCTION = auto()  # type: ignore
    """
    TODO: Currently, this case isn't actually detected (considered as :py:attr:`OTHER`).

    Base pair is located next to a five-arm-junction.
    """

    MISMATCH = auto()  # type: ignore
    """
    TODO: Currently, this case isn't actually detected (considered as :py:attr:`DANGLE_5P_3P`).

    Base pair is located next to a mismatch.

    .. code-block:: none

        #-----##-##-----#
         |||||     |||||
        #-----##-##-----#
             ^
             |
         base pair
    """

    BULGE_LOOP_3P = auto()  # type: ignore
    """
    TODO: Currently, this case isn't actually detected (considered as :py:attr:`OVERHANG_ON_BOTH_STRANDS_3P`).

    Base pair is located next to a mismatch.

    .. code-block:: none

        #-----##-##-----#
         |||||     |||||
        #-----#####-----#
             ^
             |
         base pair
    """

    BULGE_LOOP_5P = auto()  # type: ignore
    """
    TODO: Currently, this case isn't actually detected (considered as :py:attr:`OVERHANG_ON_BOTH_STRANDS_5P`).

    Base pair is located next to a mismatch.

    .. code-block:: none

        #-----#####-----#
         |||||     |||||
        #-----##-##-----#
             ^
             |
         base pair
    """

    UNPAIRED = auto()  # type: ignore
    """
    Base is unpaired.

    Probabilities specify how unlikely a base is to be paired with another base.
    """

    OTHER = auto()  # type: ignore
    """
    Other base pair types.
    """

    def default_pair_probability(self) -> float:
        if self is BasePairType.INTERIOR_TO_STRAND:
            return default_interior_to_strand_probability
        elif self is BasePairType.ADJACENT_TO_EXTERIOR_BASE_PAIR:
            return default_adjacent_to_exterior_base_pair
        elif self is BasePairType.BLUNT_END:
            return default_blunt_end_probability
        elif self is BasePairType.NICK_3P:
            return default_nick_3p_probability
        elif self is BasePairType.NICK_5P:
            return default_nick_5p_probability
        elif self is BasePairType.DANGLE_3P:
            return default_dangle_3p_probability
        elif self is BasePairType.DANGLE_5P:
            return default_dangle_5p_probability
        elif self is BasePairType.DANGLE_5P_3P:
            return default_dangle_5p_3p_probability
        elif self is BasePairType.OVERHANG_ON_THIS_STRAND_3P:
            return default_overhang_on_this_strand_3p_probability
        elif self is BasePairType.OVERHANG_ON_THIS_STRAND_5P:
            return default_overhang_on_this_strand_5p_probability
        elif self is BasePairType.OVERHANG_ON_ADJACENT_STRAND_3P:
            return default_overhang_on_adjacent_strand_3p_probability
        elif self is BasePairType.OVERHANG_ON_ADJACENT_STRAND_5P:
            return default_overhang_on_adjacent_strand_5p_probability
        elif self is BasePairType.OVERHANG_ON_BOTH_STRANDS_3P:
            return default_overhang_on_both_strand_3p_probability
        elif self is BasePairType.OVERHANG_ON_BOTH_STRANDS_5P:
            return default_overhang_on_both_strand_5p_probability
        elif self is BasePairType.THREE_ARM_JUNCTION:
            return default_three_arm_junction_probability
        elif self is BasePairType.FOUR_ARM_JUNCTION:
            return default_four_arm_junction_probability
        elif self is BasePairType.FIVE_ARM_JUNCTION:
            return default_five_arm_junction_probability
        elif self is BasePairType.OTHER:
            return default_other_probability
        elif self is BasePairType.UNPAIRED:
            return default_unpaired_probability
        elif self is BasePairType.BULGE_LOOP_3P:
            return default_bulge_loop_3p_probability
        elif self is BasePairType.BULGE_LOOP_5P:
            return default_bulge_loop_5p_probability
        elif self is BasePairType.MISMATCH:
            return default_mismatch_probability
        else:
            assert False


@dataclass
class StrandDomainAddress:
    """An addressing scheme for specifying a domain on a strand.
    """

    strand: Strand
    """strand to index
    """

    domain_idx: int
    """order in which domain appears in :py:data:`StrandDomainAddress.strand`
    """

    def neighbor_5p(self) -> Optional['StrandDomainAddress']:
        """Returns 5' domain neighbor. If domain is 5' end of strand, returns None

        :return: StrandDomainAddress of 5' neighbor or None if no 5' neighbor
        :rtype: Optional[StrandDomainAddress]
        """
        idx = self.domain_idx - 1
        if idx >= 0:
            return StrandDomainAddress(self.strand, idx)
        else:
            return None

    def neighbor_3p(self) -> Optional['StrandDomainAddress']:
        """Returns 3' domain neighbor. If domain is 3' end of strand, returns None

        :return: StrandDomainAddress of 3' neighbor or None if no 3' neighbor
        :rtype: Optional[StrandDomainAddress]
        """
        idx = self.domain_idx + 1
        if idx < len(self.strand.domains):
            return StrandDomainAddress(self.strand, idx)
        else:
            return None

    def domain(self) -> Domain:
        """Returns domain referenced by this address.

        :return: domain
        :rtype: Domain
        """
        return self.strand.domains[self.domain_idx]

    def __hash__(self) -> int:
        return hash((self.strand, self.domain_idx))

    def __eq__(self, other):
        if isinstance(other, StrandDomainAddress):
            return self.strand == other.strand and self.domain_idx == other.domain_idx
        return False

    def __str__(self) -> str:
        return f'{{strand: {self.strand}, domain_idx: {self.domain_idx}}}'

    def __repr__(self) -> str:
        return self.__str__() + f' hash: {self.__hash__()}'


def _exterior_base_type_of_domain_3p_end(domain_addr: StrandDomainAddress,
                                         all_bound_domain_addresses: Dict[
                                             StrandDomainAddress, StrandDomainAddress]) -> BasePairType:
    """Returns the BasePairType that corresponds to the base pair that sits on the
    3' end of provided domain.

    :param domain_addr: The address of the domain that contains the interested
    :type domain_addr: StrandDomainAddress
    :param all_bound_domain_addresses: A mapping of all the domain pairs in complex
    :type all_bound_domain_addresses: Dict[StrandDomainAddress, StrandDomainAddress]
    :return: BasePairType of base pair on 3' end of domain
    :rtype: BasePairType
    """
    # Declare domain variables:
    #                              # #
    #                              |-|
    #                            ? |-| adjacent_5n_addr
    #                              |-|
    #                              # #
    #             domain_addr      ? #        adjacent_addr
    #    #-------------------------# #-------------------------------------#
    #     |||||||||||||||||||||||||   |||||||||||||||||||||||||||||||||||||
    #    #-------------------------###-------------------------------------#
    #         complementary_addr              complementary_5n_addr
    assert domain_addr in all_bound_domain_addresses
    complementary_addr = all_bound_domain_addresses[domain_addr]
    complementary_5n_addr = complementary_addr.neighbor_5p()
    adjacent_addr: Optional[StrandDomainAddress] = None
    adjacent_5n_addr: Optional[StrandDomainAddress] = None

    # First assume BOTTOM_RIGHT_EMPTY
    #            domain_addr
    #    #-------------------------#
    #     |||||||||||||||||||||||||
    #    #-------------------------] <- Note this 3' end here
    #         complementary_addr
    adjacent_strand_type: _AdjacentDuplexType = _AdjacentDuplexType.BOTTOM_RIGHT_EMPTY

    if complementary_5n_addr is not None:
        #   Since complementary_5n_addr exists, assume BOTTOM_RIGHT_DANGLE
        #
        #            domain_addr
        #    #-------------------------#
        #     |||||||||||||||||||||||||
        #    #-------------------------###-------------------------------------#
        #     complementary_addr                complementary_5n_addr
        adjacent_strand_type = _AdjacentDuplexType.BOTTOM_RIGHT_DANGLE
        if complementary_5n_addr in all_bound_domain_addresses:
            # Since complementary_5n_addr is bound, meaning
            # adjacent_addr exist, assume TOP_RIGHT_5p
            #
            #             domain_addr                adjacent_addr
            #    #-------------------------# [-------------------------------------#
            #     |||||||||||||||||||||||||   |||||||||||||||||||||||||||||||||||||
            #    #-------------------------###-------------------------------------#
            #          complementary_addr          complementary_5n_addr
            adjacent_strand_type = _AdjacentDuplexType.TOP_RIGHT_5P
            adjacent_addr = all_bound_domain_addresses[complementary_5n_addr]
            adjacent_5n_addr = adjacent_addr.neighbor_5p()
            if adjacent_5n_addr is not None:
                # Since adjacent_5n_addr exists, assume TOP_RIGHT_OVERHANG
                #
                #                                #
                #                                |
                #                                | adjacent_5n_addr
                #                                |
                #                                #
                #             domain_addr        #       adjacent_addr
                #    #-------------------------# #-------------------------------------#
                #     |||||||||||||||||||||||||   |||||||||||||||||||||||||||||||||||||
                #    #-------------------------###-------------------------------------#
                #          complementary_addr          complementary_5n_addr
                adjacent_strand_type = _AdjacentDuplexType.TOP_RIGHT_OVERHANG
                if adjacent_5n_addr in all_bound_domain_addresses:
                    # Since adjacent_5n_addr is bound, two possible cases:

                    if domain_addr == adjacent_5n_addr:
                        # Since domain_addr and adjacent_5n_addr
                        # are the same, then this must be an internal base pair
                        #
                        #   domain_addr == adjacent_5n_addr        adjacent_addr
                        #    #-------------------------###-------------------------------------#
                        #     |||||||||||||||||||||||||   |||||||||||||||||||||||||||||||||||||
                        #    #-------------------------###-------------------------------------#
                        #        complementary_addr            complementary_5n_addr

                        # Assuming non-competitive, then this must be internal base pair or
                        # if the domain is length 2 and the 5' type is not interior, then
                        # it is adjacent to exterior base pair type
                        domain = domain_addr.strand.domains[domain_addr.domain_idx]
                        domain_next_to_interior_base_pair = (domain_addr.neighbor_5p() is not None
                                                             and complementary_addr.neighbor_3p() is not None)
                        if domain.get_length() == 2 and not domain_next_to_interior_base_pair:
                            #   domain_addr == adjacent_5n_addr        adjacent_addr
                            #     |                                       |
                            #    [--###-------------------------------------#
                            #     ||   |||||||||||||||||||||||||||||||||||||
                            #    <--###-------------------------------------#
                            #     |                              |
                            # complementary_addr       complementary_5n_addr
                            return BasePairType.ADJACENT_TO_EXTERIOR_BASE_PAIR
                        else:
                            return BasePairType.INTERIOR_TO_STRAND
                    else:
                        # Since adjacent_5n_addr does not equal domain_addr,
                        # must be a bound overhang:
                        #
                        #                              # #
                        #                              |-|
                        #                            ? |-| adjacent_5n_addr
                        #                              |-|
                        #                              # #
                        #             domain_addr      ? #        adjacent_addr
                        #    #-------------------------# #-------------------------------------#
                        #     |||||||||||||||||||||||||   |||||||||||||||||||||||||||||||||||||
                        #    #-------------------------###-------------------------------------#
                        #         complementary_addr            complementary_5n_addr
                        adjacent_strand_type = _AdjacentDuplexType.TOP_RIGHT_BOUND_OVERHANG

    domain_3n_addr = domain_addr.neighbor_3p()
    if domain_3n_addr is None:
        # domain_addr is at 3' end of strand
        #
        #            domain_addr
        #    #------------------------->
        #     |||||||||||||||||||||||||
        #    #-------------------------#
        #         complementary_addr

        if adjacent_strand_type is _AdjacentDuplexType.BOTTOM_RIGHT_EMPTY:
            #            domain_addr
            #    #------------------------->
            #     |||||||||||||||||||||||||
            #    #-------------------------]
            #         complementary_addr
            return BasePairType.BLUNT_END
        elif adjacent_strand_type is _AdjacentDuplexType.BOTTOM_RIGHT_DANGLE:
            #          domain_addr
            #    #------------------------->
            #     |||||||||||||||||||||||||
            #    #-------------------------###-------------------------------------#
            #       complementary_addr              complementary_5n_addr
            return BasePairType.DANGLE_5P
        elif adjacent_strand_type is _AdjacentDuplexType.TOP_RIGHT_5P:
            #             domain_addr                adjacent_addr
            #    #-------------------------> [-------------------------------------#
            #     |||||||||||||||||||||||||   |||||||||||||||||||||||||||||||||||||
            #    #-------------------------###-------------------------------------#
            #     complementary_addr    complementary_5n_addr
            return BasePairType.NICK_3P
        elif adjacent_strand_type is _AdjacentDuplexType.TOP_RIGHT_OVERHANG:
            #                                #
            #                                |
            #                                | adjacent_5n_addr
            #                                |
            #                                #
            #             domain_addr        #        adjacent_addr
            #    #-------------------------> #-------------------------------------#
            #     |||||||||||||||||||||||||   |||||||||||||||||||||||||||||||||||||
            #    #-------------------------###-------------------------------------#
            #     complementary_addr    complementary_5n_addr
            return BasePairType.OVERHANG_ON_ADJACENT_STRAND_3P
        elif adjacent_strand_type is _AdjacentDuplexType.TOP_RIGHT_BOUND_OVERHANG:
            #                              # #
            #                              |-|
            #                            ? |-| adjacent_5n_addr
            #                              |-|
            #                              # #
            #             domain_addr        #        adjacent_addr
            #    #-------------------------> #-------------------------------------#
            #     |||||||||||||||||||||||||   |||||||||||||||||||||||||||||||||||||
            #    #-------------------------###-------------------------------------#
            #     complementary_addr    complementary_5n_addr
            # TODO: Possible case (nick n-arm junction)
            return BasePairType.OTHER
        else:
            # Shouldn't reach here
            assert False
    else:
        # domain_addr is not the 3' end of the strand
        #
        #            domain_addr              domain_3n_addr
        #    #-------------------------##-------------------------#

        if domain_3n_addr not in all_bound_domain_addresses:
            # domain_addr's 3' neighbor is an unbound overhang
            if adjacent_strand_type is _AdjacentDuplexType.BOTTOM_RIGHT_EMPTY:
                #            domain_addr             domain_3n_addr
                #    #-------------------------##-------------------------#
                #     |||||||||||||||||||||||||
                #    #-------------------------]
                #     complementary_addr
                return BasePairType.DANGLE_3P
            elif adjacent_strand_type is _AdjacentDuplexType.BOTTOM_RIGHT_DANGLE:
                #            domain_addr                 domain_3n_addr
                #    #-------------------------##-------------------------------------#
                #     |||||||||||||||||||||||||
                #    #-------------------------##-------------------------------------#
                #          complementary_addr            complementary_5n_addr
                return BasePairType.DANGLE_5P_3P
            elif adjacent_strand_type is _AdjacentDuplexType.TOP_RIGHT_5P:
                #                              #
                #                              |
                #               domain_3n_addr |
                #                              |
                #                              #
                #             domain_addr      #         adjacent_addr
                #    #-------------------------# [-------------------------------------#
                #     |||||||||||||||||||||||||   |||||||||||||||||||||||||||||||||||||
                #    #-------------------------###-------------------------------------#
                #     complementary_addr    complementary_5n_addr
                return BasePairType.OVERHANG_ON_THIS_STRAND_3P
            elif adjacent_strand_type is _AdjacentDuplexType.TOP_RIGHT_OVERHANG:
                #                              # #
                #                              | |
                #               domain_3n_addr | | adjacent_5n_addr
                #                              | |
                #                              # #
                #             domain_addr      # #       adjacent_addr
                #    #-------------------------# #-------------------------------------#
                #     |||||||||||||||||||||||||   |||||||||||||||||||||||||||||||||||||
                #    #-------------------------###-------------------------------------#
                #     complementary_addr    complementary_5n_addr
                return BasePairType.OVERHANG_ON_BOTH_STRANDS_3P
            elif adjacent_strand_type is _AdjacentDuplexType.TOP_RIGHT_BOUND_OVERHANG:
                # TODO: Possible case (nick n-arm junction)
                #                              #                    # #
                #                              |                    |-|
                #               domain_3n_addr |                    |-| adjacent_5n_addr
                #                              |                    |-|
                #                              #                    # #
                #             domain_addr      #                      #       adjacent_addr
                #    #-------------------------#                      #-------------------------------------#
                #     |||||||||||||||||||||||||                        |||||||||||||||||||||||||||||||||||||
                #    #-------------------------########################-------------------------------------#
                #     complementary_addr                         complementary_5n_addr
                return BasePairType.OTHER
            else:
                # Shouldn't reach here
                assert False
        else:
            # domain_addr's 3' neighbor is a bound domain

            # Techinically, could be an interior base, but we should have caught this earlier
            # back when we were determining AdjacentDuplexType.
            #
            # Assertion checks that it is not an internal base
            #
            #             domain_addr            domain_3n_addr == adjacent_addr
            #    #-------------------------###-------------------------------------#
            #     |||||||||||||||||||||||||   |||||||||||||||||||||||||||||||||||||
            #    #-------------------------###-------------------------------------#
            #        complementary_addr              complementary_5n_addr
            assert domain_3n_addr != adjacent_addr

            # Declare new variables:
            domain_3n_complementary_addr = all_bound_domain_addresses[domain_3n_addr]
            domain_3n_complementary_3n_addr = domain_3n_complementary_addr.neighbor_3p()
            #             domain_addr                 domain_3n_addr
            #    #-------------------------###-------------------------------------#
            #     |||||||||||||||||||||||||   |||||||||||||||||||||||||||||||||||||
            #    #-------------------------# #-------------------------------------#
            #        complementary_addr      # domain_3n_complementary_addr
            #                                #
            #                                |
            #                                | domain_3n_complementary_3n_addr
            #                                |
            #                                #

            # Three cases:
            #
            # domain_3n_complementary_addr is 3' end
            #             domain_addr                 domain_3n_addr
            #    #-------------------------###-------------------------------------#
            #     |||||||||||||||||||||||||   |||||||||||||||||||||||||||||||||||||
            #    #-------------------------# <-------------------------------------#
            #         complementary_addr         domain_3n_complementary_addr
            #
            # domain_3n_complementary_3n_addr is unbound overhang
            #             domain_addr                 domain_3n_addr
            #    #-------------------------###-------------------------------------#
            #     |||||||||||||||||||||||||   |||||||||||||||||||||||||||||||||||||
            #    #-------------------------# #-------------------------------------#
            #           complementary_addr   # domain_3n_complementary_addr
            #                                #
            #                                |
            #                                | domain_3n_complementary_3n_addr
            #                                |
            #                                #
            #
            # domain_3n_complementary_3n_addr is unbound overhang
            #             domain_addr                 domain_3n_addr
            #    #-------------------------###-------------------------------------#
            #     |||||||||||||||||||||||||   |||||||||||||||||||||||||||||||||||||
            #    #-------------------------# #-------------------------------------#
            #         complementary_addr   ? #    domain_3n_complementary_addr
            #                              # #
            #                              |-|
            #                              |-| domain_3n_complementary_3n_addr
            #                              |-|
            #                              #-#
            #
            # Variable is None, False, True respectively based on cases above
            domain_3n_complementary_3n_addr_is_bound: Optional[bool] = None
            if domain_3n_complementary_3n_addr is not None:
                domain_3n_complementary_3n_addr_is_bound = \
                    domain_3n_complementary_3n_addr in all_bound_domain_addresses

            # Not an internal base pair since domain_addr's 3' neighbor is
            # bounded to a domain that is not complementary's 5' neighbor
            if adjacent_strand_type is _AdjacentDuplexType.BOTTOM_RIGHT_EMPTY:
                # NICK_5P
                #
                #             domain_addr                 domain_3n_addr
                #    #-------------------------###-------------------------------------#
                #     |||||||||||||||||||||||||   |||||||||||||||||||||||||||||||||||||
                #    #-------------------------] <-------------------------------------#
                #        complementary_addr           domain_3n_complementary_addr
                #
                #                        OR
                #
                # OVERHANG_ON_ADJACENT_STRAND_5P
                #
                #             domain_addr                 domain_3n_addr
                #    #-------------------------###-------------------------------------#
                #     |||||||||||||||||||||||||   |||||||||||||||||||||||||||||||||||||
                #    #-------------------------] #-------------------------------------#
                #          complementary_addr    #   domain_3n_complementary_addr
                #                                #
                #                                |
                #                                | domain_3n_complementary_3n_addr
                #                                |
                #                                #
                #
                #                        OR
                #
                # OTHER
                #
                #             domain_addr                 domain_3n_addr
                #    #-------------------------###-------------------------------------#
                #     |||||||||||||||||||||||||   |||||||||||||||||||||||||||||||||||||
                #    #-------------------------] #-------------------------------------#
                #           complementary_addr   # domain_3n_complementary_addr
                #                              # #
                #                              |-|
                #                              |-| domain_3n_complementary_3n_addr
                #                              |-|
                #                              # #
                if domain_3n_complementary_3n_addr_is_bound is None:
                    return BasePairType.NICK_5P
                elif domain_3n_complementary_3n_addr_is_bound is False:
                    return BasePairType.OVERHANG_ON_ADJACENT_STRAND_5P
                else:
                    return BasePairType.OTHER
            elif adjacent_strand_type is _AdjacentDuplexType.BOTTOM_RIGHT_DANGLE:
                # OVERHANG_ON_THIS_STRAND_5P
                #
                #             domain_addr               domain_3n_addr
                #    #-------------------------###-------------------------------------#
                #     |||||||||||||||||||||||||   |||||||||||||||||||||||||||||||||||||
                #    #-------------------------# #-------------------------------------#
                #          complementary_addr  #   domain_3n_complementary_addr
                #                              #
                #                              |
                #                              |
                #                              |
                #                              #
                #
                #                        OR
                #
                # OVERHANG_ON_BOTH_STRAND_5P
                #
                #             domain_addr               domain_3n_addr
                #    #-------------------------###-------------------------------------#
                #     |||||||||||||||||||||||||   |||||||||||||||||||||||||||||||||||||
                #    #-------------------------# #-------------------------------------#
                #           complementary_addr # # domain_3n_complementary_addr
                #                              # #
                #                              | |
                #                              | | domain_3n_complementary_3n_addr
                #                              | |
                #                              # #
                #
                #                        OR
                #
                #
                # OTHER
                # TODO: Possible case (nick n-arm junction)
                #
                #             domain_addr                                   domain_3n_addr
                #    #-------------------------########-------------------------------------#
                #     |||||||||||||||||||||||||        |||||||||||||||||||||||||||||||||||||
                #    #-------------------------#      #-------------------------------------#
                #           complementary_addr #      # domain_3n_complementary_addr
                #                              #    # #
                #                              |    |-|
                #                              |    |-| domain_3n_complementary_3n_addr
                #                              |    |-|
                #                              #    # #

                if domain_3n_complementary_3n_addr_is_bound is None:
                    return BasePairType.OVERHANG_ON_THIS_STRAND_5P
                elif domain_3n_complementary_3n_addr_is_bound is False:
                    return BasePairType.OVERHANG_ON_BOTH_STRANDS_5P
                else:
                    return BasePairType.OTHER
            elif adjacent_strand_type is _AdjacentDuplexType.TOP_RIGHT_5P:
                # TODO: Possible case (nick n-arm junction)
                # TODO: Bound DANGLE_5P_3P? or OTHER?
                #                              # #
                #                              |-|
                #               domain_3n_addr |-| domain_3n_complementary_addr
                #                              |-|
                #                              # v
                #             domain_addr      #           adjacent_addr
                #    #-------------------------# [-------------------------------------#
                #     |||||||||||||||||||||||||   |||||||||||||||||||||||||||||||||||||
                #    #-------------------------###-------------------------------------#
                #     complementary_addr                 complementary_5n_addr
                #
                #
                #
                #                              # #
                #                              |-|
                #               domain_3n_addr |-| domain_3n_complementary_addr
                #                              |-|
                #                              # ##---------#
                #             domain_addr      #       adjacent_addr
                #    #-------------------------# [-------------------------------------#
                #     |||||||||||||||||||||||||   |||||||||||||||||||||||||||||||||||||
                #    #-------------------------###-------------------------------------#
                #     complementary_addr                                  complementary_5n_addr
                #
                #
                #
                #
                #                              # #
                #                              |-|
                #               domain_3n_addr |-| domain_3n_complementary_addr
                #                              |-|
                #                              # ##---------#
                #             domain_addr      #   |||||||||                     adjacent_addr
                #    #-------------------------#  #---------#         [-------------------------------------#
                #     |||||||||||||||||||||||||                        |||||||||||||||||||||||||||||||||||||
                #    #-------------------------########################-------------------------------------#
                #     complementary_addr                                     complementary_5n_addr
                return BasePairType.OTHER
            elif adjacent_strand_type is _AdjacentDuplexType.TOP_RIGHT_OVERHANG:
                # TODO: Possible case (nick n-arm junction)
                # Bound DANGLE_5P_3P?
                #                              # #                    #
                #                              |-|                    |
                #               domain_3n_addr |-|                    | adjacent_5n_addr
                #                              |-|                    |
                #                              # v                    #
                #             domain_addr      #                      #       adjacent_addr
                #    #-------------------------#                      #-------------------------------------#
                #     |||||||||||||||||||||||||                        |||||||||||||||||||||||||||||||||||||
                #    #-------------------------########################-------------------------------------#
                #     complementary_addr                         complementary_5n_addr
                #
                #
                #
                #                              # #                    #
                #                              |-|                    |
                #               domain_3n_addr |-|                    | adjacent_5n_addr
                #                              |-|                    |
                #                              # ##---------#         #
                #             domain_addr      #                      #       adjacent_addr
                #    #-------------------------#                      #-------------------------------------#
                #     |||||||||||||||||||||||||                        |||||||||||||||||||||||||||||||||||||
                #    #-------------------------########################-------------------------------------#
                #     complementary_addr                         complementary_5n_addr
                #
                #                              # #                    #
                #                              |-|                    |
                #               domain_3n_addr |-|                    | adjacent_5n_addr
                #                              |-|                    |
                #                              # ##---------#         #
                #             domain_addr      #   |||||||||          #       adjacent_addr
                #    #-------------------------#  #---------#         #-------------------------------------#
                #     |||||||||||||||||||||||||                        |||||||||||||||||||||||||||||||||||||
                #    #-------------------------########################-------------------------------------#
                #     complementary_addr                         complementary_5n_addr
                return BasePairType.OTHER
            elif adjacent_strand_type is _AdjacentDuplexType.TOP_RIGHT_BOUND_OVERHANG:
                #                              # #                  # #
                #                              |-|                  |-|
                #               domain_3n_addr |-|                  |-| adjacent_5n_addr
                #                              |-|                  |-|
                #                              # #                  # #
                #             domain_addr      #                      #       adjacent_addr
                #    #-------------------------#                      #-------------------------------------#
                #     |||||||||||||||||||||||||                        |||||||||||||||||||||||||||||||||||||
                #    #-------------------------########################-------------------------------------#
                #     complementary_addr                         complementary_5n_addr
                assert adjacent_5n_addr is not None
                if domain_3n_addr == all_bound_domain_addresses[adjacent_5n_addr]:
                    #                              # #
                    #                              |-|
                    #               domain_3n_addr |-| adjacent_5n_addr
                    #                              |-|
                    #                              # #
                    #             domain_addr      # #       adjacent_addr
                    #    #-------------------------# #-------------------------------------#
                    #     |||||||||||||||||||||||||   |||||||||||||||||||||||||||||||||||||
                    #    #-------------------------###-------------------------------------#
                    #     complementary_addr    complementary_5n_addr
                    return BasePairType.THREE_ARM_JUNCTION
                else:
                    # Could possibly be n-arm junction
                    return BasePairType.OTHER
            else:
                # Should not make it here
                assert False


@dataclass(frozen=True)
class _BasePairDomainEndpoint:
    """A base pair endpoint in the context of the domain it resides on.

    Numbering the bases in strand complex order, of the two bound domains,
    domain1 is defined to be the domain that occurs earlier and domain2 is
    defined to be the domain that occurs later.

    ```(this line is to avoid Python syntax highlighting of ASCII art below)
                 domain1_5p_index
                 |
    domain1   5' --------------------------------- 3'
                 | | | | | | | | | | | | | | | | |
    domain2   3' --------------------------------- 5'
              ^  |                                 ^
              |  domain2_3p_index                  |
              |                                    |
              |                                    |
    domain1_5p_domain2_3p_exterior_base_pair_type  |
                                                   |
                            domain1_3p_domain2_5p_exterior_base_pair_type
    ```
    """
    domain1_5p_index: int
    domain2_3p_index: int
    domain_base_length: int
    domain1_5p_domain2_base_pair_type: BasePairType
    domain1_3p_domain1_base_pair_type: BasePairType


@dataclass(frozen=True)
class _BasePair:
    base_index1: int
    base_index2: int
    base_pairing_probability: float
    base_pair_type: BasePairType


BaseAddress = Union[int, Tuple[StrandDomainAddress, int]]
"""Represents a reference to a base. Can be either specified as a NUPACK base
index or an index of a dsd :py:class:`StrandDomainAddress`:
"""
BasePairAddress = Tuple[BaseAddress, BaseAddress]
"""Represents a reference to a base pair
"""
BoundDomains = Tuple[StrandDomainAddress, StrandDomainAddress]
"""Represents bound domains
"""


def _get_implicitly_bound_domain_addresses(strand_complex: Iterable[Strand],
                                           nonimplicit_base_pairs_domain_names: Optional[Set[str]] = None) \
        -> Dict[StrandDomainAddress, StrandDomainAddress]:
    """Returns a map of all the implicitly bound domain addresses

    :param strand_complex: Tuple of strands representing strand complex
    :type strand_complex: Complex
    :param nonimplicit_base_pairs_domain_names:
        Set of all domain names to ignore in this search, defaults to None
    :type nonimplicit_base_pairs_domain_names: Set[str], optional
    :return: Map of all implicitly bound domain addresses
    :rtype: Dict[StrandDomainAddress, StrandDomainAddress]
    """
    if nonimplicit_base_pairs_domain_names is None:
        nonimplicit_base_pairs_domain_names = set()

    implicitly_bound_domain_addresses = {}
    implicit_seen_domains: Dict[str, StrandDomainAddress] = {}
    for strand in strand_complex:
        for domain_idx, domain in enumerate(strand.domains):
            # Get domain_name
            domain_name = domain.name
            if domain_idx in strand.starred_domain_indices:
                domain_name = domain.starred_name

            # Move on to next domain if it was paired via nonimplicit_base_pairs
            if domain_name in nonimplicit_base_pairs_domain_names:
                continue

            # populate implicit bounded_domains
            strand_domain_address = StrandDomainAddress(strand, domain_idx)
            # Assertions checks that domain_name was not previously seen.
            # This is to check that the non-competition requirement on
            # implicit domains was properly checked earlier in input validation.
            implicit_seen_domains[domain_name] = strand_domain_address

            complementary_domain_name = Domain.complementary_domain_name(domain_name)
            if complementary_domain_name in implicit_seen_domains:
                complementary_strand_domain_address = implicit_seen_domains[complementary_domain_name]
                implicitly_bound_domain_addresses[strand_domain_address] = complementary_strand_domain_address
                implicitly_bound_domain_addresses[complementary_strand_domain_address] = strand_domain_address

    return implicitly_bound_domain_addresses


def _get_addr_to_starting_base_pair_idx(strand_complex: Complex) -> Dict[StrandDomainAddress, int]:
    """Returns a mapping between StrandDomainAddress and the base index of the
    5' end base of the domain

    :param strand_complex: Tuple of strands representing strand complex
    :type strand_complex: Complex
    :return: Map of StrandDomainAddress to starting base index
    :rtype: Dict[StrandDomainAddress, int]
    """
    # Fill addr_to_starting_base_pair_idx and all_bound_domain_addresses
    addr_to_starting_base_pair_idx = {}
    domain_base_index = 0
    for strand in strand_complex:
        for domain_idx, domain in enumerate(strand.domains):
            addr_to_starting_base_pair_idx[StrandDomainAddress(strand, domain_idx)] = domain_base_index
            domain_base_index += domain.get_length()

    return addr_to_starting_base_pair_idx


def _leafify_domain(domain: Domain) -> List[Domain]:
    """Returns the list of all leaf subdomains that make up domain

    :param domain: Domain
    :type domain: Domain
    :return: List of leaf subdomains
    :rtype: List[Domain]
    """
    if len(domain.subdomains) == 0:
        return [domain]
    else:
        ret = []
        for sd in domain.subdomains:
            ret += _leafify_domain(sd)
        return ret


def _leafify_strand(
        strand: Strand,
        addr_translation_table: Dict[StrandDomainAddress, List[StrandDomainAddress]]) -> Strand:
    """Create a new strand that is made of the leaf subdomains. Also updates an
    addr_translation_table which maps StrandDomainAddress from old strand to new
    strand. Since a domain may consist of multiple subdomains, a single StrandDomainAddress
    may map to a list of StrandDomainAddresses, listed in 5' to 3' order.

    :param strand: Strand
    :type strand: Strand
    :param addr_translation_table: Maps old StrandDomainAddress to new StrandDomainAddress
    :type addr_translation_table: Dict[StrandDomainAddress, List[StrandDomainAddress]]
    :return: Leafified strand
    :rtype: Strand
    """
    leafify_domains: List[List[Domain]] = [_leafify_domain(d) for d in strand.domains]
    new_domains: List[Domain] = []
    new_starred_domain_indices: List[int] = []
    new_starred_domain_idx = 0
    addr_translation_table_without_strand: Dict[int, List[int]] = {}
    for (idx, leaf_domain_list) in enumerate(leafify_domains):
        new_domain_indices = []
        for i in range(new_starred_domain_idx, new_starred_domain_idx + len(leaf_domain_list)):
            new_domain_indices.append(i)

        addr_translation_table_without_strand[idx] = new_domain_indices
        if idx in strand.starred_domain_indices:
            new_domains.extend(reversed(leaf_domain_list))
            # Star every single subdomain that made up original starred domain
            new_starred_domain_indices.extend(new_domain_indices)
        else:
            new_domains.extend(leaf_domain_list)

        new_starred_domain_idx += len(leaf_domain_list)
    new_strand: Strand = Strand(domains=new_domains, starred_domain_indices=new_starred_domain_indices,
                                name=f"leafifed {strand.name}")
    for idx, new_idxs in addr_translation_table_without_strand.items():
        new_addrs = [StrandDomainAddress(new_strand, new_idx) for new_idx in new_idxs]
        addr_translation_table[StrandDomainAddress(strand, idx)] = new_addrs

    new_strand.compute_derived_fields()

    return new_strand


def _get_base_pair_domain_endpoints_to_check(
        strand_complex: Iterable[Strand],
        nonimplicit_base_pairs: Iterable[BoundDomains] = None) -> Set[_BasePairDomainEndpoint]:
    """Returns the set of all the _BasePairDomainEndpoint to check

    :param strand_complex: Tuple of strands representing strand complex
    :type strand_complex: Complex
    :param nonimplicit_base_pairs:
        Set of base pairs that cannot be inferred (usually due to competition), defaults to None
    :type nonimplicit_base_pairs: Iterable[BoundDomains], optional
    :raises ValueError: If there are multiple instances of the same strand in a complex
    :raises ValueError: If competitive domains are not specificed in nonimplicit_base_pairs
    :raises ValueError: If address given in nonimplicit_base_pairs is not found
    :return: Set of all the _BasePairDomainEndpoint to check
    :rtype: Set[_BasePairDomainEndpoint]
    """
    addr_translation_table: Dict[StrandDomainAddress, List[StrandDomainAddress]] = {}

    # Need to convert strands into strands lowest level subdomains
    leafify_strand_complex = Complex(
        *[_leafify_strand(strand, addr_translation_table) for strand in strand_complex])

    new_nonimplicit_base_pairs = []
    if nonimplicit_base_pairs:
        for bp in nonimplicit_base_pairs:
            (addr1, addr2) = bp
            new_addr1_list = addr_translation_table[addr1]
            new_addr2_list = list(reversed(addr_translation_table[addr2]))

            assert len(new_addr1_list) == len(new_addr2_list)
            for idx in range(len(new_addr1_list)):
                new_nonimplicit_base_pairs.append((new_addr1_list[idx], new_addr2_list[idx]))

    return __get_base_pair_domain_endpoints_to_check(
        leafify_strand_complex, nonimplicit_base_pairs=new_nonimplicit_base_pairs)


def __get_base_pair_domain_endpoints_to_check(
        strand_complex: Complex,
        nonimplicit_base_pairs: Iterable[BoundDomains] = None) -> Set[_BasePairDomainEndpoint]:
    """Returns the set of all the _BasePairDomainEndpoint to check

    :param strand_complex: Tuple of strands representing strand complex
    :type strand_complex: Complex
    :param nonimplicit_base_pairs:
        Set of base pairs that cannot be inferred (usually due to competition), defaults to None
    :type nonimplicit_base_pairs: Iterable[BoundDomains], optional
    :raises ValueError: If there are multiple instances of the same strand in a complex
    :raises ValueError: If competitive domains are not specificed in nonimplicit_base_pairs
    :raises ValueError: If address given in nonimplicit_base_pairs is not found
    :return: Set of all the _BasePairDomainEndpoint to check
    :rtype: Set[_BasePairDomainEndpoint]
    """
    # Maps domain pairs
    all_bound_domain_addresses: Dict[StrandDomainAddress, StrandDomainAddress] = {}

    # Keep track of all the domain names that are provided as
    # part of a nonimplicit_base_pair so that input validation
    # knows to ignore these domain names.
    nonimplicit_base_pairs_domain_names: Set[str] = set()

    if nonimplicit_base_pairs is not None:
        for (addr1, addr2) in nonimplicit_base_pairs:
            d1 = addr1.strand.domains[addr1.domain_idx]
            d2 = addr2.strand.domains[addr2.domain_idx]
            if d1 is not d2:
                print('WARNING: a base pair is specified between two different domain objects')
            nonimplicit_base_pairs_domain_names.add(d1.get_name(starred=False))
            nonimplicit_base_pairs_domain_names.add(d1.get_name(starred=True))
            nonimplicit_base_pairs_domain_names.add(d2.get_name(starred=False))
            nonimplicit_base_pairs_domain_names.add(d2.get_name(starred=True))

            all_bound_domain_addresses[addr1] = addr2
            all_bound_domain_addresses[addr2] = addr1

    # Input validation checks:
    #
    # No repeated strand
    #
    # No competition:
    #   check no "competition" between domain (at most one
    #   starred domain for every domain, unless given as nonimplicit_base_pair)
    #   Count number of occuruences of each domain
    seen_strands: Set[Strand] = set()
    domain_counts: Dict[str, int] = defaultdict(int)
    for strand in strand_complex:
        if strand in seen_strands:
            raise ValueError(f"Multiple instances of a strand in a complex is not allowed."
                             " Please make a separate Strand object with the"
                             " same Domain objects in the same order"
                             " but a different strand name")
        seen_strands.add(strand)
        for idx, domain in enumerate(strand.domains):
            is_starred = idx in strand.starred_domain_indices
            domain_name = domain.get_name(is_starred)
            if domain_name not in nonimplicit_base_pairs_domain_names:
                domain_counts[domain_name] += 1

    # Check final counts of each domain for competition
    for domain_name in domain_counts:
        domain_name_complement = Domain.complementary_domain_name(domain_name)
        if domain_name_complement in domain_counts and domain_counts[domain_name_complement] > 1:
            assert domain_name not in nonimplicit_base_pairs_domain_names
            raise ValueError(
                f"Multiple instances of domain in a complex is not allowed "
                f"when its complement is also in the complex. "
                f"Violating domain: {domain_name_complement}")
    # End Input Validation #

    addr_to_starting_base_pair_idx: Dict[StrandDomainAddress,
                                         int] = _get_addr_to_starting_base_pair_idx(strand_complex)
    all_bound_domain_addresses.update(_get_implicitly_bound_domain_addresses(
        strand_complex, nonimplicit_base_pairs_domain_names))

    # Set of all bound domain endpoints to check.
    base_pair_domain_endpoints_to_check: Set[_BasePairDomainEndpoint] = set()

    for (domain_addr, comple_addr) in all_bound_domain_addresses.items():
        domain_base_length = domain_addr.domain().get_length()
        assert domain_base_length == comple_addr.domain().get_length()

        if domain_addr not in addr_to_starting_base_pair_idx:
            if domain_addr.domain().name in nonimplicit_base_pairs_domain_names:
                raise ValueError(f'StrandDomainAddress {domain_addr} is not found in given complex')
            else:
                print(f'StrandDomainAddress {domain_addr} is not found in given complex')
                assert False

        if comple_addr not in addr_to_starting_base_pair_idx:
            if comple_addr.domain().name in nonimplicit_base_pairs_domain_names:
                raise ValueError(f'StrandDomainAddress {comple_addr} is not found in given complex')
            else:
                print(f'StrandDomainAddress {comple_addr} is not found in given complex')
                assert False

        domain_5p = addr_to_starting_base_pair_idx[domain_addr]
        comple_5p = addr_to_starting_base_pair_idx[comple_addr]

        # Define domain1 to be the "earlier" domain
        if domain_5p < comple_5p:
            domain1_addr = domain_addr
            domain1_5p = domain_5p

            domain2_addr = comple_addr
            domain2_5p = comple_5p
        else:
            domain1_addr = comple_addr
            domain1_5p = comple_5p

            domain2_addr = domain_addr
            domain2_5p = domain_5p

        domain2_3p = domain2_5p + domain_base_length - 1

        # domain1                     5' --------------------------------- 3'
        #                                | | | | | | | | | | | | | | | | |
        # domain2                     3' --------------------------------- 5'
        #                             ^                                    ^
        #                             |                                    |
        #                   d1_5p_d2_3p_ext_bp_type                        |
        #                                                                  |
        #                                                       d1_3p_d2_5p_ext_bp_type
        d1_3p_d2_5p_ext_bp_type = _exterior_base_type_of_domain_3p_end(domain1_addr,
                                                                       all_bound_domain_addresses)
        d1_5p_d2_3p_ext_bp_type = _exterior_base_type_of_domain_3p_end(domain2_addr,
                                                                       all_bound_domain_addresses)

        base_pair_domain_endpoints_to_check.add(_BasePairDomainEndpoint(
            domain1_5p, domain2_3p, domain_base_length, d1_5p_d2_3p_ext_bp_type, d1_3p_d2_5p_ext_bp_type))

    return base_pair_domain_endpoints_to_check


def nupack_complex_base_pair_probability_constraint(
        strand_complexes: List[Complex],
        nonimplicit_base_pairs: Optional[Iterable[BoundDomains]] = None,
        all_base_pairs: Optional[Iterable[BoundDomains]] = None,
        base_pair_prob_by_type: Optional[Dict[BasePairType, float]] = None,
        base_pair_prob_by_type_upper_bound: Dict[BasePairType, float] = None,
        base_pair_prob: Dict[BasePairAddress, float] = None,
        base_unpaired_prob: Dict[BaseAddress, float] = None,
        base_pair_prob_upper_bound: Dict[BasePairAddress, float] = None,
        base_unpaired_prob_upper_bound: Dict[BaseAddress, float] = None,
        temperature: float = dv.default_temperature,
        sodium: float = dv.default_sodium,
        magnesium: float = dv.default_magnesium,
        weight: float = 1.0,
        score_transfer_function: Callable[[float], float] = default_score_transfer_function,
        description: Optional[str] = None,
        short_description: str = 'ComplexBPProbs',
        parallel: bool = False,
) -> ComplexConstraint:
    """Returns constraint that checks given base pairs probabilities in tuples of :any:`Strand`'s

    :param strand_complexes:
        Iterable of :any:`Strand` tuples
    :type strand_complexes:
        List[Complex]
    :param nonimplicit_base_pairs:
        List of nonimplicit base pairs that cannot be inferred because multiple
        instances of the same :py:class:`Domain` exist in complex.

        The :py:attr:`StrandDomainAddress.strand` field of each address should
        reference a strand in the first complex in ``strand_complexes``.

        For example,
        if one :py:class:`Strand` has one T :py:class:`Domain` and another
        strand in the complex has two T* :py:class:`Domain` s, then the intended
        binding graph cannot be inferred and must be stated explicitly in this
        field.
    :type nonimplicit_base_pairs:
        Optional[Iterable[BoundDomains]]
    :param all_base_pairs:
        List of all base pairs in complex. If not provided, then base pairs are
        infered based on the name of :py:class:`Domain` s in the complex as well
        as base pairs specified in ``nonimplicit_base_pairs``.


        **TODO**: This has not been implemented yet, and the behavior is as if this
        parameter is always ``None`` (binding graph is always inferred).
    :type all_base_pairs:
        Optional[Iterable[BoundDomains]]
    :param base_pair_prob_by_type:
        Probability lower bounds for each :py:class:`BasePairType`.
        All :py:class:`BasePairType` comes with a default
        such as :py:data:`default_interior_to_strand_probability` which will be
        used if a lower bound is not specified for a particular type.

        **Note**: Despite the name of this parameter, set thresholds for unpaired
        bases by specifying a threshold for :py:attr:`BasePairType.UNPAIRED`.
    :type base_pair_prob_by_type:
        Optional[Dict[BasePairType, float]]
    :param base_pair_prob_by_type_upper_bound:
        Probability upper bounds for each :py:class:`BasePairType`.
        By default, no upper bound is set.

        **Note**: Despite the name of this parameter, set thresholds for unpaired
        bases by specifying a threshold for :py:attr:`BasePairType.UNPAIRED`.

        **TODO**: This has not been implemented yet.
    :type base_pair_prob_by_type_upper_bound:
        Dict[BasePairType, float], optional
    :param base_pair_prob:
        Probability lower bounds for each :py:class:`BasePairAddress` which takes
        precedence over probabilities specified by ``base_pair_prob_by_type``.

        **TODO**: This has not been implemented yet.
    :type base_pair_prob:
        Optional[Dict[BasePairAddress, float]]
    :param base_unpaired_prob:
        Probability lower bounds for each :py:class:`BaseAddress` representing
        unpaired bases. These lower bounds take precedence over the probability
        specified by ``base_pair_prob_by_type[BasePairType.UNPAIRED]``.
    :type base_unpaired_prob:
        Optional[Dict[BaseAddress, float]]
    :param base_pair_prob_upper_bound:
        Probability upper bounds for each :py:class`BasePairAddress` which takes
        precedence over probabilties specified by ``base_pair_prob_by_type_upper_bound``.
    :type base_pair_prob_upper_bound:
        Optional[Dict[BasePairAddress, float]]
    :param base_unpaired_prob_upper_bound:
        Probability upper bounds for each :py:class:`BaseAddress` representing
        unpaired bases. These lower bounds take precedence over the probability
        specified by ``base_pair_prob_by_type_upper_bound[BasePairType.UNPAIRED]``.
    :type base_unpaired_prob_upper_bound:
        Optional[Dict[BaseAddress, float]]
    :param temperature:
        Temperature specified in °C, defaults to :py:data:`vienna_nupack.default_temperature`.
    :type temperature: float, optional
    :param sodium:
        molarity of sodium (more generally, monovalent ions such as Na+, K+, NH4+)
        in moles per liter
    :param magnesium:
        molarity of magnesium (Mg++) in moles per liter
    :param weight:
        See :py:data:`Constraint.weight`, defaults to 1.0
    :type weight:
        float, optional
    :param score_transfer_function:
        Score transfer function to use. By default, f(x) = x**2 is used, where x
        is the sum of the squared errors of each base pair that violates the
        threshold.
    :type score_transfer_function: Callable[[float], float], optional
    :param description:
        See :py:data:`Constraint.description`, defaults to None
    :type description:
        Optional[str], optional
    :param short_description:
        See :py:data:`Constraint.short_description` defaults to 'complex_secondary_structure_nupack'
    :type short_description:
        str, optional
    :param parallel:
        **TODO**: Implement this
    :type parallel:
        bool, optional
    :raises ImportError:
        If NUPACK 4 is not installed.
    :raises ValueError:
        If ``strand_complexes`` is not valid. In order for ``strand_complexes`` to
        be valid, ``strand_complexes`` must:

        * Consist of complexes (tuples of :py:class:`Strand` objects)
        * Each complex must be of the same motif

            * Same number of :py:class:`Strand` s in each complex
            * Same number of :py:class:`Domain` s in each :py:class:`Strand`
            * Same number of bases in each :py:class:`Domain`
    :return: ComplexConstraint
    :rtype: ComplexConstraint
    """
    _check_nupack_installed()

    # Start Input Validation
    if len(strand_complexes) == 0:
        raise ValueError("strand_complexes list cannot be empty.")

    strand_complex_template = strand_complexes[0]

    if not isinstance(strand_complex_template, Complex):
        raise ValueError(
            f"First element in strand_complexes was not a Complex of Strands. "
            f"Please provide a Complex of Strands.")

    for strand in strand_complex_template:
        if type(strand) is not Strand:
            raise ValueError(f"Complex at index 0 contained non-Strand object: {type(strand)}")

    for strand_complex in strand_complexes:
        for strand in strand_complex:
            for domain in strand.domains:
                if not domain.has_length():
                    raise ValueError(f'''\
Domain {domain.name} has no length yet. To use 
nupack_complex_secondary_structure_constraint, each Domain must have a length 
assigned, either by assigning it a DomainPool first, or by setting the Domain 
to have a fixed DNA sequence by calling domain.set_fixed_sequence.''')

    for idx in range(1, len(strand_complexes)):
        strand_complex = strand_complexes[idx]
        if not isinstance(strand_complex, Complex):
            raise ValueError(
                f"Element {strand_complex} at index {idx} is not a Complex of Strands. "
                f"Please provide a Complex of Strands.")
        if len(strand_complex) != len(strand_complex_template):
            raise ValueError(
                f"Inconsistent complex structures: Complex at index {idx} contained {len(strand_complex)} "
                f"strands, but complex at index 0 contained {len(strand_complex_template)} strands.")
        for s in range(len(strand_complex)):
            other_strand: Strand = strand_complex[s]
            template_strand: Strand = strand_complex_template[s]
            if type(other_strand) is not Strand:
                raise ValueError(
                    f"Complex at index {idx} contained non-Strand object at index {s}: {type(other_strand)}")
            if len(other_strand.domains) != len(template_strand.domains):
                raise ValueError(
                    f"Strand {other_strand} (index {s} of strand_complexes at index {idx}) does not match "
                    f"the provided template ({template_strand}). "
                    f"Strand {other_strand} contains {len(other_strand.domains)} domains but template "
                    f"strand {template_strand} contains {len(template_strand.domains)} domains.")
            for d in range(1, len(other_strand.domains)):
                domain_length: int = other_strand.domains[d].get_length()
                template_domain_length: int = template_strand.domains[d].get_length()
                if domain_length != template_domain_length:
                    raise ValueError(
                        f"Strand {other_strand} (the strand at index {s} of the complex located at index "
                        f"{idx} of strand_complexes) does not match the "
                        f"provided template ({template_strand}): domain at index {d} is length "
                        f"{domain_length}, but expected {template_domain_length}.")

    base_pair_domain_endpoints_to_check = _get_base_pair_domain_endpoints_to_check(
        strand_complex_template, nonimplicit_base_pairs=nonimplicit_base_pairs)

    # Start populating base_pair_probs
    base_type_probability_threshold: Dict[BasePairType, float] = (
        {} if base_pair_prob_by_type is None else base_pair_prob_by_type.copy())
    for base_type in BasePairType:
        if base_type not in base_type_probability_threshold:
            base_type_probability_threshold[base_type] = base_type.default_pair_probability()
    # End populating base_pair_probs

    if description is None:
        description = 'Base pair probability of complex'

    def evaluate(seqs: Tuple[str, ...], strand_complex_: Complex) -> Tuple[float, str]:
        assert len(seqs) == len(strand_complex)
        bps = _violation_base_pairs(strand_complex_)
        err_sq = 0.0
        # eval
        for bp in bps:
            e = base_type_probability_threshold[bp.base_pair_type] - bp.base_pairing_probability
            assert e > 0
            err_sq += e ** 2
        # summary
        if len(bps) == 0:
            summary = "\tAll base pairs satisfy thresholds."
        else:
            summary_list = []
            for bp in bps:
                i = bp.base_index1
                j = bp.base_index2
                p = bp.base_pairing_probability
                t = bp.base_pair_type
                summary_list.append(
                    f'\t{i},{j}: {math.floor(100 * p)}% '
                    f'(<{round(100 * base_type_probability_threshold[t])}% [{t}])')
            summary = '\n'.join(summary_list)
        return err_sq, summary

    def _violation_base_pairs(strand_complex_: Complex) -> List[_BasePair]:
        nupack_complex_result = dv.nupack_complex_base_pair_probabilities(strand_complex_,
                                                                          temperature=temperature,
                                                                          sodium=sodium, magnesium=magnesium)

        # DEBUG: Print out result matrix
        # for r in nupack_complex_result:
        #     for c in r:
        #         print("{:.2f}".format(c), end=' ')
        #     print()

        # DEBUG: Print out complex strands and sequences
        # for strand in strand_complex:
        #     print(f'{strand.name}: {strand.sequence()}')

        # Refactor all this into a function that returns all the base pairs that are below threshold
        # eval would take the squared sum of prob differences

        # Probability threshold
        internal_base_pair_prob = base_type_probability_threshold[BasePairType.INTERIOR_TO_STRAND]
        unpaired_base_prob = base_type_probability_threshold[BasePairType.UNPAIRED]
        border_internal_base_pair_prob = base_type_probability_threshold[
            BasePairType.ADJACENT_TO_EXTERIOR_BASE_PAIR]

        # Tracks which bases are paired. Used to determine unpaired bases.
        expected_paired_idxs: Set[int] = set()

        # Collect violating base pairs
        bps: List[_BasePair] = []
        for e in base_pair_domain_endpoints_to_check:
            domain1_5p = e.domain1_5p_index
            domain2_3p = e.domain2_3p_index
            domain_length_ = e.domain_base_length
            d1_5p_d2_3p_ext_bp_type = e.domain1_5p_domain2_base_pair_type
            d1_3p_d2_5p_ext_bp_type = e.domain1_3p_domain1_base_pair_type

            # Checks if base pairs at ends of domain to be above 40% probability
            domain1_3p = domain1_5p + (domain_length_ - 1)
            domain2_5p = domain2_3p - (domain_length_ - 1)

            d1_5p_d2_3p_ext_bp_prob_thres = base_type_probability_threshold[d1_5p_d2_3p_ext_bp_type]
            if nupack_complex_result[domain1_5p][domain2_3p] < d1_5p_d2_3p_ext_bp_prob_thres:
                bps.append(
                    _BasePair(
                        domain1_5p, domain2_3p, nupack_complex_result[domain1_5p][domain2_3p],
                        d1_5p_d2_3p_ext_bp_type))
            expected_paired_idxs.add(domain1_5p)
            expected_paired_idxs.add(domain2_3p)

            d1_3p_d2_5p_ext_bp_prob_thres = base_type_probability_threshold[d1_3p_d2_5p_ext_bp_type]
            if nupack_complex_result[domain1_3p][domain2_5p] < d1_3p_d2_5p_ext_bp_prob_thres:
                bps.append(
                    _BasePair(
                        domain1_3p, domain2_5p, nupack_complex_result[domain1_3p][domain2_5p],
                        d1_3p_d2_5p_ext_bp_type))
            expected_paired_idxs.add(domain1_3p)
            expected_paired_idxs.add(domain2_5p)
            # Check if base pairs interior to domain (note ascending base pair indices
            # for domain1 and descending base pair indices for domain2)
            #
            # Ex:
            #     0123
            #    [AGCT>    domain1
            #          \
            #          |
            #          /
            #    <TCGA]    domain2
            #     7654
            #
            # TODO: Rewrite this loop using numpy
            # domain1_idxs = np.arange(domain1_5p + 1, domain1_5p + domain_length - 1)
            # domain2_idxs = np.arange(domain2_3p - 1, ,-1)
            for i in range(1, domain_length_ - 1):
                row = domain1_5p + i
                col = domain2_3p - i

                # Determine if base pair is adjacent to exterior base pair
                prob_thres = internal_base_pair_prob
                bp_type = BasePairType.INTERIOR_TO_STRAND
                if i == 1 and d1_5p_d2_3p_ext_bp_type is not BasePairType.INTERIOR_TO_STRAND \
                        or i == domain_length_ - 2 \
                        and d1_3p_d2_5p_ext_bp_prob_thres is not BasePairType.INTERIOR_TO_STRAND:
                    prob_thres = border_internal_base_pair_prob
                    bp_type = BasePairType.ADJACENT_TO_EXTERIOR_BASE_PAIR

                if nupack_complex_result[row][col] < prob_thres:
                    bps.append(_BasePair(row, col, nupack_complex_result[row][col], bp_type))
                expected_paired_idxs.add(row)
                expected_paired_idxs.add(col)

        # Check base pairs that should not be paired are high probability
        for i in range(len(nupack_complex_result)):
            if i not in expected_paired_idxs and nupack_complex_result[i][i] < unpaired_base_prob:
                bps.append(_BasePair(i, i, nupack_complex_result[i][i], BasePairType.UNPAIRED))

        return bps

    return ComplexConstraint(description=description,
                             short_description=short_description,
                             weight=weight,
                             score_transfer_function=score_transfer_function,
                             parallel=parallel,
                             complexes=tuple(strand_complexes),
                             evaluate=evaluate)
