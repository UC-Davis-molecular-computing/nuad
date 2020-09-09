"""
Stochastic local search for finding DNA sequences to assign to
:any:`Domain`'s in a :any:`Design` to satisfy all :any:`Constraint`'s.
"""

import itertools
import os
import sys
import logging
import math
import pprint
from collections import Counter, defaultdict, deque
import collections.abc as abc
from dataclasses import dataclass, field
from typing import List, Tuple, Sequence, Set, FrozenSet, Optional, Dict, Callable, Iterable, Generic, Any, \
    Deque, TypeVar
import statistics
import textwrap

import numpy as np  # noqa

import dsd.np as dn

# XXX: If I understand ThreadPool versus Pool, ThreadPool will get no benefit from multiple cores,
# but Pool will. However, when I check the core usage, all of them spike when using ThreadPool, which
# is what we want (all processes going full).
# So clearly I don't understand the difference between ThreadPool and Pool.
# Actually I can't really find any official documentation of ThreadPool, though it has the same API as Pool.
# I'm using ThreadPool instead of Pool mainly because Pool
# is a pain to call; all information must be pickle-able, but the only functions that are pickle-able are
# defined at the top level of a module. The constraints call local functions defined by the user or
# by us in higher-order functions such as rna_duplex_strand_pairs_constraint, so it's not how to use Pool.
# There may also be a performance overhead for doing this pickling, but I don't know because I haven't
# tested it.
# from multiprocessing.pool import Pool
from multiprocessing.pool import ThreadPool

from dsd.constraints import Domain, Strand, Design, Constraint, DomainConstraint, StrandConstraint, \
    DomainPairConstraint, StrandPairConstraint, ConstraintWithDomainPairs, ConstraintWithStrandPairs, \
    logger, DesignPart, all_pairs, all_pairs_iterator
import dsd.constraints as dc

from dsd.stopwatch import Stopwatch

_thread_pool = ThreadPool(processes=dc.cpu_count())

log_names_of_domains_and_strands_checked = False
pprint_indent = 4


@dataclass(frozen=True)
class _Violation(Generic[DesignPart]):
    """
    Represents a violation of a single :any:`Constraint` in a :any:`Design`. The "part" of the :any:`Design`
    that violated the constraint is generic type `DesignPart` (e.g., for :any:`StrandPairConstraint`,
    DesignPart = :any:`Pair` [:any:`Strand`]).
    """

    constraint: Constraint
    """:any:`Constraint` that was violated to result in this :any:`Violation`."""

    domains: FrozenSet[Domain]  # = field(init=False, hash=False, compare=False, default=None)
    """:any:`Domain`'s that were involved in violating :py:data:`Violation.constraint`"""

    _weight: Optional[float]

    def __init__(self, constraint: Constraint, domains: Iterable[Domain], weight: Optional[float] = None):
        """
        :param constraint: :any:`Constraint` that was violated to result in this
        :param domains: :any:`Domain`'s that were involved in violating :py:data:`Violation.constraint`
        """
        object.__setattr__(self, 'constraint', constraint)
        domains_frozen = frozenset(domains)
        object.__setattr__(self, 'domains', domains_frozen)
        object.__setattr__(self, '_weight', weight)

    @property
    def weight(self) -> float:
        return self.constraint.weight if self._weight is None else self._weight


@dataclass
class _ViolationSet:
    """
    Represents violations of :any:`Constraint`'s in a :any:`Design`.

    It is designed to be efficiently updateable when a single :any:`Domain` changes, to efficiently update
    only those violations of :any:`Constraint`'s that could have been affected by the changed :any:`Domain`.
    """

    all_violations: Set[_Violation] = field(default_factory=set)
    """Set of all :any:`Violation`'s."""

    domain_to_violations: Dict[Domain, Set[_Violation]] = field(default_factory=lambda: defaultdict(set))
    """Dict mapping each :any:`Domain` to the set of all :any:`Violation`'s for which it is blamed."""

    def update(self, new_violations: Dict[Domain, Set[_Violation]]) -> None:
        """
        Update this :any:`ViolationSet` by merging in new violations from `new_violations`.

        :param new_violations: dict mapping each :any:`Domain` to the set of :any:`Violation`'s
                               for which it is blamed
        """
        for domain, domain_violations in new_violations.items():
            self.all_violations.update(domain_violations)
            self.domain_to_violations[domain].update(domain_violations)

    def clone(self) -> '_ViolationSet':
        """
        Returns a deep-ish copy of this :any:`ViolationSet`.
        :py:data:`ViolationSet.all_violations` is a new list,
        but containing the same :any:`Violation`'s.
        :py:data:`ViolationSet.domain_to_violations` is a new dict,
        and each of its values is a new set, but each of the :any:`Domain`'s and :any:`Violation`'s
        is the same object as in the original :any:`ViolationSet`.

        This is required for efficiently processing :any:`Violation`'s from one search iteration to the next.

        :return: A deep-ish copy of this :any:`ViolationSet`.
        """
        domain_to_violations_deep_copy = defaultdict(set, self.domain_to_violations)
        for domain, violations in domain_to_violations_deep_copy.items():
            domain_to_violations_deep_copy[domain] = set(violations)
        return _ViolationSet(set(self.all_violations), domain_to_violations_deep_copy)

    def remove_violations_of_domain(self, domain: Domain) -> None:
        """
        Removes any :any:`Violation`'s blamed on `domain`.
        :param domain: the :any:`Domain` whose :any:`Violation`'s should be removed
        """
        # XXX: need to make a copy of this set, since we are modifying the sets in place
        # (values in self.domain_to_violations)
        violations_of_domain = set(self.domain_to_violations[domain])
        self.all_violations -= violations_of_domain
        for violations_of_other_domain in self.domain_to_violations.values():
            violations_of_other_domain -= violations_of_domain
        assert len(self.domain_to_violations[domain]) == 0

    def total_weight(self) -> float:
        """
        :return: Total weight of all violations.
        """
        return sum(violation.weight for violation in self.all_violations)


def _violations_of_constraints(design: Design,
                               never_increase_weight: bool,
                               domain_changed: Optional[Domain],
                               violation_set_old: Optional[_ViolationSet]) -> _ViolationSet:
    """
    :param design:
        The :any:`Design` for which to find DNA sequences.
    :param domain_changed:
        The :any:`Domain` that just changed; if None, then recalculate all constraints, otherwise assume no
        constraints changed that do not involve `domain`.
    :param violation_set_old:
        :any:`ViolationSet` to update, assuming `domain_changed` is the only :any:`Domain` that changed.
    :param never_increase_weight:
        Indicates whether the search algorithm is using an update rule that never increases the total weight
        of violations (i.e., it only goes downhill). If so we can optimize and stop this function early as
        soon as we find that the violations discovered so far exceed the total weight of the current optimal
        solution. In later stages of the search, when the optimal solution so far has very few violated
        constraints, this vastly speeds up the search by allowing most of the constraint checking to be
        skipping for most choices of DNA sequences to `domain_changed`.
    :return:
        dict mapping each :any:`Domain` to the list of constraints it violated
    """

    if not ((domain_changed is None and violation_set_old is None) or (
            domain_changed is not None and violation_set_old is not None)):
        raise ValueError('domain_changed and violation_set_old should both be None or both be not None; '
                         f'domain_changed = {domain_changed}'
                         f'violation_set_old = {violation_set_old}')

    violation_set: _ViolationSet
    if domain_changed is None:
        violation_set = _ViolationSet()
    else:
        assert violation_set_old is not None
        assert not domain_changed.fixed
        violation_set = violation_set_old.clone()
        violation_set.remove_violations_of_domain(domain_changed)

    # individual domain constraints within each DomainPool
    pools_domains = design.domain_pools.items() if domain_changed is None \
        else [(domain_changed.pool, [domain_changed])]
    for domain_pool, domains_in_pool in pools_domains:
        for domain_constraint_pool in domain_pool.domain_constraints:
            domains = domains_in_pool if domain_changed is None else [domain_changed]
            domain_violations_pool = _violations_of_domain_constraint(domains=domains,
                                                                      constraint=domain_constraint_pool)
            violation_set.update(domain_violations_pool)

            if _quit_early(never_increase_weight, violation_set, violation_set_old):
                return violation_set

    # individual strand constraints within each StrandGroup
    for strand_pool, strands in design.strand_groups.items():
        for strand_constraint_pool in strand_pool.strand_constraints:
            current_weight_gap = violation_set_old.total_weight() - violation_set.total_weight() \
                if never_increase_weight and violation_set_old is not None else None
            strands = _strands_containing_domain(domain_changed, strands)
            strand_violations_pool = _violations_of_strand_constraint(strands=strands,
                                                                      constraint=strand_constraint_pool,
                                                                      current_weight_gap=current_weight_gap)
            violation_set.update(strand_violations_pool)

            if _quit_early(never_increase_weight, violation_set, violation_set_old):
                return violation_set

    # individual domain constraints across all domains in Design
    # most of the time we only check one of these, so we don't bother passing in the current weight gap
    for domain_constraint in design.domain_constraints:
        if domain_changed is None:
            domains_to_check = design.domains
        else:  # don't bother with thread pools if there's just one domain
            domains_to_check = [domain_changed]
        domain_violations = _violations_of_domain_constraint(domains=domains_to_check,
                                                             constraint=domain_constraint)
        violation_set.update(domain_violations)

        if _quit_early(never_increase_weight, violation_set, violation_set_old):
            return violation_set

    # individual strand constraints across all strands in Design
    for strand_constraint in design.strand_constraints:
        current_weight_gap = violation_set_old.total_weight() - violation_set.total_weight() \
            if never_increase_weight and violation_set_old is not None else None
        strands = _strands_containing_domain(domain_changed, design.strands)
        strand_violations = _violations_of_strand_constraint(strands=strands,
                                                             constraint=strand_constraint,
                                                             current_weight_gap=current_weight_gap)
        violation_set.update(strand_violations)

        if _quit_early(never_increase_weight, violation_set, violation_set_old):
            return violation_set

    # all pairs of domains in Design
    for domain_pair_constraint in design.domain_pair_constraints:
        current_weight_gap = violation_set_old.total_weight() - violation_set.total_weight() \
            if never_increase_weight and violation_set_old is not None else None
        domain_pair_violations = _violations_of_domain_pair_constraint(domains=design.domains,
                                                                       constraint=domain_pair_constraint,
                                                                       domain_changed=domain_changed,
                                                                       current_weight_gap=current_weight_gap)
        violation_set.update(domain_pair_violations)

        if _quit_early(never_increase_weight, violation_set, violation_set_old):
            return violation_set

    # all pairs of strands in Design
    for strand_pair_constraint in design.strand_pair_constraints:
        current_weight_gap = violation_set_old.total_weight() - violation_set.total_weight() \
            if never_increase_weight and violation_set_old is not None else None
        strand_pair_violations = _violations_of_strand_pair_constraint(strands=design.strands,
                                                                       constraint=strand_pair_constraint,
                                                                       domain_changed=domain_changed,
                                                                       current_weight_gap=current_weight_gap)
        violation_set.update(strand_pair_violations)

        if _quit_early(never_increase_weight, violation_set, violation_set_old):
            return violation_set

    # constraints that process each domain, but all at once (e.g., to hand off in batch to RNAduplex)
    for domains_constraint in design.domains_constraints:
        domains = design.domains if domain_changed is None else [domain_changed]

        if log_names_of_domains_and_strands_checked:
            logger.debug(f'$ for domains constraint {domains_constraint.description}, '
                         f'checking these domains:\n'
                         f'${pprint.pformat(domains, indent=pprint_indent)}')

        sets_of_violating_domains = domains_constraint(domains)
        domains_violations = _convert_sets_of_violating_domains_to_violations(domains_constraint,
                                                                              sets_of_violating_domains)
        violation_set.update(domains_violations)

        if _quit_early(never_increase_weight, violation_set, violation_set_old):
            return violation_set

    # constraints that process each strand, but all at once (e.g., to hand off in batch to RNAduplex)
    for strands_constraint in design.strands_constraints:
        strands = _strands_containing_domain(domain_changed, design.strands)

        if log_names_of_domains_and_strands_checked:
            logger.debug(f'$ for strands constraint {strands_constraint.description}, '
                         f'checking these strands:\n'
                         f'${pprint.pformat(strands, indent=pprint_indent)}')

        if len(strands) > 0:
            sets_of_violating_domains = strands_constraint(strands)
            domains_violations = _convert_sets_of_violating_domains_to_violations(strands_constraint,
                                                                                  sets_of_violating_domains)
            violation_set.update(domains_violations)
            if _quit_early(never_increase_weight, violation_set, violation_set_old):
                return violation_set

    # constraints that process all pairs of domains at once (e.g., to hand off in batch to RNAduplex)
    for domain_pairs_constraint in design.domain_pairs_constraints:
        domain_pairs_to_check = _determine_domain_pairs_to_check(design.domains, domain_changed,
                                                                 domain_pairs_constraint)

        if log_names_of_domains_and_strands_checked:
            logger.debug(f'$ for strand pairs constraint {domain_pairs_constraint.description}, '
                         f'checking these strand pairs:\n'
                         f'${pprint.pformat(domain_pairs_to_check, indent=pprint_indent)}')

        if len(domain_pairs_to_check) > 0:
            sets_of_violating_domains = domain_pairs_constraint(domain_pairs_to_check)
            domains_violations = _convert_sets_of_violating_domains_to_violations(domain_pairs_constraint,
                                                                                  sets_of_violating_domains)
            violation_set.update(domains_violations)
            if _quit_early(never_increase_weight, violation_set, violation_set_old):
                return violation_set

    # constraints that process all pairs of strands at once (e.g., to hand off in batch to RNAduplex)
    for strand_pairs_constraint in design.strand_pairs_constraints:
        strand_pairs_to_check = _determine_strand_pairs_to_check(design.strands, domain_changed,
                                                                 strand_pairs_constraint)

        if log_names_of_domains_and_strands_checked:
            logger.debug(f'$ for strand pairs constraint {strand_pairs_constraint.description}, '
                         f'checking these strand pairs:\n'
                         f'${pprint.pformat(strand_pairs_to_check, indent=pprint_indent)}')

        if len(strand_pairs_to_check) > 0:
            sets_of_violating_domains = strand_pairs_constraint(strand_pairs_to_check)
            domains_violations = _convert_sets_of_violating_domains_to_violations(strand_pairs_constraint,
                                                                                  sets_of_violating_domains)
            violation_set.update(domains_violations)
            if _quit_early(never_increase_weight, violation_set, violation_set_old):
                return violation_set

    # constraints that processes whole design at once (for anything not captured by the above, e.g.,
    # processing all triples of strands)
    for design_constraint in design.design_constraints:
        sets_of_violating_domains = design_constraint(design, domain_changed)
        domains_violations = _convert_sets_of_violating_domains_to_violations(design_constraint,
                                                                              sets_of_violating_domains)
        violation_set.update(domains_violations)
        if _quit_early(never_increase_weight, violation_set, violation_set_old):
            return violation_set

    return violation_set


def _quit_early(never_increase_weight: bool,
                violation_set: _ViolationSet,
                violation_set_old: Optional[_ViolationSet]) -> bool:
    return (never_increase_weight and violation_set_old is not None
            and violation_set.total_weight() > violation_set_old.total_weight())


def _at_least_one_domain_unfixed(pair: Tuple[Domain, Domain]) -> bool:
    return not (pair[0].fixed and pair[1].fixed)


def _determine_domain_pairs_to_check(all_domains: Iterable[Domain],
                                     domain_changed: Optional[Domain],
                                     constraint: ConstraintWithDomainPairs) \
        -> Sequence[Tuple[Domain, Domain]]:
    """
    Determines domain pairs to check between domains in `all_domains`.
    If `domain_changed` is None, then this is all pairs where they are not both fixed if constraint.pairs
    is None, otherwise it is constraint.pairs.
    If `domain_changed` is not None, then among those pairs specified above,
    it is all pairs where one of the two is `domain_changed`.
    """
    # either all pairs, or just constraint.pairs if specified
    domain_pairs_to_check_if_domain_changed_none = constraint.pairs if constraint.pairs is not None \
        else all_pairs_iterator(all_domains, where=_at_least_one_domain_unfixed)

    # filter out those not containing domain_change if specified
    domain_pairs_to_check = list(domain_pairs_to_check_if_domain_changed_none) if domain_changed is None \
        else [(domain1, domain2) for domain1, domain2 in domain_pairs_to_check_if_domain_changed_none
              if domain1 is domain_changed or domain2 is domain_changed]

    return domain_pairs_to_check


def _at_least_one_strand_unfixed(pair: Tuple[Strand, Strand]) -> bool:
    return not (pair[0].fixed and pair[1].fixed)


def _determine_strand_pairs_to_check(all_strands: Iterable[Strand],
                                     domain_changed: Optional[Domain],
                                     constraint: ConstraintWithStrandPairs) -> \
        Sequence[Tuple[Strand, Strand]]:
    """
    Similar to _determine_domain_pairs_to_check but for strands.
    """
    # either all pairs, or just constraint.pairs if specified
    strand_pairs_to_check_if_domain_changed_none = constraint.pairs if constraint.pairs is not None \
        else all_pairs(all_strands, where=_at_least_one_strand_unfixed)

    # filter out those not containing domain_change if specified
    strand_pairs_to_check = strand_pairs_to_check_if_domain_changed_none if domain_changed is None \
        else [(strand1, strand2) for strand1, strand2 in strand_pairs_to_check_if_domain_changed_none
              if domain_changed in strand1.domains or domain_changed in strand2.domains]

    return strand_pairs_to_check


def _strands_containing_domain(domain: Optional[Domain], strands: List[Strand]) -> List[Strand]:
    """
    :param domain: :any:`Domain` to check for, or None to return all of `strands`
    :param strands: `strands` in which to search for :any:`Strand`'s that contain `domain`
    :return: If `domain` is None, just return `strands`, otherwise return :any:`Strand`'s in `strands`
             that contain `domain`
    """
    return strands if domain is None else [strand for strand in strands if domain in strand.domains]


def _convert_sets_of_violating_domains_to_violations(constraint: Constraint,
                                                     sets_of_violating_domains: Iterable[Set[Domain]]) \
        -> Dict[Domain, Set[_Violation]]:
    domains_violations: Dict[Domain, Set[_Violation]] = defaultdict(set)
    for domain_set in sets_of_violating_domains:
        violation = _Violation(constraint, domain_set)
        for domain in domain_set:
            domain_violations = domains_violations[domain]
            domain_violations.add(violation)
    return domains_violations


_empty_frozen_set: FrozenSet = frozenset()


# XXX: Although this is written very generally for multiple domains; for most iterationg only one domain
# changes and we are only checking that one domain in domains. So there's no optimization here for
# quitting early since we are usually only checking a single constraint.
def _violations_of_domain_constraint(domains: Iterable[Domain],
                                     constraint: DomainConstraint,
                                     ) -> Dict[Domain, Set[_Violation]]:
    violations: Dict[Domain, Set[_Violation]] = defaultdict(set)
    unfixed_domains = [domain for domain in domains if not domain.fixed]
    violating_domains: List[Optional[Domain]] = []

    num_threads = dc.cpu_count() if constraint.threaded else 1

    if log_names_of_domains_and_strands_checked:
        logger.debug(f'$ for domain constraint {constraint.description}, checking these domains:')
        logger.debug(f'{pprint.pformat(unfixed_domains, indent=pprint_indent)}')

    if (not constraint.threaded
            or num_threads == 1
            or len(unfixed_domains) == 1):
        logger.debug(f'NOT using threading for domain constraint {constraint.description}')
        for domain in unfixed_domains:
            if not constraint(domain):
                violating_domains.append(domain)
    else:
        logger.debug(f'using threading for domain constraint {constraint.description}')

        def domain_if_violates(domain: Domain) -> Optional[Domain]:
            # return domain if it violates the constraint, else None
            if not constraint(domain):
                return domain
            else:
                return None

        violating_domains = _thread_pool.map(domain_if_violates, unfixed_domains)

    for violating_domain in violating_domains:
        if violating_domain is not None:
            violation = _Violation(constraint, [violating_domain])
            violations[violating_domain].add(violation)

    return violations


def _violations_of_strand_constraint(strands: Iterable[Strand],
                                     constraint: StrandConstraint,
                                     current_weight_gap: Optional[float],
                                     ) -> Dict[Domain, Set[_Violation]]:
    violations: Dict[Domain, Set[_Violation]] = defaultdict(set)
    unfixed_strands = [strand for strand in strands if not strand.fixed]
    sets_of_violating_domains: List[FrozenSet[Domain]] = []

    weight_discovered_here: float = 0.0

    num_threads = dc.cpu_count() if constraint.threaded else 1
    chunk_size = num_threads

    if log_names_of_domains_and_strands_checked:
        logger.debug(f'$ for strand constraint {constraint.description}, checking these strands:')
        logger.debug(f'$ {pprint.pformat(unfixed_strands, indent=pprint_indent)}')

    if (not constraint.threaded
            or num_threads == 1
            or len(unfixed_strands) == 1
            or (current_weight_gap is not None and chunk_size == 1)):
        logger.debug(f'NOT using threading for strand constraint {constraint.description}')
        for strand in unfixed_strands:
            if not constraint(strand):
                set_of_violating_domains = frozenset(strand.unfixed_domains())
                sets_of_violating_domains.append(set_of_violating_domains)
                if current_weight_gap is not None:
                    weight_discovered_here += constraint.weight
                    if weight_discovered_here > current_weight_gap:
                        break
    else:
        logger.debug(f'using threading for strand constraint {constraint.description}')

        def strand_to_unfixed_domains_set(strand: Strand) -> FrozenSet[Domain]:
            # return unfixed domains on strand if strand violates the constraint, else empty set
            if not constraint(strand):
                return frozenset(strand.unfixed_domains())
            else:
                return _empty_frozen_set

        if current_weight_gap is None:
            sets_of_violating_domains = _thread_pool.map(strand_to_unfixed_domains_set, unfixed_strands)
        else:
            for strand_chunk in dc.chunker(unfixed_strands, chunk_size):
                sets_of_violating_domains_chunk = _thread_pool.map(strand_to_unfixed_domains_set,
                                                                   strand_chunk)
                sets_of_violating_domains.extend(sets_of_violating_domains_chunk)

                # quit early if possible
                weight_discovered_here += constraint.weight * len(sets_of_violating_domains_chunk)
                if weight_discovered_here > current_weight_gap:
                    break

    for set_of_violating_domains in sets_of_violating_domains:
        if len(set_of_violating_domains) > 0:
            violation = _Violation(constraint, set_of_violating_domains)
            for domain in set_of_violating_domains:
                violations[domain].add(violation)

    return violations


T = TypeVar('T')


def remove_none_from_list(lst: Iterable[Optional[T]]) -> List[T]:
    return [elt for elt in lst if elt is not None]


def _violations_of_domain_pair_constraint(domains: Iterable[Domain],
                                          constraint: DomainPairConstraint,
                                          domain_changed: Optional[Domain],
                                          current_weight_gap: Optional[float],
                                          ) -> Dict[Domain, Set[_Violation]]:
    # If specified, current_weight_gap is the current difference between the weight of violated constraints
    # that have been found so far in the current iteration, compared to the total weight of violated
    # constraints in the optimal solution so far. It is positive
    # (i.e., total_weight_opt - total_weight_cur_so_far)
    # If specified and it is discovered while looping in this function that total_weight_cur_so_far plus
    # the weight of violated constraints discovered in this function exceeds total_weight_opt, quit early.
    domain_pairs_to_check: Sequence[Tuple[Domain, Domain]] = \
        _determine_domain_pairs_to_check(domains, domain_changed, constraint)

    if log_names_of_domains_and_strands_checked:
        logger.debug(f'$ for domain pair constraint {constraint.description}, checking these domain pairs:')
        logger.debug(f'$ {pprint.pformat(domain_pairs_to_check, indent=pprint_indent)}')

    violating_domain_pairs: List[Optional[Tuple[Domain, Domain]]] = []

    weight_discovered_here: float = 0.0

    cpu_count = dc.cpu_count() if constraint.threaded else 1

    # since each domain pair check is already parallelized for the four domain pairs
    # (d1,d2), (d1,w2), (w1,d2), (w1,w2), we take smaller chunks
    chunk_size = cpu_count // 4

    if (not constraint.threaded
            or cpu_count == 1
            or (current_weight_gap is not None and chunk_size == 1)):
        logger.debug(f'NOT using threading for domain pair constraint {constraint.description}')
        for domain1, domain2 in domain_pairs_to_check:
            assert not domain1.fixed or not domain2.fixed
            assert domain1.name != domain2.name
            if not constraint((domain1, domain2)):
                violating_domain_pairs.append((domain1, domain2))
                if current_weight_gap is not None:
                    weight_discovered_here += constraint.weight
                    if weight_discovered_here > current_weight_gap:
                        break
    else:
        logger.debug(f'using threading for domain pair constraint {constraint.description}')

        def domain_pair_if_violates(domain_pair: Tuple[Domain, Domain]) -> Optional[Tuple[Domain, Domain]]:
            # return domain pair if it violates the constraint, else None
            if not constraint(domain_pair):
                return domain_pair
            else:
                return None

        if current_weight_gap is None:
            violating_domain_pairs = list(
                _thread_pool.map(domain_pair_if_violates, domain_pairs_to_check))
        else:
            chunks = dc.chunker(domain_pairs_to_check, chunk_size)
            for domain_chunk in chunks:
                violating_domain_pairs_chunk_with_none = \
                    _thread_pool.map(domain_pair_if_violates, domain_chunk)
                violating_domain_pairs_chunk = remove_none_from_list(violating_domain_pairs_chunk_with_none)
                violating_domain_pairs.extend(violating_domain_pairs_chunk)

                # quit early if possible
                weight_discovered_here += constraint.weight * len(violating_domain_pairs_chunk)
                if weight_discovered_here > current_weight_gap:
                    break

    violations: Dict[Domain, Set[_Violation]] = defaultdict(set)
    violating_domain_pair: Optional[Tuple[Domain, Domain]]
    for violating_domain_pair in violating_domain_pairs:
        if violating_domain_pair is not None:
            domain1, domain2 = violating_domain_pair
            unfixed_domains_set: Set[Domain] = set()
            if not domain1.fixed:
                unfixed_domains_set.add(domain1)
            if not domain2.fixed:
                unfixed_domains_set.add(domain2)
            violation = _Violation(constraint, frozenset(unfixed_domains_set))
            if not domain1.fixed:
                violations[domain1].add(violation)
            if not domain2.fixed:
                violations[domain2].add(violation)

    return violations


def _violations_of_strand_pair_constraint(strands: Iterable[Strand],
                                          constraint: StrandPairConstraint,
                                          domain_changed: Optional[Domain],
                                          current_weight_gap: Optional[float],
                                          ) -> Dict[Domain, Set[_Violation]]:
    strand_pairs_to_check: Sequence[Tuple[Strand, Strand]] = \
        _determine_strand_pairs_to_check(strands, domain_changed, constraint)

    if log_names_of_domains_and_strands_checked:
        logger.debug(f'$ for strand pair constraint {constraint.description}, checking these strand pairs:')
        logger.debug(f'$ {pprint.pformat(strand_pairs_to_check, indent=pprint_indent)}')

    violating_strand_pairs: List[Optional[Tuple[Strand, Strand]]] = []

    weight_discovered_here: float = 0.0

    cpu_count = dc.cpu_count() if constraint.threaded else 1

    chunk_size = cpu_count

    if (not constraint.threaded
            or cpu_count == 1
            or (current_weight_gap is not None and chunk_size == 1)):
        logger.debug(f'NOT using threading for strand pair constraint {constraint.description}')
        for strand1, strand2 in strand_pairs_to_check:
            assert not strand1.fixed or not strand2.fixed
            if not constraint((strand1, strand2)):
                violating_strand_pairs.append((strand1, strand2))
                if current_weight_gap is not None:
                    weight_discovered_here += constraint.weight
                    if weight_discovered_here > current_weight_gap:
                        break
    else:
        logger.debug(f'NOT using threading for strand pair constraint {constraint.description}')

        def strand_pair_if_violates(strand_pair: Tuple[Strand, Strand]) -> Optional[Tuple[Strand, Strand]]:
            # return strand pair if it violates the constraint, else None
            if not constraint(strand_pair):
                return strand_pair
            else:
                return None

        if current_weight_gap is None:
            violating_strand_pairs = list(
                _thread_pool.map(strand_pair_if_violates, strand_pairs_to_check))
        else:
            chunks = dc.chunker(strand_pairs_to_check, chunk_size)
            for strand_chunk in chunks:
                violating_strand_pairs_chunk_with_none: List[Optional[Tuple[Strand, Strand]]] = \
                    _thread_pool.map(strand_pair_if_violates, strand_chunk)
                violating_strand_pairs_chunk: List[Tuple[Strand, Strand]] = \
                    remove_none_from_list(violating_strand_pairs_chunk_with_none)
                violating_strand_pairs.extend(violating_strand_pairs_chunk)

                # quit early if possible
                weight_discovered_here += constraint.weight * len(violating_strand_pairs_chunk)
                if weight_discovered_here > current_weight_gap:
                    break

    violations: Dict[Domain, Set[_Violation]] = defaultdict(set)
    violating_strand_pair: Optional[Tuple[Strand, Strand]]
    for violating_strand_pair in violating_strand_pairs:
        if violating_strand_pair is not None:
            strand1, strand2 = violating_strand_pair
            unfixed_domains_set = frozenset(strand1.unfixed_domains() + strand2.unfixed_domains())
            violation = _Violation(constraint, unfixed_domains_set)
            for domain in unfixed_domains_set:
                violations[domain].add(violation)

    return violations


def _sequences_fragile_format_output_to_file(design: Design,
                                             include_group: bool = True) -> str:
    return '\n'.join(
        f'{strand.name}  '
        f'{strand.group.name if include_group else ""}  '
        f'{strand.sequence(spaces_between_domains=True)}' for strand in design.strands)


def _write_sequences(design: Design, directory: str,
                     filename_with_iteration: str, filename_final: str,
                     include_group: bool = True) -> None:
    content_fragile_format = _sequences_fragile_format_output_to_file(design, include_group)
    for filename in [filename_with_iteration, filename_final]:
        path = os.path.join(directory, filename)
        with open(path, 'w') as file:
            file.write(content_fragile_format)


def _write_design_json(design: Design, directory: str,
                       filename_with_iteration_no_ext: str, filename_final_no_ext: str) -> None:
    json_str = design.to_json()
    for filename in [filename_with_iteration_no_ext, filename_final_no_ext]:
        filename += '.json'
        path = os.path.join(directory, filename)
        with open(path, 'w') as file:
            file.write(json_str)


def _write_report(design: Design, directory: str,
                  filename_with_iteration: str, filename_final: str) -> None:
    sequences = _sequences_fragile_format_output_to_file(design, include_group=True)
    sequences_content = f'''\
Design
======
{sequences}

'''

    report_str = design.summary_of_constraints()
    report = f'''\
Report on constraints
=====================
{report_str}
'''

    for filename in [filename_with_iteration, filename_final]:
        path = os.path.join(directory, filename)
        with open(path, 'w') as file:
            file.write(sequences_content)
            file.write(report)


def search_for_dna_sequences(*, design: dc.Design,
                             probability_of_keeping_change: Optional[Callable[[float], float]] = None,
                             random_seed: Optional[int] = None,
                             never_increase_weight: Optional[bool] = None,
                             directory_output_files: str = '.',
                             design_filename_no_ext: Optional[str] = None,
                             sequences_filename_no_ext: Optional[str] = None,
                             report_filename_no_ext: Optional[str] = None,
                             ) -> None:
    """
    Search for DNA sequences to assign to each :any:`Domain` in `design`, satisfying the various
    :any:`Constraint`'s associated with `design`.

    **Search algorithm:**
    This is a stochastic local search. It determines which :any:`Constraint`'s are violated.
    More precisely, it adds the total weight of all violated constraints
    (sum of :py:data:`constraints.Constraint.weight` over all violated :any:`Constraint`'s).
    The goal is to reduce this total weight until it is 0 (i.e., no violated constraints).
    Any :any:`Domain` "involved" in the violated :any:`Constraint` is noted as being one of the
    :any:`Domain`'s responsible for the violation. (For example, if a :any:`DomainConstraint` is violated,
    only one :any:`Domain` is blamed, whereas if a :any:`StrandConstraint` is violated, every :any:`Domain`
    in the :any:`Strand` is blamed.) While any :any:`Constraint`'s are violated, a :any:`Domain` is picked
    at random, with probability proportional to the total weight of all the :any:`Constraint`'s
    for which the :any:`Domain` was blamed. A new DNA sequence is assigned to this
    :any:`Domain` by calling :py:meth:`constraints.DomainPool.generate_sequence` on the :any:`DomainPool`
    of that :any:`Domain`. The way to decide whether to keep the changed sequence, or revert to the
    old sequence, is to calculate the total weight of all violated constraints in the original and changed
    :any:`Design`, calling their difference `weight_delta` = `new_total_weight` - `old_total_weight`.
    The value ``probability_of_keeping_change(weight_delta)`` is the probability that the change
    is kept. The default function computing this probability is returned by
    :py:meth:`default_probability_of_keeping_change_function`.

    The :any:`Design` is modified in place; each :any:`Domain` is modified to have a DNA sequence.

    Only :any:`Domain`'s with :py:data:`constraints.Domain.fixed` = False are eligible to have their
    DNA sequences modified; fixed :any:`Domain`'s are never blamed for violating :any:`Constraint`'s.

    If no DNA sequences are assigned to the :any:`Domain`'s initially, they are picked at random
    from the :any:`DomainPool` associated to each :any:`Domain` by calling
    :py:meth:`constraints.DomainPool.generate_sequence`.

    Otherwise, if DNA sequences are already assigned to the :any:`Domain`'s initially, these sequences
    are used as a starting point for finding sequences that satisfy all :any:`Constraint`'s.
    (In this case, those sequences are not checked against any :any:`NumpyConstraint`'s
    or :any:`SequenceConstraint`'s in the :any:`Design`, since those checks are applied prior to
    assigning DNA sequences to any :any:`Domain`.)

    The function has some side effects. It writes a report on the optimal sequence assignment found so far
    every time a new improve assignment is found. This re-evaluates the entire design, so can be expensive,
    but in practice the design is strictly improved many fewer times than total iterations.

    :param design:
        The :any:`Design` containing the :any:`Domain`'s to which to assign DNA sequences
        and the :any:`Constraint`'s that apply to them
    :param probability_of_keeping_change:
        Function giving the probability of keeping a change in one
        :any:`Domain`'s DNA sequence, if the new sequence affects the total weight of all violated
        :any:`Constraint`'s by `weight_delta`, the input to `probability_of_keeping_change`.
        See :py:meth:`default_probability_of_keeping_change_function` for a description of the default
        behavior if this parameter is not specified.
    :param never_increase_weight:
        If specified and True, then it is assumed that the function
        probability_of_keeping_change returns 0 for any negative value of `weight_delta` (i.e., the search
        never goes "uphill"), and the search for violations is optimized to quit as soon as the total weight
        of violations exceeds that of the current optimal solution. This vastly speeds up the search in later
        stages, when the current optimal solution is low weight. If both `probability_of_keeping_change` and
        `never_increase_weight` are left unspecified, then `probability_of_keeping_change` uses the default,
        which never goes uphill, and `never_increase_weight` is set to True. If
        `probability_of_keeping_change` is specified and `never_increase_weight` is not, then
        `never_increase_weight` is set to False. If both are specified and `never_increase_weight` is set to
        True, then take caution that `probability_of_keeping_change` really has the property that it never
        goes uphill; the optimization will essentially prevent most uphill climbs from occurring.
    :param directory_output_files:
        Directory in which to write output files (report on constraint violations and DNA sequences)
        whenever a new optimal sequence assignment is found.
    :param sequences_filename_no_ext:
        DNA sequences of each strand are written to a text file whenever
        a new optimal sequence assignment is found. No extension should be given. The filename will be
        `sequences_filename_no_ext-<i>.txt`, where i is the number of times the optimal solution
        has changed (so there will be a record of all optimal solutions recorded).
    :param design_filename_no_ext:
        The whole design is written to a JSON file whenever
        a new optimal sequence assignment is found. No extension should be given. The filename will be
        `design_filename_no_ext-<i>.txt`, where i is the number of times the optimal solution
        has changed (so there will be a record of all optimal solutions recorded).
    :param report_filename_no_ext:
        A report on the DNA sequences is written to a file when
        all constraints are satisfied. The method :py:meth:`constraints.Constraint.generate_summary` is
        called for each :any:`Constraint` in order to generate this report.
    :param random_seed:
        Integer given as a random seed to the numpy random number generator, used for
        all random choices in the algorithm. Set this to a fixed value to allow reproducibility.
    """
    debug_file_handler = logging.FileHandler(os.path.join(directory_output_files, '_debug.log'))
    info_file_handler = logging.FileHandler(os.path.join(directory_output_files, '_info.log'))
    debug_file_handler.setLevel(logging.DEBUG)
    info_file_handler.setLevel(logging.INFO)
    dc.logger.addHandler(debug_file_handler)
    dc.logger.addHandler(info_file_handler)

    if design_filename_no_ext is None:
        design_filename_no_ext = f'{script_name_no_ext()}_design'
    if sequences_filename_no_ext is None:
        sequences_filename_no_ext = f'{script_name_no_ext()}_sequences'
    if report_filename_no_ext is None:
        report_filename_no_ext = f'{script_name_no_ext()}_report'

    if not os.path.exists(directory_output_files):
        os.makedirs(directory_output_files)

    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = dn.default_rng

    if probability_of_keeping_change is None:
        probability_of_keeping_change = default_probability_of_keeping_change_function(design)
        if never_increase_weight is None:
            never_increase_weight = True
    elif never_increase_weight is None:
        never_increase_weight = False

    assert never_increase_weight is not None

    cpu_count = dc.cpu_count()
    logger.info(f'number of processes in system: {cpu_count}')

    try:
        assign_sequences_to_domains_randomly_from_pools(design=design,
                                                        rng=rng,
                                                        overwrite_existing_sequences=False)

        violation_set_opt, domains_opt, weights_opt = _find_violations_and_weigh(
            design, never_increase_weight=never_increase_weight)

        # write initial sequences and report
        _write_design_json(design,
                           directory=directory_output_files,
                           filename_with_iteration_no_ext=f'{design_filename_no_ext}-0',
                           filename_final_no_ext=f'_final-{design_filename_no_ext}')
        _write_sequences(design,
                         directory=directory_output_files,
                         filename_with_iteration=f'{sequences_filename_no_ext}-0.txt',
                         filename_final=f'_final-{sequences_filename_no_ext}.txt')
        _write_report(design,
                      directory=directory_output_files,
                      filename_with_iteration=f'{report_filename_no_ext}-0.txt',
                      filename_final=f'_final-{report_filename_no_ext}.txt')

        # this helps with logging if we execute no iterations
        violation_set_new = violation_set_opt

        iteration = 0
        num_new_optimal = 0
        while len(violation_set_opt.all_violations) > 0:
            if cpu_count != dc.cpu_count():
                logger.info(f'number of processes in system changed from {cpu_count} to {dc.cpu_count()}'
                            f'\nallocating new ThreadPool')
                cpu_count = dc.cpu_count()
                global _thread_pool
                _thread_pool.close()
                _thread_pool.terminate()
                _thread_pool = ThreadPool(processes=cpu_count)

            # pick domain to change, with probability proportional to total weight of constraints it violates
            # domain_changed: Domain = random.choices(domains_opt, weights_opt)[0]
            probs_opt = np.asarray(weights_opt)
            probs_opt /= probs_opt.sum()
            domain_changed: Domain = rng.choice(a=domains_opt, p=probs_opt)
            assert not domain_changed.fixed  # fixed Domains should never be blamed for constraint violation

            # set sequence of domain_changed to random new sequence from its DomainPool
            original_sequence = domain_changed.sequence
            domain_changed.sequence = domain_changed.pool.generate_sequence()

            # evaluate constraints on new Design with domain_to_change's new sequence
            violation_set_new, domains_new, weights_new = _find_violations_and_weigh(
                design, domain_changed, violation_set_opt, never_increase_weight)

            _log_constraint_summary(design=design,
                                    violation_set_opt=violation_set_opt, violation_set_new=violation_set_new,
                                    iteration=iteration, num_new_optimal=num_new_optimal)

            # based on total weight of new constraint violations compared to optimal assignment so far,
            # decide whether to keep the change
            weight_delta = violation_set_new.total_weight() - violation_set_opt.total_weight()
            prob_keep_change = probability_of_keeping_change(weight_delta)
            keep_change = rng.random() < prob_keep_change

            if not keep_change:
                # set back to old sequence
                domain_changed.sequence = original_sequence
            else:
                # keep new sequence and update information about optimal solution so far
                domains_opt = domains_new
                weights_opt = weights_new
                violation_set_opt = violation_set_new
                if weight_delta < 0:  # increment whenever we actually improve the design
                    num_new_optimal += 1
                    # put leading 0s on sequential filenames 
                    str_num_new_optimal_with_leading_zeros = '0' * (
                            2 - int(math.log(max(num_new_optimal, 1), 10))) + str(num_new_optimal)
                    _write_design_json(design,
                                       directory=directory_output_files,
                                       # filename_with_iteration_no_ext=f'{design_filename_no_ext}-{num_new_optimal}',
                                       filename_with_iteration_no_ext=f'{design_filename_no_ext}-' + str_num_new_optimal_with_leading_zeros,
                                       filename_final_no_ext=f'_final-{design_filename_no_ext}')
                    _write_sequences(design,
                                     directory=directory_output_files,
                                     # filename_with_iteration=f'{sequences_filename_no_ext}-{num_new_optimal}.txt',
                                     filename_with_iteration=f'{sequences_filename_no_ext}-' + str_num_new_optimal_with_leading_zeros + '.txt',
                                     filename_final=f'_final-{sequences_filename_no_ext}.txt')
                    _write_report(design,
                                  directory=directory_output_files,
                                  # filename_with_iteration=f'{report_filename_no_ext}-{num_new_optimal}.txt',
                                  filename_with_iteration=f'{report_filename_no_ext}-' + str_num_new_optimal_with_leading_zeros + '.txt',
                                  filename_final=f'_final-{report_filename_no_ext}.txt')

            iteration += 1

        _log_constraint_summary(design=design,
                                violation_set_opt=violation_set_opt, violation_set_new=violation_set_new,
                                iteration=iteration, num_new_optimal=num_new_optimal)

    except KeyboardInterrupt:
        _pfunc_killall()
        raise
    finally:
        _thread_pool.close()  # noqa
        _thread_pool.terminate()

    dc.logger.removeHandler(debug_file_handler)
    dc.logger.removeHandler(info_file_handler)


def _pfunc_killall() -> None:
    import subprocess
    delim = '#'
    logger.warning('\n' + delim * 79)
    logger.warning('# attempting to kill all pfunc processes with `killall pfunc Pfunc`')
    command = ['killall', 'pfunc', 'Pfunc']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    if output is None:
        output = 'None'
    if error is None:
        error = 'None'
    output_str = output.decode() if isinstance(output, bytes) else output
    error_str = error.decode() if isinstance(error, bytes) else error
    output_str = textwrap.indent(output_str, delim + '   ')
    error_str = textwrap.indent(error_str, delim + '   ')
    logger.warning(f'{delim} output from killall command:\n{output_str}')
    logger.warning(f'{delim} error from killall command:\n{error_str}')
    logger.warning('#' * 79)


n_in_last_n_calls = 20
time_last_n_calls: Deque = deque(maxlen=n_in_last_n_calls)
time_last_n_calls_available = False


def _log_time(stopwatch: Stopwatch) -> None:
    global time_last_n_calls_available
    if time_last_n_calls_available:
        time_last_n_calls.append(stopwatch.milliseconds())
        ave_time = statistics.mean(time_last_n_calls)
        med_time = statistics.median(time_last_n_calls)
        logger.info(('-' * 79) +
                    f'\n| time: {stopwatch.milliseconds_str()} ms; '
                    f'last {n_in_last_n_calls} calls '
                    f'average: {ave_time:.2f} ms '
                    f'median: {med_time:.2f} ms'
                    )
    else:
        # skip appending first time, since it is much larger and skews the average
        logger.info(f'time for first call: {stopwatch.milliseconds_str()} ms')
        time_last_n_calls_available = True


def script_name_no_ext() -> str:
    """
    :return: Name of the Python script currently running, without the .py extension.
    """
    script_name = os.path.basename(sys.argv[0])
    last_dot_idx = script_name.rfind('.')
    if last_dot_idx >= 0:
        script_name = script_name[:last_dot_idx]
    return script_name


def _find_violations_and_weigh(design: Design,
                               domain_changed: Optional[Domain] = None,
                               violation_set_old: Optional[_ViolationSet] = None,
                               never_increase_weight: bool = False) \
        -> Tuple[_ViolationSet, List[Domain], List[float]]:
    """
    :param design:
        :any:`Design` to evaluate
    :param domain_changed:
        The :any:`Domain` that just changed;
        if None, then recalculate all constraints,
        otherwise assume no constraints changed that do not involve `domain`
    :param violation_set_old:
        :any:`ViolationSet` to update, assuming `domain_changed` is the only :any:`Domain` that changed
    :param never_increase_weight:
        See _violations_of_constraints for explanation of this parameter.
    :return:
        Tuple (violations, domains, weights)
            `violations`: dict mapping each domain to list of constraints that they violated
            `domains`:    list of :any:`Domain`'s that caused violations
            `weights`:    list of weights for each :any:`Domain`, in same order the domains appear, giving
                          the total weight of :any:`Constraint`'s violated by the corresponding :any:`Domain`
    """
    stopwatch = Stopwatch()

    violation_set: _ViolationSet = _violations_of_constraints(
        design, never_increase_weight, domain_changed, violation_set_old)
    # violation_set: _ViolationSet = _violations_of_constraints(design) # uncomment to recompute all violations
    domain_to_weights: Dict[Domain, float] = {
        domain: sum(violation.constraint.weight for violation in domain_violations)
        for domain, domain_violations in violation_set.domain_to_violations.items()
    }
    domains = list(domain_to_weights.keys())
    weights = list(domain_to_weights.values())

    stopwatch.stop()

    _log_time(stopwatch)

    return violation_set, domains, weights


def _flatten(list_of_lists: Iterable[Iterable[Any]]) -> Iterable[Any]:
    #  Flatten one level of nesting
    return itertools.chain.from_iterable(list_of_lists)


def _log_constraint_summary(*, design: Design,
                            violation_set_opt: _ViolationSet,
                            violation_set_new: _ViolationSet,
                            iteration: int,
                            num_new_optimal: int) -> None:
    total_count_new = len(violation_set_new.all_violations)
    total_count_opt = len(violation_set_opt.all_violations)

    all_constraints = design.all_constraints()
    all_violation_descriptions = [
        violation.constraint.short_description for violation in violation_set_new.all_violations]
    violation_description_counts: Counter = Counter(all_violation_descriptions)

    weight_header = 'iteration|#updates|opt weight|new weight|opt count|new count||'
    all_constraints_header = '|'.join(
        f'{constraint.short_description}' for constraint in all_constraints)
    header = weight_header + all_constraints_header
    header_width = len(header)
    logger.info('-' * header_width + '\n' + header)

    weight_str = f'{iteration:9}|{num_new_optimal:8}|' \
                 f'{violation_set_opt.total_weight():10.1f}|{violation_set_new.total_weight():10.1f}|' \
                 f'{total_count_opt:9}|{total_count_new:9}||'
    all_constraints_str = '|'.join(
        f'{violation_description_counts[constraint.short_description]:{len(constraint.short_description)}}'
        for constraint in all_constraints)
    logger.info(weight_str + all_constraints_str)


def assign_sequences_to_domains_randomly_from_pools(design: Design,
                                                    rng: np.random.Generator = dn.default_rng,
                                                    overwrite_existing_sequences: bool = False) -> None:
    """
    Assigns to each :any:`Domain` in this :any:`Design` a random DNA sequence from its
    :any:`DomainPool`, calling :py:meth:`constraints.DomainPool.generate_sequence` to get the sequence.

    This is step #1 in the search algorithm.

    :param design:
        Design to which to assign DNA sequences.
    :param rng:
        numpy random number generator (type returned by numpy.random.default_rng()).
    :param overwrite_existing_sequences:
        Whether to overwrite in this initial assignment any existing sequences for :any:`Domain`'s
        that already have a DNA sequence. The DNA sequence of a :any:`Domain` with
        :py:data:`constraints.Domain.fixed` = True are never overwritten, neither here nor later in the
        search. Non-fixed sequences can be skipped for overwriting on this initial assignment, but they
        are subject to change by the subsequent search algorithm.
    """
    for domain in design.domains:
        skip_fixed_msg = skip_nonfixed_msg = None
        if domain.has_sequence():
            skip_fixed_msg = f'Skipping assignment of DNA sequence to domain {domain.name}. ' \
                             f'That domain has a NON-FIXED sequence {domain.sequence}, ' \
                             f'which the search will attempt to replace.'
            skip_nonfixed_msg = f'Skipping assignment of DNA sequence to domain {domain.name}. ' \
                                f'That domain has a FIXED sequence {domain.sequence}.'
        if overwrite_existing_sequences:
            if not domain.fixed:
                domain.sequence = domain.pool.generate_sequence(rng)
                assert len(domain.sequence) == domain.pool.length
            else:
                logger.info(skip_fixed_msg)
        else:
            if not domain.has_sequence():
                domain.sequence = domain.pool.generate_sequence(rng)
                assert len(domain.sequence) == domain.pool.length
            else:
                if domain.fixed:
                    logger.info(skip_nonfixed_msg)
                else:
                    logger.info(skip_fixed_msg)


_sentinel = object()


def _iterable_is_empty(iterable: abc.Iterable) -> bool:
    iterator = iter(iterable)
    return next(iterator, _sentinel) is _sentinel


def default_probability_of_keeping_change_function(design: dc.Design) -> Callable[[float], float]:
    """
    Returns a function that takes a float input `weight_delta` representing a change in weight of
    violated constraint, which returns a probability of keeping the change in the DNA sequence assignment.
    The probability is 1 if the change it is at least as good as the previous
    (roughly, the weight change is not positive), and the probability is 0 otherwise.

    To mitigate floating-point rounding errors, the actual condition checked is that
    `weight_delta` < :py:data:`epsilon`,
    on the assumption that if the same weight of constraints are violated,
    rounding errors in calculating `weight_delta` could actually make it slightly above than 0
    and result in reverting to the old assignment when we really want to keep the change.
    If all values of :py:data:`Constraint.weight` are significantly about :py:data:`epsilon`
    (e.g., 1.0 or higher), then this should be is equivalent to keeping a change in the DNA sequence
    assignment if and only if it is no worse than the previous.

    :param design: :any:`Design` to apply this rule for; `design` is required because the weight of
                   :any:`Constraint`'s in the :any:`Design` are used to calculate an appropriate
                   epsilon value for determining when a weight change is too small to be significant
                   (i.e., is due to rounding error)
    :return: the "keep change" function `f`: :math:`\\mathbb{R} \\to [0,1]`,
             where :math:`f(w_\\delta) = 1` if :math:`w_\\delta \\leq \\epsilon`
             (where :math:`\\epsilon` is chosen to be 1,000,000 times smaller than
             the smallest :any:`Constraint.weight` for any :any:`Constraint` in `design`),
             and :math:`f(w_\\delta) = 0` otherwise.
    """
    min_weight = min(constraint.weight for constraint in design.all_constraints())
    epsilon_from_min_weight = min_weight / 1000000.0

    def keep_change_only_if_no_worse(weight_delta: float) -> float:
        return 1.0 if weight_delta <= epsilon_from_min_weight else 0.0

    # def keep_change_only_if_better(weight_delta: float) -> float:
    #     return 1.0 if weight_delta <= -epsilon_from_min_weight else 0.0

    return keep_change_only_if_no_worse
    # return keep_change_only_if_better
