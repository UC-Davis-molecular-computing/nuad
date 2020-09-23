"""
Stochastic local search for finding DNA sequences to assign to
:any:`Domain`'s in a :any:`Design` to satisfy all :any:`Constraint`'s.
"""

import itertools
import os
import shutil
import sys
import logging
import pprint
from collections import Counter, defaultdict, deque
import collections.abc as abc
from dataclasses import dataclass, field
from typing import List, Tuple, Sequence, Set, FrozenSet, Optional, Dict, Callable, Iterable, Generic, Any, \
    Deque, TypeVar
import statistics
import textwrap
import time
import re
import datetime
from pprint import pprint

from ordered_set import OrderedSet
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


def default_output_directory() -> str:
    return os.path.join('output', f'{script_name_no_ext()}--{timestamp()}')


@dataclass
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

    _weight: float

    def __init__(self, constraint: Constraint, domains: Iterable[Domain], weight: float):
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
        return self.constraint.weight * self._weight

    def __repr__(self) -> str:
        return f'Violation({self.constraint.short_description}, weight={self._weight:.2f})'

    def __str__(self) -> str:
        return repr(self)

    # _Violation equality based on identity; different Violations in memory are considered different,
    # even if all data between them matches. Don't create the same Violation twice!
    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        return self is other


@dataclass
class _ViolationSet:
    """
    Represents violations of :any:`Constraint`'s in a :any:`Design`.

    It is designed to be efficiently updateable when a single :any:`Domain` changes, to efficiently update
    only those violations of :any:`Constraint`'s that could have been affected by the changed :any:`Domain`.
    """

    all_violations: OrderedSet[_Violation] = field(default_factory=OrderedSet)
    """Set of all :any:`Violation`'s."""

    domain_to_violations: Dict[Domain, OrderedSet[_Violation]] = field(
        default_factory=lambda: defaultdict(OrderedSet))
    """Dict mapping each :any:`Domain` to the set of all :any:`Violation`'s for which it is blamed."""

    def __repr__(self):
        lines = "\n  ".join(map(str, self.all_violations))
        return f'ViolationSet(\n  {lines})'

    def __str__(self):
        return repr(self)

    def update(self, new_violations: Dict[Domain, OrderedSet[_Violation]]) -> None:
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
        domain_to_violations_deep_copy = defaultdict(OrderedSet, self.domain_to_violations)
        for domain, violations in domain_to_violations_deep_copy.items():
            domain_to_violations_deep_copy[domain] = OrderedSet(violations)
        return _ViolationSet(OrderedSet(self.all_violations), domain_to_violations_deep_copy)

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

    def num_violations(self) -> float:
        """
        :return: Total number of violations.
        """
        return len(self.all_violations)


def _violations_of_constraints(design: Design,
                               never_increase_weight: bool,
                               domains_changed: Optional[Iterable[Domain]],
                               violation_set_old: Optional[_ViolationSet],
                               weigh_violations_equally: bool,
                               iteration: int) -> _ViolationSet:
    """
    :param design:
        The :any:`Design` for which to find DNA sequences.
    :param domains_changed:
        The :any:`Domain`'s that just changed; if None, then recalculate all constraints, otherwise assume no
        constraints changed that do not involve a :any:`Domain` in `domains_changed`.
    :param violation_set_old:
        :any:`ViolationSet` to update, assuming `domain_changed` is the only :any:`Domain` that changed.
    :param never_increase_weight:
        Indicates whether the search algorithm is using an update rule that never increases the total weight
        of violations (i.e., it only goes downhill). If so we can optimize and stop this function early as
        soon as we find that the violations discovered so far exceed the total weight of the current optimal
        solution. In later stages of the search, when the optimal solution so far has very few violated
        constraints, this vastly speeds up the search by allowing most of the constraint checking to be
        skipping for most choices of DNA sequences to `domain_changed`.
    :param weigh_violations_equally:
        See other functions with this parameter.
    :param iteration:
        Current iteration number; useful for debugging (e.g., conditional breakpoints).
    :return:
        dict mapping each :any:`Domain` to the list of constraints it violated
    """

    if not ((domains_changed is None and violation_set_old is None) or (
            domains_changed is not None and violation_set_old is not None)):
        raise ValueError('domains_changed and violation_set_old should both be None or both be not None; '
                         f'domains_changed = {domains_changed}'
                         f'violation_set_old = {violation_set_old}')

    violation_set: _ViolationSet
    if domains_changed is None:
        violation_set = _ViolationSet()
    else:
        assert violation_set_old is not None
        violation_set = violation_set_old.clone()
        for domain_changed in domains_changed:
            assert not domain_changed.fixed
            violation_set.remove_violations_of_domain(domain_changed)

    # individual domain constraints within each DomainPool
    pools_domains = design.domain_pools.items() if domains_changed is None \
        else [(domain_changed.pool, [domain_changed]) for domain_changed in domains_changed]
    for domain_pool, domains_in_pool in pools_domains:
        for domain_constraint_pool in domain_pool.domain_constraints:
            domains = domains_in_pool if domains_changed is None else domains_changed
            domain_violations_pool = _violations_of_domain_constraint(
                domains=domains, constraint=domain_constraint_pool,
                weigh_constraint_violations_equally=weigh_violations_equally)
            violation_set.update(domain_violations_pool)

            if _quit_early(never_increase_weight, violation_set, violation_set_old):
                return violation_set

    # individual strand constraints within each StrandGroup
    for strand_pool, strands in design.strand_groups.items():
        for strand_constraint_pool in strand_pool.strand_constraints:
            current_weight_gap = violation_set_old.total_weight() - violation_set.total_weight() \
                if never_increase_weight and violation_set_old is not None else None
            strands = _strands_containing_domains(domains_changed, strands)
            strand_violations_pool, quit_early_in_func = _violations_of_strand_constraint(
                strands=strands, constraint=strand_constraint_pool, current_weight_gap=current_weight_gap,
                weigh_constraint_violations_equally=weigh_violations_equally)
            violation_set.update(strand_violations_pool)

            quit_early = _quit_early(never_increase_weight, violation_set, violation_set_old)
            assert quit_early == quit_early_in_func
            if quit_early:
                return violation_set

    # individual domain constraints across all domains in Design
    # most of the time we only check one of these, so we don't bother passing in the current weight gap
    for domain_constraint in design.domain_constraints:
        if domains_changed is None:
            domains_to_check = design.domains
        else:  # don't bother with thread pools if there's just one domain
            domains_to_check = domains_changed
        domain_violations = _violations_of_domain_constraint(
            domains=domains_to_check, constraint=domain_constraint,
            weigh_constraint_violations_equally=weigh_violations_equally)
        violation_set.update(domain_violations)

        if _quit_early(never_increase_weight, violation_set, violation_set_old):
            return violation_set

    # individual strand constraints across all strands in Design
    for strand_constraint in design.strand_constraints:
        current_weight_gap = violation_set_old.total_weight() - violation_set.total_weight() \
            if never_increase_weight and violation_set_old is not None else None
        strands = _strands_containing_domains(domains_changed, design.strands)
        strand_violations, quit_early_in_func = _violations_of_strand_constraint(
            strands=strands, constraint=strand_constraint, current_weight_gap=current_weight_gap,
            weigh_constraint_violations_equally=weigh_violations_equally)
        violation_set.update(strand_violations)

        quit_early = _quit_early(never_increase_weight, violation_set, violation_set_old)
        assert quit_early == quit_early_in_func
        if quit_early:
            return violation_set

    # all pairs of domains in Design
    for domain_pair_constraint in design.domain_pair_constraints:
        current_weight_gap = violation_set_old.total_weight() - violation_set.total_weight() \
            if never_increase_weight and violation_set_old is not None else None
        domain_pair_violations, quit_early_in_func = _violations_of_domain_pair_constraint(
            domains=design.domains, constraint=domain_pair_constraint, domains_changed=domains_changed,
            current_weight_gap=current_weight_gap,
            weigh_constraint_violations_equally=weigh_violations_equally)
        violation_set.update(domain_pair_violations)

        quit_early = _quit_early(never_increase_weight, violation_set, violation_set_old)
        assert quit_early == quit_early_in_func
        if quit_early:
            return violation_set

    # all pairs of strands in Design
    for strand_pair_constraint in design.strand_pair_constraints:
        current_weight_gap = violation_set_old.total_weight() - violation_set.total_weight() \
            if never_increase_weight and violation_set_old is not None else None
        strand_pair_violations, quit_early_in_func = _violations_of_strand_pair_constraint(
            strands=design.strands, constraint=strand_pair_constraint, domains_changed=domains_changed,
            current_weight_gap=current_weight_gap,
            weigh_constraint_violations_equally=weigh_violations_equally)
        violation_set.update(strand_pair_violations)

        quit_early = _quit_early(never_increase_weight, violation_set, violation_set_old)
        assert quit_early == quit_early_in_func
        if quit_early:
            return violation_set

    # constraints that process each domain, but all at once (e.g., to hand off in batch to RNAduplex)
    for domains_constraint in design.domains_constraints:
        domains = design.domains if domains_changed is None else domains_changed

        if log_names_of_domains_and_strands_checked:
            logger.debug(f'$ for domains constraint {domains_constraint.description}, '
                         f'checking these domains:\n'
                         f'${pprint.pformat(domains, indent=pprint_indent)}')

        sets_of_violating_domains_weights = domains_constraint(domains)
        domains_violations = _convert_sets_of_violating_domains_to_violations(
            domains_constraint, sets_of_violating_domains_weights, weigh_violations_equally)
        violation_set.update(domains_violations)

        quit_early = _quit_early(never_increase_weight, violation_set, violation_set_old)
        if quit_early:
            return violation_set

    # constraints that process each strand, but all at once (e.g., to hand off in batch to RNAduplex)
    for strands_constraint in design.strands_constraints:
        strands = _strands_containing_domains(domains_changed, design.strands)

        if log_names_of_domains_and_strands_checked:
            logger.debug(f'$ for strands constraint {strands_constraint.description}, '
                         f'checking these strands:\n'
                         f'${pprint.pformat(strands, indent=pprint_indent)}')

        if len(strands) > 0:
            sets_of_violating_domains_weights = strands_constraint(strands)
            domains_violations = _convert_sets_of_violating_domains_to_violations(
                strands_constraint, sets_of_violating_domains_weights, weigh_violations_equally)
            violation_set.update(domains_violations)

            quit_early = _quit_early(never_increase_weight, violation_set, violation_set_old)
            if quit_early:
                return violation_set

    # constraints that process all pairs of domains at once (e.g., to hand off in batch to RNAduplex)
    for domain_pairs_constraint in design.domain_pairs_constraints:
        domain_pairs_to_check = _determine_domain_pairs_to_check(design.domains, domains_changed,
                                                                 domain_pairs_constraint)

        if log_names_of_domains_and_strands_checked:
            logger.debug(f'$ for domain pairs constraint {domain_pairs_constraint.description}, '
                         f'checking these strand pairs:\n'
                         f'${pprint.pformat(domain_pairs_to_check, indent=pprint_indent)}')

        if len(domain_pairs_to_check) > 0:
            sets_of_violating_domains_weights = domain_pairs_constraint(domain_pairs_to_check)
            domains_violations = _convert_sets_of_violating_domains_to_violations(
                domain_pairs_constraint, sets_of_violating_domains_weights,
                weigh_violations_equally)
            violation_set.update(domains_violations)

            quit_early = _quit_early(never_increase_weight, violation_set, violation_set_old)
            if quit_early:
                return violation_set

    # constraints that process all pairs of strands at once (e.g., to hand off in batch to RNAduplex)
    for strand_pairs_constraint in design.strand_pairs_constraints:
        strand_pairs_to_check = _determine_strand_pairs_to_check(design.strands, domains_changed,
                                                                 strand_pairs_constraint)
        if log_names_of_domains_and_strands_checked:
            logger.debug(f'$ for strand pairs constraint {strand_pairs_constraint.description}, '
                         f'checking these strand pairs:\n'
                         f'${pprint.pformat(strand_pairs_to_check, indent=pprint_indent)}')

        if len(strand_pairs_to_check) > 0:
            sets_of_violating_domains_weights = strand_pairs_constraint(strand_pairs_to_check)
            domains_violations = _convert_sets_of_violating_domains_to_violations(
                strand_pairs_constraint, sets_of_violating_domains_weights,
                weigh_violations_equally)
            violation_set.update(domains_violations)

            quit_early = _quit_early(never_increase_weight, violation_set, violation_set_old)
            if quit_early:
                return violation_set

    # constraints that processes whole design at once (for anything not captured by the above, e.g.,
    # processing all triples of strands)
    for design_constraint in design.design_constraints:
        sets_of_violating_domains_weights = design_constraint(design, domains_changed)
        domains_violations = _convert_sets_of_violating_domains_to_violations(
            design_constraint, sets_of_violating_domains_weights, weigh_violations_equally)
        violation_set.update(domains_violations)

        quit_early = _quit_early(never_increase_weight, violation_set, violation_set_old)
        if quit_early:
            return violation_set

    # print('violation_set.domain_to_violations:')
    # pprint(violation_set.domain_to_violations)
    return violation_set



def _is_significantly_greater(x: float, y: float) -> bool:
    # epsilon = min(abs(x), abs(y)) * 0.001
    # XXX: important that this is absolute constant. Sometimes this is called for the total weight of all
    # violations, and sometimes just for the difference between old and new (the latter are smaller).
    # If using relative epsilon, then those can disagree and trigger the assert statement that
    # checks that _violations_of_constraints quit_early agrees with the subroutines it calls.
    epsilon = 0.001
    return x > y + epsilon


def _quit_early(never_increase_weight: bool,
                violation_set: _ViolationSet,
                violation_set_old: Optional[_ViolationSet]) -> bool:
    return (never_increase_weight and violation_set_old is not None
            and _is_significantly_greater(violation_set.total_weight(), violation_set_old.total_weight()))


def _at_least_one_domain_unfixed(pair: Tuple[Domain, Domain]) -> bool:
    return not (pair[0].fixed and pair[1].fixed)


def _determine_domain_pairs_to_check(all_domains: Iterable[Domain],
                                     domains_changed: Optional[Iterable[Domain]],
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
    domain_pairs_to_check = list(domain_pairs_to_check_if_domain_changed_none) if domains_changed is None \
        else [(domain1, domain2) for domain1, domain2 in domain_pairs_to_check_if_domain_changed_none
              if domain1 in domains_changed or domain2 in domains_changed]

    return domain_pairs_to_check


def _at_least_one_strand_unfixed(pair: Tuple[Strand, Strand]) -> bool:
    return not (pair[0].fixed and pair[1].fixed)


def _determine_strand_pairs_to_check(all_strands: Iterable[Strand],
                                     domains_changed: Optional[Iterable[Domain]],
                                     constraint: ConstraintWithStrandPairs) -> \
        Sequence[Tuple[Strand, Strand]]:
    """
    Similar to _determine_domain_pairs_to_check but for strands.
    """
    # either all pairs, or just constraint.pairs if specified
    strand_pairs_to_check_if_domain_changed_none = constraint.pairs if constraint.pairs is not None \
        else all_pairs(all_strands, where=_at_least_one_strand_unfixed)

    # filter out those not containing domain_change if specified
    strand_pairs_to_check: List[Tuple[Strand, Strand]] = []
    if domains_changed is None:
        strand_pairs_to_check = strand_pairs_to_check_if_domain_changed_none
    else:
        for strand1, strand2 in strand_pairs_to_check_if_domain_changed_none:
            for domain_changed in domains_changed:
                if domain_changed in strand1.domains or domain_changed in strand2.domains:
                    strand_pairs_to_check.append((strand1, strand2))
                    break

    return strand_pairs_to_check


def _strands_containing_domains(domains: Optional[Iterable[Domain]], strands: List[Strand]) -> List[Strand]:
    """
    :param domains:
        :any:`Domain`'s to check for, or None to return all of `strands`
    :param strands:
        `strands` in which to search for :any:`Strand`'s that contain `domain`
    :return:
        If `domain` is None, just return `strands`, otherwise return :any:`Strand`'s in `strands`
        that contain `domain`
    """
    if domains is None:
        return strands
    else:
        # ensure we don't return duplicates of strands, and keep original order
        strands_set = OrderedSet(strand for strand in strands for domain in domains
                                 if domain in strand.domains)
        return list(strands_set)


def _convert_sets_of_violating_domains_to_violations(
        constraint: Constraint, sets_of_violating_domains: Iterable[Tuple[OrderedSet[Domain], float]],
        weigh_constraint_violations_equally: bool) -> Dict[Domain, OrderedSet[_Violation]]:
    domains_violations: Dict[Domain, OrderedSet[_Violation]] = defaultdict(OrderedSet)
    for domain_set, weight in sets_of_violating_domains:
        weight_to_use = 1.0 if weigh_constraint_violations_equally else weight
        violation = _Violation(constraint, domain_set, weight_to_use)
        for domain in domain_set:
            domain_violations = domains_violations[domain]
            domain_violations.add(violation)
    return domains_violations


_empty_frozen_set: FrozenSet = frozenset()
_empty_ordered_set: OrderedSet = OrderedSet()


# XXX: Although this is written very generally for multiple domains; for most iterationg only one domain
# changes and we are only checking that one domain in domains. So there's no optimization here for
# quitting early since we are usually only checking a single constraint.
def _violations_of_domain_constraint(domains: Iterable[Domain],
                                     constraint: DomainConstraint,
                                     weigh_constraint_violations_equally: bool,
                                     ) -> Dict[Domain, OrderedSet[_Violation]]:
    violations: Dict[Domain, OrderedSet[_Violation]] = defaultdict(OrderedSet)
    unfixed_domains = [domain for domain in domains if not domain.fixed]
    violating_domains_weights: List[Optional[Tuple[Domain, float]]] = []

    num_threads = dc.cpu_count() if constraint.threaded else 1

    if log_names_of_domains_and_strands_checked:
        logger.debug(f'$ for domain constraint {constraint.description}, checking these domains:')
        logger.debug(f'{pprint.pformat(unfixed_domains, indent=pprint_indent)}')

    if (not constraint.threaded
            or num_threads == 1
            or len(unfixed_domains) == 1):
        logger.debug(f'NOT using threading for domain constraint {constraint.description}')
        for domain in unfixed_domains:
            weight: float = constraint(domain)
            if weight > 0.0:
                violating_domains_weights.append((domain, weight))
    else:
        logger.debug(f'using threading for domain constraint {constraint.description}')

        def domain_if_violates(domain: Domain) -> Optional[Tuple[Domain, float]]:
            # return domain if it violates the constraint, else None
            weight_: float = constraint(domain)
            if weight_ > 0.0:
                return domain, weight_
            else:
                return None

        violating_domains_weights = _thread_pool.map(domain_if_violates, unfixed_domains)

    for violating_domain_weight in violating_domains_weights:
        if violating_domain_weight is not None:
            violating_domain, weight = violating_domain_weight
            weight_to_use = 1.0 if weigh_constraint_violations_equally else weight
            violation = _Violation(constraint, [violating_domain], weight_to_use)
            violations[violating_domain].add(violation)

    return violations


def _violations_of_strand_constraint(strands: Iterable[Strand],
                                     constraint: StrandConstraint,
                                     current_weight_gap: Optional[float],
                                     weigh_constraint_violations_equally: bool,
                                     ) -> Tuple[Dict[Domain, OrderedSet[_Violation]], bool]:
    """
    :param strands:
    :param constraint:
    :param current_weight_gap:
    :param weigh_constraint_violations_equally:
    :return:
        1. dict mapping each domain to the set of violations that blame it
        2. bool indicating whether we quit early
    """
    violations: Dict[Domain, OrderedSet[_Violation]] = defaultdict(OrderedSet)
    unfixed_strands = [strand for strand in strands if not strand.fixed]
    sets_of_violating_domains_weights: List[Tuple[OrderedSet[Domain], float]] = []

    weight_discovered_here: float = 0.0
    quit_early = False

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
            weight: float = constraint(strand)
            if weight > 0.0:
                set_of_violating_domains_weight = OrderedSet(strand.unfixed_domains())
                sets_of_violating_domains_weights.append((set_of_violating_domains_weight, weight))
                if current_weight_gap is not None:
                    weight_discovered_here += constraint.weight * weight
                    if _is_significantly_greater(weight_discovered_here, current_weight_gap):
                        quit_early = True
                        break
    else:
        logger.debug(f'using threading for strand constraint {constraint.description}')

        def strand_to_unfixed_domains_set(strand: Strand) -> Tuple[OrderedSet[Domain], float]:
            # return unfixed domains on strand if strand violates the constraint, else empty set
            weight_: float = constraint(strand)
            if weight_ > 0.0:
                return OrderedSet(strand.unfixed_domains()), weight_
            else:
                return _empty_ordered_set, 0.0

        if current_weight_gap is None:
            sets_of_violating_domains_weights = _thread_pool.map(strand_to_unfixed_domains_set,
                                                                 unfixed_strands)
        else:
            for strand_chunk in dc.chunker(unfixed_strands, chunk_size):
                sets_of_violating_domains_excesses_chunk = _thread_pool.map(strand_to_unfixed_domains_set,
                                                                            strand_chunk)
                sets_of_violating_domains_weights.extend(sets_of_violating_domains_excesses_chunk)

                # quit early if possible; check weights of violations we just added
                total_weight_chunk = sum(weight_chunk
                                         for _, weight_chunk in sets_of_violating_domains_excesses_chunk)
                weight_discovered_here += constraint.weight * total_weight_chunk
                if _is_significantly_greater(weight_discovered_here, current_weight_gap):
                    quit_early = True
                    break

    # print(f'{[domains for domains,_ in sets_of_violating_domains_weights]}')
    for set_of_violating_domains_weight, weight in sets_of_violating_domains_weights:
        if len(set_of_violating_domains_weight) > 0:
            weight_to_use = 1.0 if weigh_constraint_violations_equally else weight
            violation = _Violation(constraint, set_of_violating_domains_weight, weight_to_use)
            for domain in set_of_violating_domains_weight:
                violations[domain].add(violation)

    return violations, quit_early


T = TypeVar('T')


def remove_none_from_list(lst: Iterable[Optional[T]]) -> List[T]:
    return [elt for elt in lst if elt is not None]


def _violations_of_domain_pair_constraint(domains: Iterable[Domain],
                                          constraint: DomainPairConstraint,
                                          domains_changed: Optional[Iterable[Domain]],
                                          current_weight_gap: Optional[float],
                                          weigh_constraint_violations_equally: bool,
                                          ) -> Tuple[Dict[Domain, OrderedSet[_Violation]], bool]:
    # If specified, current_weight_gap is the current difference between the weight of violated constraints
    # that have been found so far in the current iteration, compared to the total weight of violated
    # constraints in the optimal solution so far. It is positive
    # (i.e., total_weight_opt - total_weight_cur_so_far)
    # If specified and it is discovered while looping in this function that total_weight_cur_so_far plus
    # the weight of violated constraints discovered in this function exceeds total_weight_opt, quit early.
    domain_pairs_to_check: Sequence[Tuple[Domain, Domain]] = \
        _determine_domain_pairs_to_check(domains, domains_changed, constraint)

    if log_names_of_domains_and_strands_checked:
        logger.debug(f'$ for domain pair constraint {constraint.description}, checking these domain pairs:')
        logger.debug(f'$ {pprint.pformat(domain_pairs_to_check, indent=pprint_indent)}')

    violating_domain_pairs_weights: List[Optional[Tuple[Domain, Domain, float]]] = []

    weight_discovered_here: float = 0.0
    quit_early = False

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
            weight: float = constraint((domain1, domain2))
            if weight > 0.0:
                violating_domain_pairs_weights.append((domain1, domain2, weight))
                if current_weight_gap is not None:
                    weight_discovered_here += constraint.weight * weight
                    if _is_significantly_greater(weight_discovered_here, current_weight_gap):
                        quit_early = True
                        break
    else:
        logger.debug(f'using threading for domain pair constraint {constraint.description}')

        def domain_pair_if_violates(domain_pair: Tuple[Domain, Domain]) \
                -> Optional[Tuple[Domain, Domain, float]]:
            # return domain pair if it violates the constraint, else None
            weight_: float = constraint(domain_pair)
            if weight_ > 0.0:
                return domain_pair[0], domain_pair[1], weight_
            else:
                return None

        if current_weight_gap is None:
            violating_domain_pairs_weights = list(
                _thread_pool.map(domain_pair_if_violates, domain_pairs_to_check))
        else:
            chunks = dc.chunker(domain_pairs_to_check, chunk_size)
            for domain_chunk in chunks:
                violating_domain_pairs_chunk_with_none = \
                    _thread_pool.map(domain_pair_if_violates, domain_chunk)
                violating_domain_pairs_weights_chunk = \
                    remove_none_from_list(violating_domain_pairs_chunk_with_none)
                violating_domain_pairs_weights.extend(violating_domain_pairs_weights_chunk)

                # quit early if possible
                total_weight_chunk: float = sum(
                    domain_pair_weight[2]
                    for domain_pair_weight in violating_domain_pairs_weights_chunk
                    if domain_pair_weight is not None)
                weight_discovered_here += constraint.weight * total_weight_chunk
                if _is_significantly_greater(weight_discovered_here, current_weight_gap):
                    quit_early = True
                    break

    violations: Dict[Domain, OrderedSet[_Violation]] = defaultdict(OrderedSet)
    violating_domain_pair_weight: Optional[Tuple[Domain, Domain, float]]
    for violating_domain_pair_weight in violating_domain_pairs_weights:
        if violating_domain_pair_weight is not None:
            domain1, domain2, weight = violating_domain_pair_weight
            weight_to_use = 1.0 if weigh_constraint_violations_equally else weight
            unfixed_domains_set: Set[Domain] = set()
            if not domain1.fixed:
                unfixed_domains_set.add(domain1)
            if not domain2.fixed:
                unfixed_domains_set.add(domain2)
            violation = _Violation(constraint, frozenset(unfixed_domains_set), weight_to_use)
            if not domain1.fixed:
                violations[domain1].add(violation)
            if not domain2.fixed:
                violations[domain2].add(violation)

    return violations, quit_early


def _violations_of_strand_pair_constraint(strands: Iterable[Strand],
                                          constraint: StrandPairConstraint,
                                          domains_changed: Optional[Iterable[Domain]],
                                          current_weight_gap: Optional[float],
                                          weigh_constraint_violations_equally: bool,
                                          ) -> Tuple[Dict[Domain, OrderedSet[_Violation]], bool]:
    strand_pairs_to_check: Sequence[Tuple[Strand, Strand]] = \
        _determine_strand_pairs_to_check(strands, domains_changed, constraint)

    if log_names_of_domains_and_strands_checked:
        logger.debug(f'$ for strand pair constraint {constraint.description}, checking these strand pairs:')
        logger.debug(f'$ {pprint.pformat(strand_pairs_to_check, indent=pprint_indent)}')

    violating_strand_pairs_weights: List[Optional[Tuple[Strand, Strand, float]]] = []

    weight_discovered_here: float = 0.0
    quit_early = False

    cpu_count = dc.cpu_count() if constraint.threaded else 1

    chunk_size = cpu_count

    if (not constraint.threaded
            or cpu_count == 1
            or (current_weight_gap is not None and chunk_size == 1)):
        logger.debug(f'NOT using threading for strand pair constraint {constraint.description}')
        for strand1, strand2 in strand_pairs_to_check:
            assert not strand1.fixed or not strand2.fixed
            weight = constraint((strand1, strand2))
            if weight > 0.0:
                violating_strand_pairs_weights.append((strand1, strand2, weight))
                if current_weight_gap is not None:
                    weight_discovered_here += constraint.weight * weight
                    if _is_significantly_greater(weight_discovered_here, current_weight_gap):
                        quit_early = True
                        break
    else:
        logger.debug(f'NOT using threading for strand pair constraint {constraint.description}')

        def strand_pair_weight_if_violates(strand_pair: Tuple[Strand, Strand]) \
                -> Optional[Tuple[Strand, Strand, float]]:
            # return strand pair if it violates the constraint, else None
            weight_ = constraint(strand_pair)
            if weight_ > 0.0:
                return strand_pair[0], strand_pair[1], weight_
            else:
                return None

        if current_weight_gap is None:
            violating_strand_pairs_weights = list(
                _thread_pool.map(strand_pair_weight_if_violates, strand_pairs_to_check))
        else:
            chunks = dc.chunker(strand_pairs_to_check, chunk_size)
            for strand_chunk in chunks:
                violating_strand_pairs_chunk_with_none: List[Optional[Tuple[Strand, Strand, float]]] = \
                    _thread_pool.map(strand_pair_weight_if_violates, strand_chunk)
                violating_strand_pairs_chunk: List[Tuple[Strand, Strand, float]] = \
                    remove_none_from_list(violating_strand_pairs_chunk_with_none)
                violating_strand_pairs_weights.extend(violating_strand_pairs_chunk)

                # quit early if possible
                total_weight_chunk = sum(
                    domain_pair_weight[2]
                    for domain_pair_weight in violating_strand_pairs_chunk
                    if domain_pair_weight is not None)
                weight_discovered_here += constraint.weight * total_weight_chunk
                if _is_significantly_greater(weight_discovered_here, current_weight_gap):
                    quit_early = True
                    break

    violations: Dict[Domain, OrderedSet[_Violation]] = defaultdict(OrderedSet)
    violating_strand_pair_weight: Optional[Tuple[Strand, Strand, float]]
    for violating_strand_pair_weight in violating_strand_pairs_weights:
        if violating_strand_pair_weight is not None:
            strand1, strand2, weight = violating_strand_pair_weight
            weight_to_use = 1.0 if weigh_constraint_violations_equally else weight
            unfixed_domains_set = frozenset(strand1.unfixed_domains() + strand2.unfixed_domains())
            violation = _Violation(constraint, unfixed_domains_set, weight_to_use)
            for domain in unfixed_domains_set:
                violations[domain].add(violation)

    return violations, quit_early


def _sequences_fragile_format_output_to_file(design: Design,
                                             include_group: bool = True) -> str:
    return '\n'.join(
        f'{strand.name}  '
        f'{strand.group.name if include_group else ""}  '
        f'{strand.sequence(spaces_between_domains=True)}' for strand in design.strands)


def _write_sequences(design: Design, directory_intermediate: str, directory_final: str,
                     filename_with_iteration: str, filename_final: str,
                     include_group: bool = True) -> None:
    content_fragile_format = _sequences_fragile_format_output_to_file(design, include_group)
    for directory, filename in zip([directory_intermediate, directory_final],
                                   [filename_with_iteration, filename_final]):
        path = os.path.join(directory, filename)
        with open(path, 'w') as file:
            file.write(content_fragile_format)


def _write_dsd_design_json(design: Design, directory_intermediate: str, directory_final: str,
                           filename_with_iteration_no_ext: str, filename_final_no_ext: str) -> None:
    json_str = design.to_json()
    for directory, filename in zip([directory_intermediate, directory_final],
                                   [filename_with_iteration_no_ext, filename_final_no_ext]):
        filename += '.json'
        path = os.path.join(directory, filename)
        with open(path, 'w') as file:
            file.write(json_str)


def _write_report(design: Design, directory_intermediate: str, directory_final: str,
                  filename_with_iteration: str, filename_final: str,
                  report_only_violations: bool) -> None:
    #     sequences = _sequences_fragile_format_output_to_file(design, include_group=True)
    #     sequences_content = f'''\
    # Design
    # ======
    # {sequences}
    #
    # '''

    report_str = design.summary_of_constraints(report_only_violations)
    report = f'''\
Report on constraints
=====================
{report_str}
'''

    for directory, filename in zip([directory_intermediate, directory_final],
                                   [filename_with_iteration, filename_final]):
        path = os.path.join(directory, filename)
        with open(path, 'w') as file:
            # file.write(sequences_content)
            file.write(report)


def _clear_directory(directory: str, force_overwrite: bool) -> None:
    files_relative = os.listdir(directory)
    files_and_directories = [os.path.join(directory, file) for file in files_relative]

    if len(files_and_directories) > 0 and not force_overwrite:
        warning = f'''\
The directory {directory} 
is not empty. Its files and subdirectories will be deleted before continuing. 
To restart a previously cancelled run starting from the files currently in 
{directory}, 
call search_for_dna_sequences with the parameter restart=True.
'''
        print(warning)
        done = False
        while not done:
            ans = input(f'Are you sure you wish to proceed with deleting the contents of\n'
                        f'{directory}? [n]/y ')
            ans = ans.strip().lower()
            if ans in ['n', '']:
                print('No problem! Exiting...')
                sys.exit(0)
            if ans == 'y':
                done = True
            else:
                print(f'I don\'t understand the response "{ans}". '
                      f'Please respond n (for no) or y (for yes).')

    files = [file for file in files_and_directories if os.path.isfile(file)]
    subdirs = [subdir for subdir in files_and_directories if not os.path.isfile(subdir)]
    for file in files:
        logger.info(f'deleting file {file}')
        os.remove(file)
    for sub in subdirs:
        logger.info(f'deleting subdirectory {sub}')
        shutil.rmtree(sub)


@dataclass
class Directories:
    # Container for various directories and files associated with output from the search.
    # Easier than passing around several strings as parameters/return values.
    out: str
    dsd_design: str = field(init=False)
    report: str = field(init=False)
    sequence: str = field(init=False)
    dsd_design_subdirectory: str = field(init=False, default='dsd_designs')
    report_subdirectory: str = field(init=False, default='reports')
    sequence_subdirectory: str = field(init=False, default='sequences')
    dsd_design_filename_no_ext: str = field(init=False, default='design')
    sequences_filename_no_ext: str = field(init=False, default='sequences')
    report_filename_no_ext: str = field(init=False, default='report')
    debug_file_handler: Optional[logging.FileHandler] = field(init=False, default=None)
    info_file_handler: Optional[logging.FileHandler] = field(init=False, default=None)

    def __init__(self, out: str, debug: bool, info: bool) -> None:
        self.out = out
        self.dsd_design = os.path.join(self.out, self.dsd_design_subdirectory)
        self.report = os.path.join(self.out, self.report_subdirectory)
        self.sequence = os.path.join(self.out, self.sequence_subdirectory)

        if debug:
            self.debug_file_handler = logging.FileHandler(os.path.join(self.out, 'log_debug.log'))
            self.debug_file_handler.setLevel(logging.DEBUG)
            dc.logger.addHandler(self.debug_file_handler)

        if info:
            self.info_file_handler = logging.FileHandler(os.path.join(self.out, 'log_info.log'))
            self.info_file_handler.setLevel(logging.INFO)
            dc.logger.addHandler(self.info_file_handler)


def _check_design(design: dc.Design) -> Dict[Domain, Strand]:
    # verify design is legal, and also build map of non-independent domains to the strand that contains them,
    # to help when changing DNA sequences for those Domains.

    domain_to_strand: Dict[dc.Domain, dc.Strand] = {}

    for strand in design.strands:
        if strand.pool is not None:
            for domain in strand.domains:
                domain.dependent = True
                if domain.fixed:
                    raise ValueError(f'for strand {strand.name}, Strand.pool is not None, but it has a '
                                     f'domain {domain.name} that is fixed.\nNone of the domains can be '
                                     f'fixed on a strand with a StrandPool.')
                if domain in domain_to_strand.keys():
                    other_strand = domain_to_strand[domain]
                    raise ValueError(f'strand {strand.name}, which has a StrandPool, has domain {domain}. '
                                     f'But another strand {other_strand.name} also has a StrandPool, and it '
                                     f'shares the domain.\nA strand with a StrandPool may only share domains '
                                     f'with strands have have no StrandPool.')
                domain_to_strand[domain] = strand
        else:
            for domain in strand.domains:
                if domain.pool_ is None and not domain.fixed:
                    raise ValueError(f'for strand {strand.name}, Strand.pool is None, but it has a '
                                     f'non-fixed domain {domain.name} with a DomainPool set to None.\n'
                                     f'For non-fixed domains, exactly one of these must be None.')
                elif domain.fixed and domain.pool_ is not None:
                    raise ValueError(f'for strand {strand.name}, it has a '
                                     f'domain {domain.name} that is fixed, even though that Domain has a '
                                     f'DomainPool.\nA Domain cannot be fixed and have a DomainPool.')

    return domain_to_strand


def search_for_dna_sequences(*, design: dc.Design,
                             probability_of_keeping_change: Optional[Callable[[float], float]] = None,
                             random_seed: Optional[int] = None,
                             never_increase_weight: Optional[bool] = None,
                             out_directory: Optional[str] = None,
                             weigh_violations_equally: bool = False,
                             report_delay: float = 60.0,
                             on_improved_design: Callable[[int], None] = lambda _: None,
                             restart: bool = False,
                             force_overwrite: bool = False,
                             debug_log_file: bool = False,
                             info_log_file: bool = False,
                             report_only_violations: bool = True,
                             max_iterations: Optional[int] = None,
                             max_domains_to_change: int = 1,
                             num_digits_update: Optional[int] = None,
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

    Whenever a new optimal sequence assignment is found, the following are written to files:
    - DNA sequences of each strand are written to a text file .
    - the whole dsd design
    - a report on the DNA sequences indicating how well they do on constraints.

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
    :param out_directory:
        Directory in which to write output files (report on constraint violations and DNA sequences)
        whenever a new optimal sequence assignment is found.
    :param random_seed:
        Integer given as a random seed to the numpy random number generator, used for
        all random choices in the algorithm. Set this to a fixed value to allow reproducibility.
    :param weigh_violations_equally:
        Constraints, when checking a Domain, Strand, pair of Strands, etc., return a nonnegative float whose
        interpretation is "how bad was the violation" if positive, and "constraint passed" if 0.0.
        The total weight assigned to the violation is then this value times :py:data:`Constraint.weight`.
        If `weigh_violations_equally` is True, then all positive values are treated as though
        they are equal to 1.0, so all are assigned a weight of :py:data:`constraints.Constraint.weight`.
        For example, with a :any:`StrandConstraint` checking that energy is above -2.0, an energy of
        -2.01 and -8.0 would be weighted the same if `weigh_constraint_violations_equally` is True,
        but the latter weighted 6 (compared to 0.01 for the former) if `weigh_constraint_violations_equally`
        is False.

        Note that this does **NOT** mean "*weigh all types of constraints equally*"; the value
        :py:data:`constraints.Constraint.weight` is still used.
        But if `weigh_violations_equally` is True,
        then the value returned when calling the :any:`Constraint` is converted to 1.0 if it is positive,
        before being multiplied by :py:data:`constraints.Constraint.weight`.

        The idea behind the default of False is that there is a "smoother gradient" to try to descend
        if violations are weighted by "how bad" they are.
        However, in practice, it can sometimes find a solution faster if all violations are treated the same.
    :param report_delay:
        Every time the design improves, a report on the constraints is written, as long as it has been as
        `report_delay` seconds since the last report was written. Since writing a report requires evaluating
        all constraints, it requires more time than a single iteration, which requires evaluating only those
        constraints involving the :any:`constraints.Domain` whose DNA sequence was changed.
        Thus the default value of 60 seconds avoids spending too much time writing reports, since the
        search finds many new improved designs frequently at the start of the search.
        By setting this to 0, a new report will be written every time the design improves.
    :param on_improved_design:
        Function to call whenever the design improves. Takes an integer as input indicating the number
        of times the design has improved.
    :param restart:
        If this function was previous called and placed files in `out_directory`, calling with this
        parameter True will re-start the search at that point.
    :param force_overwrite:
        If `restart` is False and there are files/subdirectories in `out_directory`,
        then the user will be prompted to confirm that they want to delete these,
        UNLESS force_overwrite is True.
    :param debug_log_file:
        If True, a very detailed log of events is written to the file debug.log in the directory
        `out_directory`. If run for several hours, this file can grow to hundreds of megabytes.
    :param info_log_file:
        By default, the text written to the screen through logger.info (on the logger instance used in
        dsd.constraints) is written to the file log_info.log in the directory `out_directory`.
    :param report_only_violations:
        If True, does not give report on each constraint that was satisfied; only reports violations
        and summary of all constraint checks of a certain type (e.g., how many constraint checks there were).
    :param max_iterations:
        Maximum number of iterations of search to perform.
    :param max_domains_to_change:
        Maximum number of :any:`constraint.Domain`'s to change at a time. A number between 1 and
        `max_domains_to_change` is selected uniformly at random, and then that many
        :any:`constraints.Domain`'s are selected proportional to the weight of :any:`constraint.Constraint`'s
        that they violated.
    :param num_digits_update:
        Number of digits to use when writing update number in filenames. By default,
        they will be written using just enough digits for each integer,
        (for example, for sequences)
        sequences-0.txt, sequences-1.txt, ...,
        sequences-9.txt, sequences-10.txt, ...
        If -nd 3 is specified, for instance, they will be written
        sequences-000.txt, sequences-001.txt, ...,
        sequences-009.txt, sequences-010.txt, ...,
        sequences-099.txt, sequences-100.txt, ...,
        sequences-999.txt, sequences-1000.txt, ...,
        i.e., using leading zeros to have exactly 3 digits,
        until the integers are sufficiently large that more digits are required.
    """
    # keys should be the non-independent Domains in this Design, mapping to the unique Strand with a
    # StrandPool that contains them.
    domain_to_strand: Dict[dc.Domain, dc.Strand] = _check_design(design)

    directories = _setup_directories(
        debug=debug_log_file, info=info_log_file, force_overwrite=force_overwrite,
        restart=restart, out_directory=out_directory)

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

    if random_seed is not None:
        logger.info(f'using random seed of {random_seed}; use this same seed to reproduce this run')

    try:
        if not restart:
            assign_sequences_to_domains_randomly_from_pools(design=design, domain_to_strand=domain_to_strand,
                                                            rng=rng, overwrite_existing_sequences=False)
            num_new_optimal = 0
        else:
            num_new_optimal = _restart_from_directory(directories.out, design,
                                                      directories.dsd_design_subdirectory)

        violation_set_opt, domains_opt, weights_opt = _find_violations_and_weigh(
            design=design, weigh_violations_equally=weigh_violations_equally,
            never_increase_weight=never_increase_weight, iteration=-1)

        if not restart:
            # write initial sequences and report
            _write_intermediate_files(design=design, num_new_optimal=0, write_report=True,
                                      directories=directories, report_only_violations=report_only_violations,
                                      num_digits_update=num_digits_update)

        # this helps with logging if we execute no iterations
        violation_set_new = violation_set_opt

        iteration = 0
        time_of_last_improvement: float = -1.0

        while len(violation_set_opt.all_violations) > 0 and \
                (max_iterations is None or iteration < max_iterations):
            _check_cpu_count(cpu_count)

            domains_changed, original_sequences = _reassign_domains(domains_opt, weights_opt,
                                                                    max_domains_to_change,
                                                                    domain_to_strand, rng)

            # evaluate constraints on new Design with domain_to_change's new sequence
            violation_set_new, domains_new, weights_new = _find_violations_and_weigh(
                design=design, weigh_violations_equally=weigh_violations_equally,
                domains_changed=domains_changed, violation_set_old=violation_set_opt,
                never_increase_weight=never_increase_weight, iteration=iteration)

            _debug = False
            # _debug = True
            if _debug:
                _double_check_violations_from_scratch(design, iteration, never_increase_weight,
                                                      violation_set_new, violation_set_opt,
                                                      weigh_violations_equally)

            _log_constraint_summary(design=design,
                                    violation_set_opt=violation_set_opt, violation_set_new=violation_set_new,
                                    iteration=iteration, num_new_optimal=num_new_optimal)

            # based on total weight of new constraint violations compared to optimal assignment so far,
            # decide whether to keep the change
            weight_delta = violation_set_new.total_weight() - violation_set_opt.total_weight()
            prob_keep_change = probability_of_keeping_change(weight_delta)
            keep_change = rng.random() < prob_keep_change

            if not keep_change:
                _unassign_domains(domains_changed, original_sequences)
            else:
                # keep new sequence and update information about optimal solution so far
                domains_opt = domains_new
                weights_opt = weights_new
                violation_set_opt = violation_set_new
                if weight_delta < 0:  # increment whenever we actually improve the design
                    num_new_optimal += 1
                    on_improved_design(num_new_optimal)

                    current_time: float = time.time()
                    write_report = False
                    # don't write report unless it is
                    #   the first iteration (time_of_last_improvement < 0 ),
                    #   the last iteration (len(violation_set_opt.all_violations) == 0), or
                    #   it has been more than report_delay seconds since writing the last report
                    if (time_of_last_improvement < 0
                            or len(violation_set_opt.all_violations) == 0
                            or current_time - time_of_last_improvement >= report_delay):
                        time_of_last_improvement = current_time
                        write_report = True

                    _write_intermediate_files(design=design, num_new_optimal=num_new_optimal,
                                              write_report=write_report, directories=directories,
                                              report_only_violations=report_only_violations,
                                              num_digits_update=num_digits_update)

            iteration += 1

        _log_constraint_summary(design=design,
                                violation_set_opt=violation_set_opt, violation_set_new=violation_set_new,
                                iteration=iteration, num_new_optimal=num_new_optimal)

    finally:
        if sys.platform != 'win32':
            _pfunc_killall()
        _thread_pool.close()  # noqa
        _thread_pool.terminate()

    if directories.debug_file_handler is not None:
        dc.logger.removeHandler(directories.debug_file_handler)  # noqa

    if directories.info_file_handler is not None:
        dc.logger.removeHandler(directories.info_file_handler)  # noqa


def _check_cpu_count(cpu_count: int) -> None:
    # alters number of threads in ThreadPool if cpu count changed. (Lets us hot-swap CPUs, e.g.,
    # in Amazon web services, without stopping the program.)
    if cpu_count != dc.cpu_count():
        logger.info(f'number of processes in system changed from {cpu_count} to {dc.cpu_count()}'
                    f'\nallocating new ThreadPool')
        cpu_count = dc.cpu_count()
        global _thread_pool
        _thread_pool.close()
        _thread_pool.terminate()
        _thread_pool = ThreadPool(processes=cpu_count)


def _setup_directories(*, debug: bool, info: bool, force_overwrite: bool, restart: bool,
                       out_directory: str) -> Directories:
    if out_directory is None:
        out_directory = default_output_directory()
    directories = Directories(out=out_directory, debug=debug, info=info)
    if not os.path.exists(directories.out):
        os.makedirs(directories.out)
    if not restart:
        _clear_directory(directories.out, force_overwrite)
    for subdir in [directories.dsd_design, directories.report, directories.sequence]:
        if not os.path.exists(subdir):
            os.makedirs(subdir)
    return directories


def _reassign_domains(domains_opt: List[Domain], weights_opt: List[float], max_domains_to_change: int,
                      domain_to_strand: Dict[Domain, Strand], rng: np.random.Generator,
                      # design: Design
                      ) -> Tuple[List[Domain], Dict[Domain, str]]:
    # pick domain to change, with probability proportional to total weight of constraints it violates
    probs_opt = np.asarray(weights_opt)
    probs_opt /= probs_opt.sum()
    num_domains_to_change = rng.choice(a=range(1, max_domains_to_change + 1))
    domains_changed: List[Domain] = list(rng.choice(a=domains_opt, p=probs_opt, replace=False,
                                                    size=num_domains_to_change))

    # fixed Domains should never be blamed for constraint violation
    assert all(not domain_changed.fixed for domain_changed in domains_changed)

    original_sequences: Dict[Domain, str] = {}
    independent_domains = [domain for domain in domains_changed if not domain.dependent]

    for domain in independent_domains:
        # set sequence of domain_changed to random new sequence from its DomainPool
        assert domain not in original_sequences
        original_sequences[domain] = domain.sequence
        domain.sequence = domain.pool.generate_sequence(rng)

    # for dependent domains, ensure each strand is only changed once
    dependent_domains = [domain for domain in domains_changed if domain.dependent]
    strands_dependent = OrderedSet(domain_to_strand[domain] for domain in dependent_domains)
    for strand in strands_dependent:
        for domain in strand.domains:
            assert domain not in original_sequences
            original_sequences[domain] = domain.sequence
            if domain not in domains_changed:
                domains_changed.append(domain)
        strand.assign_dna_from_pool(rng)

    return domains_changed, original_sequences


def _unassign_domains(domains_changed: Iterable[Domain], original_sequences: Dict[Domain, str]) -> None:
    for domain_changed in domains_changed:
        domain_changed.sequence = original_sequences[domain_changed]


# used for debugging; early on, the algorithm for quitting early had a bug and was causing the search
# to think a new assignment was better than the optimal so far, but a mistake in weight accounting
# from quitting early meant we had simply stopped looking for violations too soon.
def _double_check_violations_from_scratch(design: dc.Design, iteration: int, never_increase_weight: bool,
                                          violation_set_new: _ViolationSet, violation_set_opt: _ViolationSet,
                                          weigh_violations_equally: bool):
    violation_set_new_fs, domains_new_fs, weights_new_fs = _find_violations_and_weigh(
        design=design, weigh_violations_equally=weigh_violations_equally,
        never_increase_weight=never_increase_weight, iteration=iteration)
    # XXX: we shouldn't check that the actual weights are close if quit_early is enabled, because then
    # the total weight found on quitting early will be less than the total weight if not.
    # But uncomment this, while disabling quitting early, to test more precisely for "wrong total weight".
    # import math
    # if not math.isclose(violation_set_new.total_weight(), violation_set_new_fs.total_weight()):
    # Instead, we check whether the total weight lie on different sides of the opt total weight, i.e.,
    # they make different decisions about whether to change to the new assignment
    if (violation_set_new_fs.total_weight()
        > violation_set_opt.total_weight()
        >= violation_set_new.total_weight()) or \
            (violation_set_new_fs.total_weight()
             <= violation_set_opt.total_weight()
             < violation_set_new.total_weight()):
        logger.warning(f'WARNING! There is a bug in dsd.')
        logger.warning(f'total weight opt = {violation_set_opt.total_weight()}')
        logger.warning(f'from scratch, we found {violation_set_new_fs.total_weight()} total weight.')
        logger.warning(f'iteratively, we found  {violation_set_new.total_weight()} total weight.')
        logger.warning(f'This means the iterative search is saying something different about '
                       f'quitting early than the full search. It indicates a bug in dsd.')
        logger.warning(f'This happened on iteration {iteration}.')
        sys.exit(-1)


def script_name_no_ext() -> str:
    """
    :return: Name of the Python script currently running, without the .py extension.
    """
    script_name = os.path.basename(sys.argv[0])
    last_dot_idx = script_name.rfind('.')
    if last_dot_idx >= 0:
        script_name = script_name[:last_dot_idx]
    return script_name


def timestamp() -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    time_str = now.strftime("%Y-%m-%dT%H.%M.%S")
    return time_str


def _restart_from_directory(directory: str, design: dc.Design, dsd_design_subdirectory: str) -> int:
    # returns highest index found
    design_filename, num_new_optimal = _find_latest_design_filename(directory, dsd_design_subdirectory)
    with open(design_filename, 'r') as file:
        design_json_str: str = file.read()
    design_with_sequences = dc.Design.from_json(design_json_str)

    # dc.verify_designs_match(design_from_sc, initial_design, check_fixed=False)
    domains_with_seq = [domain for domain in design_with_sequences.domains if not domain.fixed]
    domains = [domain for domain in design.domains if not domain.fixed]
    domains_with_seq.sort(key=lambda domain: domain.name)
    domains.sort(key=lambda domain: domain.name)

    for domain_with_seq, domain in zip(domains_with_seq, domains):
        domain.sequence = domain_with_seq.sequence

    return num_new_optimal


def _find_latest_design_filename(directory: str, dsd_design_subdirectory: str) -> Tuple[str, int]:
    # return filename with latest new optimal sequences (and index)
    dsd_design_directory = os.path.join(directory, dsd_design_subdirectory)
    filenames = [filename
                 for filename in os.listdir(dsd_design_directory)
                 if os.path.isfile(os.path.join(dsd_design_directory, filename))]

    pattern = re.compile(r'-(\d+)\.json')
    filenames_matching = [filename for filename in filenames if pattern.search(filename)]

    if len(filenames_matching) == 0:
        raise ValueError(f'no files in directory "{dsd_design_directory}" '
                         f'match the pattern "*-<index>.json";\n'
                         f'files:\n'
                         f'{filenames}')

    max_filename = filenames_matching[0]
    max_index_str = pattern.search(max_filename).group(1)
    max_index = int(max_index_str)
    for filename in filenames_matching:
        index_str = pattern.search(filename).group(1)
        index = int(index_str)
        if max_index < index:
            max_index = index
            max_filename = filename

    full_filename = os.path.join(directory, dsd_design_subdirectory, max_filename)
    return full_filename, max_index


def _write_intermediate_files(*, design: dc.Design, num_new_optimal: int, write_report: bool,
                              directories: Directories, report_only_violations: bool,
                              num_digits_update: Optional[int]) -> None:
    num_new_optimal_padded = f'{num_new_optimal}' if num_digits_update is None \
        else f'{num_new_optimal:0{num_digits_update}d}'
    _write_dsd_design_json(design,
                           directory_intermediate=directories.dsd_design,
                           directory_final=directories.out,
                           filename_with_iteration_no_ext=f'{directories.dsd_design_filename_no_ext}'
                                                          f'-{num_new_optimal_padded}',
                           filename_final_no_ext=f'current-best-{directories.dsd_design_filename_no_ext}', )
    _write_sequences(design,
                     directory_intermediate=directories.sequence,
                     directory_final=directories.out,
                     filename_with_iteration=f'{directories.sequences_filename_no_ext}'
                                             f'-{num_new_optimal_padded}.txt',
                     filename_final=f'current-best-{directories.sequences_filename_no_ext}.txt')
    if write_report:
        _write_report(design,
                      directory_intermediate=directories.report,
                      directory_final=directories.out,
                      filename_with_iteration=f'{directories.report_filename_no_ext}'
                                              f'-{num_new_optimal_padded}.txt',
                      filename_final=f'current-best-{directories.report_filename_no_ext}.txt',
                      report_only_violations=report_only_violations)


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


def _find_violations_and_weigh(design: Design,
                               weigh_violations_equally: bool,
                               domains_changed: Optional[Iterable[Domain]] = None,
                               violation_set_old: Optional[_ViolationSet] = None,
                               never_increase_weight: bool = False,
                               iteration: int = -1) \
        -> Tuple[_ViolationSet, List[Domain], List[float]]:
    """
    :param design:
        :any:`Design` to evaluate
    :param domains_changed:
        The :any:`Domain` that just changed;
        if None, then recalculate all constraints,
        otherwise assume no constraints changed that do not involve `domain`
    :param violation_set_old:
        :any:`ViolationSet` to update, assuming `domain_changed` is the only :any:`Domain` that changed
    :param never_increase_weight:
        See _violations_of_constraints for explanation of this parameter.
    :param iteration:
        Current iteration number; useful for debugging (e.g., conditional breakpoints).
    :return:
        Tuple (violations, domains, weights)
            `violations`: dict mapping each domain to list of constraints that they violated
            `domains`:    list of :any:`Domain`'s that caused violations
            `weights`:    list of weights for each :any:`Domain`, in same order the domains appear, giving
                          the total weight of :any:`Constraint`'s violated by the corresponding :any:`Domain`
    """
    stopwatch = Stopwatch()

    violation_set: _ViolationSet = _violations_of_constraints(
        design, never_increase_weight, domains_changed, violation_set_old, weigh_violations_equally,
        iteration)
    # violation_set: _ViolationSet = _violations_of_constraints(design) # uncomment to recompute all violations

    domain_to_weights: Dict[Domain, float] = {
        domain: sum(violation.weight for violation in domain_violations)
        for domain, domain_violations in violation_set.domain_to_violations.items()
    }
    domains = list(domain_to_weights.keys())
    weights = list(domain_to_weights.values())

    stopwatch.stop()

    log_time = False
    if log_time:
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

    weight_opt = violation_set_opt.total_weight()
    weight_new = violation_set_new.total_weight()
    weight_decimals = 2 if weight_opt < 10 else 1
    weight_str = f'{iteration:9}|{num_new_optimal:8}|' \
                 f'{weight_opt :10.{weight_decimals}f}|' \
                 f'{weight_new :10.{weight_decimals}f}|' \
                 f'{violation_set_opt.num_violations():9}|' \
                 f'{violation_set_new.num_violations():9}||'
    all_constraints_str = '|'.join(
        f'{violation_description_counts[constraint.short_description]:{len(constraint.short_description)}}'
        for constraint in all_constraints)
    logger.info(weight_str + all_constraints_str)


def assign_sequences_to_domains_randomly_from_pools(design: Design, domain_to_strand: Dict[Domain, Strand],
                                                    rng: np.random.Generator = dn.default_rng,
                                                    overwrite_existing_sequences: bool = False) -> None:
    """
    Assigns to each :any:`Domain` in this :any:`Design` a random DNA sequence from its
    :any:`DomainPool`, calling :py:meth:`constraints.DomainPool.generate_sequence` to get the sequence.

    This is step #1 in the search algorithm.

    :param design:
        Design to which to assign DNA sequences.
    :param domain_to_strand:
        Indicates, for each dependent :any:`Domain`, the unique :any:`Strand` with a :any:`StrandPool` used
        to assign DNA sequences to the :any:`Strand` (thus also to this :any:`Domain`).
    :param rng:
        numpy random number generator (type returned by numpy.random.default_rng()).
    :param overwrite_existing_sequences:
        Whether to overwrite in this initial assignment any existing sequences for :any:`Domain`'s
        that already have a DNA sequence. The DNA sequence of a :any:`Domain` with
        :py:data:`constraints.Domain.fixed` = True are never overwritten, neither here nor later in the
        search. Non-fixed sequences can be skipped for overwriting on this initial assignment, but they
        are subject to change by the subsequent search algorithm.
    """
    independent_domains = [domain for domain in design.domains if not domain.dependent]
    for domain in independent_domains:
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

    dependent_domains = [domain for domain in design.domains if domain.dependent]
    dependent_strands = OrderedSet(domain_to_strand[domain] for domain in dependent_domains)
    for strand in dependent_strands:
        strand.assign_dna_from_pool(rng)


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
