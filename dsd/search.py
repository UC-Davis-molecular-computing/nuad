"""
Stochastic local search for finding DNA sequences to assign to
:any:`Domain`'s in a :any:`Design` to satisfy all :any:`Constraint`'s.
"""

# Since dsd is distributed with NUPACK, we include the following license
# agreement as required by NUPACK. (http://www.nupack.org/downloads/register)
#
# NUPACK Software License Agreement for Non-Commercial Academic Use and
# Redistribution
# Copyright © 2021 California Institute of Technology. All rights reserved.
#
# Use and redistribution in source form and/or binary form, with or without
# modification, are permitted for non-commercial academic purposes only,
# provided that the following conditions are met:
#
# Redistributions in source form must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# provided with the distribution.
#
# Web applications that use the software in source form or binary form must
# reproduce the above copyright notice, this list of conditions and the
# following disclaimer in online documentation provided with the web
# application.
#
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote derivative works without specific prior
# written permission.
#
# Disclaimer
# This software is provided by the copyright holders and contributors "as is"
# and any express or implied warranties, including, but not limited to, the
# implied warranties of merchantability and fitness for a particular purpose
# are disclaimed.  In no event shall the copyright holder or contributors be
# liable for any direct, indirect, incidental, special, exemplary, or
# consequential damages (including, but not limited to, procurement of
# substitute goods or services; loss of use, data, or profits; or business
# interruption) however caused and on any theory of liability, whether in
# contract, strict liability, or tort (including negligence or otherwise)
# arising in any way out of the use of this software, even if advised of the
# possibility of such damage.

from __future__ import annotations

import json
import math
import itertools
import os
import shutil
import sys
import logging
import pprint
from collections import defaultdict, deque
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

import numpy.random
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
# defined at the top level of a module. The constraints call local functions defined by the user or by us in
# higher-order functions such as rna_duplex_strand_pairs_constraint, so it's not clear how to use Pool.
# There may also be a performance overhead for doing this pickling, but I don't know because I haven't
# tested it.
# from multiprocessing.pool import Pool
# from multiprocessing.pool import ThreadPool
import pathos

from dsd.constraints import Domain, Strand, Design, Constraint, DomainConstraint, StrandConstraint, \
    DomainPairConstraint, StrandPairConstraint, ConstraintWithDomainPairs, ConstraintWithStrandPairs, \
    logger, DesignPart, all_pairs, all_pairs_iterator, ConstraintWithDomains, ConstraintWithStrands, \
    ComplexConstraint, ConstraintWithComplexes, Complex
import dsd.constraints as dc

from dsd.stopwatch import Stopwatch


def new_process_pool(cpu_count: int):
    return pathos.multiprocessing.Pool(processes=cpu_count)
    # return ThreadPool(processes=cpu_count)


_process_pool = new_process_pool(dc.cpu_count())

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

    _unweighted_score: float

    def __init__(self, constraint: Constraint, domains: Iterable[Domain], score: float):
        """
        :param constraint:
            :any:`Constraint` that was violated to result in this
        :param domains:
            :any:`Domain`'s that were involved in violating :py:data:`Violation.constraint`
        :param score:
            total "score" of this violation, typically something like an excess energy over a
            threshold, squared, multiplied by the :data:`Constraint.weight`
        """
        object.__setattr__(self, 'constraint', constraint)
        domains_frozen = frozenset(domains)
        object.__setattr__(self, 'domains', domains_frozen)
        object.__setattr__(self, '_unweighted_score', score)

    @property
    def score(self) -> float:
        return self.constraint.weight * self._unweighted_score

    def __repr__(self) -> str:
        return f'Violation({self.constraint.short_description}, score={self._unweighted_score:.2f})'

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

    def total_score(self) -> float:
        """
        :return: Total score of all violations.
        """
        return sum(violation.score for violation in self.all_violations)

    def score_of_constraint(self, constraint: Constraint) -> float:
        """
        :param constraint:
            constraint to filter scores on
        :return:
            Total score of all violations due to `constraint`.
        """
        return sum(violation.score for violation in self.all_violations if violation.constraint == constraint)

    def num_violations(self) -> float:
        """
        :return: Total number of violations.
        """
        return len(self.all_violations)


def _violations_of_constraints(design: Design,
                               never_increase_score: bool,
                               domains_changed: Optional[Iterable[Domain]],
                               violation_set_old: Optional[_ViolationSet],
                               iteration: int,
                               ) -> _ViolationSet:
    """
    :param design:
        The :any:`Design` for which to find DNA sequences.
    :param domains_changed:
        The :any:`Domain`'s that just changed; if None, then recalculate all constraints, otherwise assume no
        constraints changed that do not involve a :any:`Domain` in `domains_changed`.
    :param violation_set_old:
        :any:`ViolationSet` to update, assuming `domain_changed` is the only :any:`Domain` that changed.
    :param never_increase_score:
        Indicates whether the search algorithm is using an update rule that never increases the total score
        of violations (i.e., it only goes downhill). If so we can optimize and stop this function early as
        soon as we find that the violations discovered so far exceed the total score of the current optimal
        solution. In later stages of the search, when the optimal solution so far has very few violated
        constraints, this vastly speeds up the search by allowing most of the constraint checking to be
        skipping for most choices of DNA sequences to `domain_changed`.
    :param iteration:
        Current iteration number; useful for debugging (e.g., conditional breakpoints).
    :return:
        dict mapping each :any:`Domain` to the list of constraints it violated
    """

    if iteration > 0:
        pass  # to quiet PEP warnings

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
        violation_set = violation_set_old.clone()  # Keep old in case no improvement
        for domain_changed in domains_changed:
            assert not domain_changed.fixed
            violation_set.remove_violations_of_domain(domain_changed)

    # individual domain constraints across all domains in Design
    for domain_constraint in design.domain_constraints:
        domains_to_check = _determine_domains_to_check(design.domains, domains_changed, domain_constraint)

        current_score_gap = violation_set_old.total_score() - violation_set.total_score() \
            if never_increase_score and violation_set_old is not None else None

        domain_violations, quit_early_in_func = _violations_of_domain_constraint(
            domains=domains_to_check, constraint=domain_constraint, current_score_gap=current_score_gap)
        violation_set.update(domain_violations)

        quit_early = _quit_early(never_increase_score, violation_set, violation_set_old)
        assert quit_early == quit_early_in_func
        if quit_early:
            return violation_set

    # individual strand constraints across all strands in Design
    for strand_constraint in design.strand_constraints:
        strands_to_check = _determine_strands_to_check(design.strands, domains_changed, strand_constraint)

        current_score_gap = violation_set_old.total_score() - violation_set.total_score() \
            if never_increase_score and violation_set_old is not None else None

        strand_violations, quit_early_in_func = _violations_of_strand_constraint(
            strands=strands_to_check, constraint=strand_constraint, current_score_gap=current_score_gap)
        violation_set.update(strand_violations)

        quit_early = _quit_early(never_increase_score, violation_set, violation_set_old)
        assert quit_early == quit_early_in_func
        if quit_early:
            return violation_set

    # pairs of domains
    for domain_pair_constraint in design.domain_pair_constraints:
        current_score_gap = violation_set_old.total_score() - violation_set.total_score() \
            if never_increase_score and violation_set_old is not None else None
        domain_pair_violations, quit_early_in_func = _violations_of_domain_pair_constraint(
            domains=design.domains, constraint=domain_pair_constraint, domains_changed=domains_changed,
            current_score_gap=current_score_gap)
        violation_set.update(domain_pair_violations)

        quit_early = _quit_early(never_increase_score, violation_set, violation_set_old)
        assert quit_early == quit_early_in_func
        if quit_early:
            return violation_set

    # pairs of strands
    for strand_pair_constraint in design.strand_pair_constraints:
        current_score_gap = violation_set_old.total_score() - violation_set.total_score() \
            if never_increase_score and violation_set_old is not None else None
        strand_pair_violations, quit_early_in_func = _violations_of_strand_pair_constraint(
            strands=design.strands, constraint=strand_pair_constraint, domains_changed=domains_changed,
            current_score_gap=current_score_gap)
        violation_set.update(strand_pair_violations)

        quit_early = _quit_early(never_increase_score, violation_set, violation_set_old)
        assert quit_early == quit_early_in_func
        if quit_early:
            return violation_set

    # complexes
    for complex_constraint in design.complex_constraints:
        current_score_gap = violation_set_old.total_score() - violation_set.total_score() \
            if never_increase_score and violation_set_old is not None else None
        complex_violations, quit_early_in_func = _violations_of_complex_constraint(
            constraint=complex_constraint, domains_changed=domains_changed,
            current_score_gap=current_score_gap)
        violation_set.update(complex_violations)

        quit_early = _quit_early(never_increase_score, violation_set, violation_set_old)
        assert quit_early == quit_early_in_func
        if quit_early:
            return violation_set

    # constraints that process each domain, but all at once (e.g., to hand off in batch to RNAduplex)
    for domains_constraint in design.domains_constraints:
        domains_to_check = _determine_domains_to_check(design.domains, domains_changed, domains_constraint)

        if log_names_of_domains_and_strands_checked:
            logger.debug(f'$ for domains constraint {domains_constraint.description}, '
                         f'checking these domains:\n'
                         f'${pprint.pformat(domains_to_check, indent=pprint_indent)}')

        sets_of_violating_domains_weights = domains_constraint(domains_to_check)
        domains_violations = _convert_sets_of_violating_domains_to_violations(
            domains_constraint, sets_of_violating_domains_weights)
        violation_set.update(domains_violations)

        quit_early = _quit_early(never_increase_score, violation_set, violation_set_old)
        if quit_early:
            return violation_set

    # constraints that process each strand, but all at once (e.g., to hand off in batch to RNAduplex)
    for strands_constraint in design.strands_constraints:
        strands_to_check = _determine_strands_to_check(design.strands, domains_changed, strands_constraint)

        if log_names_of_domains_and_strands_checked:
            logger.debug(f'$ for strands constraint {strands_constraint.description}, '
                         f'checking these strands:\n'
                         f'${pprint.pformat(strands_to_check, indent=pprint_indent)}')

        if len(strands_to_check) > 0:
            sets_of_violating_domains_weights = strands_constraint(strands_to_check)
            domains_violations = _convert_sets_of_violating_domains_to_violations(
                strands_constraint, sets_of_violating_domains_weights)
            violation_set.update(domains_violations)

            quit_early = _quit_early(never_increase_score, violation_set, violation_set_old)
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
                domain_pairs_constraint, sets_of_violating_domains_weights)
            violation_set.update(domains_violations)

            quit_early = _quit_early(never_increase_score, violation_set, violation_set_old)
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
                strand_pairs_constraint, sets_of_violating_domains_weights)
            violation_set.update(domains_violations)

            quit_early = _quit_early(never_increase_score, violation_set, violation_set_old)
            if quit_early:
                return violation_set

    # constraints that processes whole design at once (for anything not captured by the above, e.g.,
    # processing all triples of strands)
    for design_constraint in design.design_constraints:
        sets_of_violating_domains_weights = design_constraint(design, domains_changed)
        domains_violations = _convert_sets_of_violating_domains_to_violations(
            design_constraint, sets_of_violating_domains_weights)
        violation_set.update(domains_violations)

        quit_early = _quit_early(never_increase_score, violation_set, violation_set_old)
        if quit_early:
            return violation_set

    return violation_set


def _is_significantly_greater(x: float, y: float) -> bool:
    # epsilon = min(abs(x), abs(y)) * 0.001
    # XXX: important that this is absolute constant. Sometimes this is called for the total weight of all
    # violations, and sometimes just for the difference between old and new (the latter are smaller).
    # If using relative epsilon, then those can disagree and trigger the assert statement that
    # checks that _violations_of_constraints quit_early agrees with the subroutines it calls.
    epsilon = 0.001
    return x > y + epsilon


def _quit_early(never_increase_score: bool,
                violation_set: _ViolationSet,
                violation_set_old: Optional[_ViolationSet]) -> bool:
    return (never_increase_score and violation_set_old is not None
            and _is_significantly_greater(violation_set.total_score(), violation_set_old.total_score()))


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
    # Note with_replacement is False, i.e., we don't check a domain against itself.
    domain_pairs_to_check_if_domain_changed_none = constraint.pairs if constraint.pairs is not None \
        else all_pairs_iterator(all_domains, with_replacement=False, where=_at_least_one_domain_unfixed)

    # filter out those not containing domain_change if specified
    domain_pairs_to_check = list(domain_pairs_to_check_if_domain_changed_none) if domains_changed is None \
        else [(domain1, domain2) for domain1, domain2 in domain_pairs_to_check_if_domain_changed_none
              if domain1 in domains_changed or domain2 in domains_changed]

    return domain_pairs_to_check


def _determine_domains_to_check(all_domains: Iterable[Domain],
                                domains_changed: Optional[Iterable[Domain]],
                                constraint: ConstraintWithDomains) -> Sequence[Domain]:
    """
    Determines domains to check in `all_domains`.
    If `domains_changed` is None, then this is all that are not fixed if constraint.domains
    is None, otherwise it is constraint.domains.
    If `domains_changed` is not None, then among those domains specified above,
    it is just those in `domains_changed` that appear in `all_domains`.
    """
    # either all pairs, or just constraint.pairs if specified
    domains_to_check_if_domain_changed_none = constraint.domains if constraint.domains is not None \
        else [domain for domain in all_domains if not domain.fixed]

    # filter out those not containing domain_change if specified
    domains_to_check = list(domains_to_check_if_domain_changed_none) if domains_changed is None \
        else [domain for domain in domains_to_check_if_domain_changed_none
              if domain in domains_changed]

    return domains_to_check


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


def _determine_complexes_to_check(domains_changed: Optional[Iterable[Domain]],
                                  constraint: ConstraintWithComplexes) -> \
        Tuple[Complex]:
    """
    Similar to _determine_domain_pairs_to_check but for complexes.
    """
    # filter out those not containing domain_change if specified
    if domains_changed is None:
        return constraint.complexes
    else:
        complexes_to_check: List[Complex] = []
        for strand_complex in constraint.complexes:
            complex_added = False
            for strand in strand_complex:
                for domain_changed in domains_changed:
                    if domain_changed in strand.domains:
                        complexes_to_check.append(strand_complex)
                        complex_added = True
                        break

                if complex_added:
                    # Need to break out of checking each strand in complex since we added complex already
                    break

        return tuple(complexes_to_check)


def _determine_strands_to_check(all_strands: Iterable[Strand],
                                domains_changed: Optional[Iterable[Domain]],
                                constraint: ConstraintWithStrands) -> Sequence[Strand]:
    """
    Similar to _determine_domain_pairs_to_check but for strands.
    """
    # either all pairs, or just constraint.pairs if specified
    strands_to_check_if_domain_changed_none = constraint.strands if constraint.strands is not None \
        else [strand for strand in all_strands if not strand.fixed]

    # filter out those not containing domain_change if specified
    strands_to_check: List[Strand] = []
    if domains_changed is None:
        strands_to_check = strands_to_check_if_domain_changed_none
    else:
        for strand in strands_to_check_if_domain_changed_none:
            for domain_changed in domains_changed:
                if domain_changed in strand.domains:
                    strands_to_check.append(strand)
                    break

    return strands_to_check


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
        constraint: Constraint, sets_of_violating_domains: Iterable[Tuple[OrderedSet[Domain], float]]) \
        -> Dict[Domain, OrderedSet[_Violation]]:
    domains_violations: Dict[Domain, OrderedSet[_Violation]] = defaultdict(OrderedSet)
    for domain_set, score in sets_of_violating_domains:
        violation = _Violation(constraint, domain_set, score)
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
                                     current_score_gap: Optional[float],
                                     ) -> Tuple[Dict[Domain, OrderedSet[_Violation]], bool]:
    violations: Dict[Domain, OrderedSet[_Violation]] = defaultdict(OrderedSet)
    unfixed_domains = [domain for domain in domains if not domain.fixed]
    violating_domains_scores: List[Optional[Tuple[Domain, float]]] = []

    score_discovered_here: float = 0.0
    quit_early = False
    num_threads = dc.cpu_count()
    chunk_size = num_threads

    if log_names_of_domains_and_strands_checked:
        logger.debug(f'$ for domain constraint {constraint.description}, checking these domains:')
        logger.debug(f'{pprint.pformat(unfixed_domains, indent=pprint_indent)}')

    if (not constraint.threaded
            or num_threads == 1
            or len(unfixed_domains) == 1):
        logger.debug(f'NOT using threading for domain constraint {constraint.description}')
        for domain in unfixed_domains:
            score: float = constraint(domain.sequence, domain)
            if score > 0.0:
                violating_domains_scores.append((domain, score))
                if current_score_gap is not None:
                    score_discovered_here += constraint.weight * score
                    if _is_significantly_greater(score_discovered_here, current_score_gap):
                        quit_early = True
                        break
    else:
        logger.debug(f'using threading for domain constraint {constraint.description}')

        domains_to_check = unfixed_domains

        def sequence_to_score(sequence: str) -> float:
            return constraint(sequence, None)

        if current_score_gap is None:
            sequences_to_check = (domain.sequence for domain in domains_to_check)
            scores = list(_process_pool.map(sequence_to_score, sequences_to_check))
            violating_domains_scores = [(domain, score) for domain, score in zip(domains_to_check, scores)
                                        if score > 0]
        else:
            chunks = dc.chunker(domains_to_check, chunk_size)
            for domain_chunk in chunks:
                sequence_chunk = [domain.sequence for domain in domain_chunk]
                scores = list(_process_pool.map(sequence_to_score, sequence_chunk))
                violating_domains_chunk = [(strand, score) for strand, score in zip(domain_chunk, scores)
                                           if score > 0]
                violating_domains_scores.extend(violating_domains_chunk)

                # quit early if possible
                total_score_chunk = sum(score for _, score in violating_domains_chunk)
                score_discovered_here += constraint.weight * total_score_chunk
                if _is_significantly_greater(score_discovered_here, current_score_gap):
                    quit_early = True
                    break

    for violating_domain_score in violating_domains_scores:
        if violating_domain_score is not None:
            violating_domain, score = violating_domain_score
            violation = _Violation(constraint, [violating_domain], score)
            violations[violating_domain].add(violation)

    return violations, quit_early


def _violations_of_strand_constraint(strands: Iterable[Strand],
                                     constraint: StrandConstraint,
                                     current_score_gap: Optional[float],
                                     ) -> Tuple[Dict[Domain, OrderedSet[_Violation]], bool]:
    """
    :param strands:
        Strands to check for violations
    :param constraint:
        Constraint to check.
    :param current_score_gap:
        Current gap between total score of constraint violations found so far and total score of
        optimal design. Used for quitting early.
    :return:
        1. dict mapping each domain to the set of violations that blame it
        2. bool indicating whether we quit early
    """
    strands_to_check = [strand for strand in strands if not strand.fixed]

    violating_strands_scores: List[Tuple[Strand, float]] = []

    score_discovered_here: float = 0.0
    quit_early = False
    num_threads = dc.cpu_count()
    chunk_size = num_threads

    if log_names_of_domains_and_strands_checked:
        logger.debug(f'$ for strand constraint {constraint.description}, checking these strands:')
        logger.debug(f'$ {pprint.pformat(strands_to_check, indent=pprint_indent)}')

    if (not constraint.threaded
            or num_threads == 1
            or len(strands_to_check) == 1
            or (current_score_gap is not None and chunk_size == 1)):
        logger.debug(f'NOT using threading for strand constraint {constraint.description}')
        for strand in strands_to_check:
            score: float = constraint(strand.sequence(), strand)
            if score > 0.0:
                violating_strands_scores.append((strand, score))
                if current_score_gap is not None:
                    score_discovered_here += constraint.weight * score
                    if _is_significantly_greater(score_discovered_here, current_score_gap):
                        quit_early = True
                        break
    else:
        logger.debug(f'using threading for strand constraint {constraint.description}')
        assert constraint.sequence_only  # should have been checked in constraint post_init

        def sequence_to_score(sequence: str) -> float:
            return constraint(sequence, None)

        if current_score_gap is None:
            sequences_to_check = (strand.sequence() for strand in strands_to_check)
            scores = list(_process_pool.map(sequence_to_score, sequences_to_check))
            violating_strands_scores = [(strand, score) for strand, score in zip(strands_to_check, scores)
                                        if score > 0]
        else:
            chunks = dc.chunker(strands_to_check, chunk_size)
            for strand_chunk in chunks:
                sequence_chunk = [strand.sequence() for strand in strand_chunk]
                scores = list(_process_pool.map(sequence_to_score, sequence_chunk))
                violating_strands_chunk = [(strand, score) for strand, score in zip(strand_chunk, scores)
                                           if score > 0]
                violating_strands_scores.extend(violating_strands_chunk)

                # quit early if possible
                total_score_chunk = sum(score for _, score in violating_strands_chunk)
                score_discovered_here += constraint.weight * total_score_chunk
                if _is_significantly_greater(score_discovered_here, current_score_gap):
                    quit_early = True
                    break

    violations: Dict[Domain, OrderedSet[_Violation]] = defaultdict(OrderedSet)
    for strand, score in violating_strands_scores:
        unfixed_domains_set = OrderedSet(strand.unfixed_domains())
        violation = _Violation(constraint, unfixed_domains_set, score)
        for domain in unfixed_domains_set:
            violations[domain].add(violation)

    return violations, quit_early


T = TypeVar('T')


def remove_none_from_list(lst: Iterable[Optional[T]]) -> List[T]:
    return [elt for elt in lst if elt is not None]


def _violations_of_domain_pair_constraint(domains: Iterable[Domain],
                                          constraint: DomainPairConstraint,
                                          domains_changed: Optional[Iterable[Domain]],
                                          current_score_gap: Optional[float],
                                          ) -> Tuple[Dict[Domain, OrderedSet[_Violation]], bool]:
    # If specified, current_score_gap is the current difference between the score of violated constraints
    # that have been found so far in the current iteration, compared to the total score of violated
    # constraints in the optimal solution so far. It is positive
    # (i.e., total_score_opt - total_score_cur_so_far)
    # If specified and it is discovered while looping in this function that total_score_cur_so_far plus
    # the score of violated constraints discovered in this function exceeds total_score_opt, quit early.
    domain_pairs_to_check: Sequence[Tuple[Domain, Domain]] = \
        _determine_domain_pairs_to_check(domains, domains_changed, constraint)

    if log_names_of_domains_and_strands_checked:
        logger.debug(f'$ for domain pair constraint {constraint.description}, checking these domain pairs:')
        logger.debug(f'$ {pprint.pformat(domain_pairs_to_check, indent=pprint_indent)}')

    violating_domain_pairs_scores: List[Optional[Tuple[Domain, Domain, float]]] = []

    score_discovered_here: float = 0.0
    quit_early = False

    cpu_count = dc.cpu_count()

    # since each domain pair check is already parallelized for the four domain pairs
    # (d1,d2), (d1,w2), (w1,d2), (w1,w2), we take smaller chunks
    chunk_size = cpu_count // 4

    if (not constraint.threaded
            or cpu_count == 1
            or (current_score_gap is not None and chunk_size == 1)):
        logger.debug(f'NOT using threading for domain pair constraint {constraint.description}')
        for domain1, domain2 in domain_pairs_to_check:
            assert not domain1.fixed or not domain2.fixed
            assert domain1.name != domain2.name
            score: float = constraint(domain1.sequence, domain2.sequence, domain1, domain2)
            if score > 0.0:
                violating_domain_pairs_scores.append((domain1, domain2, score))
                if current_score_gap is not None:
                    score_discovered_here += constraint.weight * score
                    if _is_significantly_greater(score_discovered_here, current_score_gap):
                        quit_early = True
                        break
    else:
        logger.debug(f'using threading for domain pair constraint {constraint.description}')

        def sequence_pair_to_score(seq_pair: Tuple[str, str]) -> float:
            seq1, seq2 = seq_pair
            return constraint(seq1, seq2, None, None)

        if current_score_gap is None:
            sequence_pairs_to_check = [(domain1.sequence, domain2.sequence)
                                       for domain1, domain2 in domain_pairs_to_check]
            scores = list(_process_pool.map(sequence_pair_to_score, sequence_pairs_to_check))
            violating_domain_pairs_scores = [(domain1, domain2, score) for (domain1, domain2), score in
                                             zip(domain_pairs_to_check, scores) if score > 0]

        else:
            chunks = dc.chunker(domain_pairs_to_check, chunk_size)
            for domain_pair_chunk in chunks:
                sequence_chunk = [(domain1.sequence, domain2.sequence)
                                  for domain1, domain2 in domain_pair_chunk]
                scores = list(_process_pool.map(sequence_pair_to_score, sequence_chunk))
                violating_domain_pairs_chunk = [(domain1, domain2, score) for (domain1, domain2), score in
                                                zip(domain_pair_chunk, scores) if score > 0]
                violating_domain_pairs_scores.extend(violating_domain_pairs_chunk)

                # quit early if possible
                total_score_chunk = sum(score for _, _, score in violating_domain_pairs_chunk)
                score_discovered_here += constraint.weight * total_score_chunk
                if _is_significantly_greater(score_discovered_here, current_score_gap):
                    quit_early = True
                    break

    violations: Dict[Domain, OrderedSet[_Violation]] = defaultdict(OrderedSet)
    violating_domain_pair_score: Optional[Tuple[Domain, Domain, float]]
    for violating_domain_pair_score in violating_domain_pairs_scores:
        if violating_domain_pair_score is not None:
            domain1, domain2, score = violating_domain_pair_score
            unfixed_domains_set: Set[Domain] = set()
            if not domain1.fixed:
                unfixed_domains_set.add(domain1)
            if not domain2.fixed:
                unfixed_domains_set.add(domain2)
            violation = _Violation(constraint, frozenset(unfixed_domains_set), score)
            if not domain1.fixed:
                violations[domain1].add(violation)
            if not domain2.fixed:
                violations[domain2].add(violation)

    return violations, quit_early


def _violations_of_strand_pair_constraint(strands: Iterable[Strand],
                                          constraint: StrandPairConstraint,
                                          domains_changed: Optional[Iterable[Domain]],
                                          current_score_gap: Optional[float],
                                          ) -> Tuple[Dict[Domain, OrderedSet[_Violation]], bool]:
    strand_pairs_to_check: Sequence[Tuple[Strand, Strand]] = \
        _determine_strand_pairs_to_check(strands, domains_changed, constraint)

    if log_names_of_domains_and_strands_checked:
        logger.debug(f'$ for strand pair constraint {constraint.description}, checking these strand pairs:')
        logger.debug(f'$ {pprint.pformat(strand_pairs_to_check, indent=pprint_indent)}')

    violating_strand_pairs_scores: List[Tuple[Strand, Strand, float]] = []

    score_discovered_here: float = 0.0
    quit_early = False
    cpu_count = dc.cpu_count()
    chunk_size = cpu_count

    if (not constraint.threaded
            or cpu_count == 1
            or (current_score_gap is not None and chunk_size == 1)):
        logger.debug(f'NOT using threading for strand pair constraint {constraint.description}')
        for strand1, strand2 in strand_pairs_to_check:
            assert not strand1.fixed or not strand2.fixed
            if constraint.sequence_only:
                score = constraint(strand1.sequence(), strand2.sequence(), None, None)
            else:
                score = constraint(strand1.sequence(), strand2.sequence(), strand1, strand2)
            if score > 0.0:
                violating_strand_pairs_scores.append((strand1, strand2, score))
                if current_score_gap is not None:
                    score_discovered_here += constraint.weight * score
                    if _is_significantly_greater(score_discovered_here, current_score_gap):
                        quit_early = True
                        break
    else:
        logger.debug(f'using threading for strand pair constraint {constraint.description}')

        assert constraint.sequence_only  # should have been checked in StrandPairConstraint post_init

        def sequence_pair_to_score(seq_pair: Tuple[str, str]) -> float:
            seq1, seq2 = seq_pair
            return constraint(seq1, seq2, None, None)

        if current_score_gap is None:
            sequence_pairs_to_check = [(strand1.sequence(), strand2.sequence())
                                       for strand1, strand2 in strand_pairs_to_check]
            scores = list(_process_pool.map(sequence_pair_to_score, sequence_pairs_to_check))
            violating_strand_pairs_scores = [(strand1, strand2, score) for (strand1, strand2), score in
                                             zip(strand_pairs_to_check, scores) if score > 0]

        else:
            chunks = dc.chunker(strand_pairs_to_check, chunk_size)
            for strand_pair_chunk in chunks:
                sequence_chunk = [(strand1.sequence(), strand2.sequence())
                                  for strand1, strand2 in strand_pair_chunk]
                scores = list(_process_pool.map(sequence_pair_to_score, sequence_chunk))
                violating_strand_pairs_chunk = [(strand1, strand2, score) for (strand1, strand2), score in
                                                zip(strand_pair_chunk, scores) if score > 0]
                violating_strand_pairs_scores.extend(violating_strand_pairs_chunk)

                # quit early if possible
                total_score_chunk = sum(score for _, _, score in violating_strand_pairs_chunk)
                score_discovered_here += constraint.weight * total_score_chunk
                if _is_significantly_greater(score_discovered_here, current_score_gap):
                    quit_early = True
                    break

    violations: Dict[Domain, OrderedSet[_Violation]] = defaultdict(OrderedSet)
    for strand1, strand2, score in violating_strand_pairs_scores:
        unfixed_domains_set = OrderedSet(strand1.unfixed_domains() + strand2.unfixed_domains())
        violation = _Violation(constraint, unfixed_domains_set, score)
        for domain in unfixed_domains_set:
            violations[domain].add(violation)

    return violations, quit_early


def _violations_of_complex_constraint(constraint: ComplexConstraint,
                                      domains_changed: Optional[Iterable[Domain]],
                                      current_score_gap: Optional[float],
                                      ) -> Tuple[Dict[Domain, OrderedSet[_Violation]], bool]:
    complexes_to_check: Tuple[Complex] = \
        _determine_complexes_to_check(domains_changed, constraint)

    if log_names_of_domains_and_strands_checked:
        logger.debug(f'$ for complex constraint {constraint.description}, checking these complexes:')
        logger.debug(f'$ {pprint.pformat(complexes_to_check, indent=pprint_indent)}')

    violating_complexes_scores: List[Optional[Tuple[Complex, float]]] = []

    score_discovered_here: float = 0.0
    quit_early = False
    cpu_count = dc.cpu_count()
    chunk_size = cpu_count

    if (not constraint.threaded
            or cpu_count == 1
            or (current_score_gap is not None and chunk_size == 1)):
        logger.debug(f'NOT using threading for strand pair constraint {constraint.description}')
        for strand_complex in complexes_to_check:
            score = constraint(strand_complex)
            if score > 0.0:
                violating_complexes_scores.append((strand_complex, score))
                if current_score_gap is not None:
                    score_discovered_here += constraint.weight * score
                    if _is_significantly_greater(score_discovered_here, current_score_gap):
                        quit_early = True
                        break
            if quit_early:
                # Need to break out of checking each strand in complex since we added complex already
                break
    else:
        logger.debug(f'NOT using threading for strand pair constraint {constraint.description}')

        def complex_score_if_violates(strand_complex_: Complex) \
                -> Optional[Tuple[Complex, float]]:
            # return strand pair if it violates the constraint, else None
            score_ = constraint(strand_complex_)
            if score_ > 0.0:
                return strand_complex_, score_
            else:
                return None

        if current_score_gap is None:
            violating_complexes_scores = list(
                _process_pool.map(complex_score_if_violates, complexes_to_check))
        else:
            chunks = dc.chunker(complexes_to_check, chunk_size)
            for complex_chunk in chunks:
                violating_complexes_chunk_with_none: List[Optional[Tuple[Complex, float]]] = \
                    _process_pool.map(complex_score_if_violates, complex_chunk)
                violating_complexes_chunk: List[Tuple[Complex, float]] = \
                    remove_none_from_list(violating_complexes_chunk_with_none)
                violating_complexes_scores.extend(violating_complexes_chunk)

                # quit early if possible
                total_score_chunk = sum(
                    complex_pair_score[1]
                    for complex_pair_score in violating_complexes_chunk
                    if complex_pair_score is not None)
                score_discovered_here += constraint.weight * total_score_chunk
                if _is_significantly_greater(score_discovered_here, current_score_gap):
                    quit_early = True
                    break

    violations: Dict[Domain, OrderedSet[_Violation]] = defaultdict(OrderedSet)
    violating_complex_score: Optional[Tuple[Complex, float]]
    for violating_complex_score in violating_complexes_scores:
        if violating_complex_score is not None:
            strand_complex, score = violating_complex_score
            unfixed_domains_set_builder = set()
            strand: Strand
            for strand in strand_complex:
                unfixed_domains_set_builder.update(strand.unfixed_domains())
            unfixed_domains_set = frozenset(unfixed_domains_set_builder)
            violation = _Violation(constraint, unfixed_domains_set, score)
            for domain in unfixed_domains_set:
                violations[domain].add(violation)

    return violations, quit_early


def _sequences_fragile_format_output_to_file(design: Design,
                                             include_group: bool = True) -> str:
    return '\n'.join(
        f'{strand.name}  '
        f'{strand.group.name if include_group else ""}  '
        f'{strand.sequence(delimiter="-")}' for strand in design.strands)


def _write_sequences(design: Design, directory_intermediate: str, directory_final: str,
                     filename_with_iteration_no_ext: str, filename_final_no_ext: str,
                     include_group: bool = True) -> None:
    sequences_content = _sequences_fragile_format_output_to_file(design, include_group)
    _write_text_intermediate_and_final_files(directory_final, directory_intermediate,
                                             filename_final_no_ext, filename_with_iteration_no_ext,
                                             sequences_content, '.txt')


def _write_dsd_design_json(design: Design, directories: _Directories,
                           num_new_optimal_padded: str) -> None:
    directory_intermediate = directories.dsd_design
    filename_with_iteration_no_ext = f'{directories.dsd_design_filename_no_ext}-{num_new_optimal_padded}'
    json_str = design.to_json()
    _write_text_intermediate_and_final_files(None, directory_intermediate,
                                             None, filename_with_iteration_no_ext,
                                             json_str, '.json')


def _write_rng_state(rng: numpy.random.Generator, directories: _Directories,
                     num_new_optimal_padded: str) -> None:
    directory_intermediate = directories.rng_state
    filename_with_iteration_no_ext = f'{directories.rng_state_filename_no_ext}-{num_new_optimal_padded}'
    state = rng.bit_generator.state
    json_str = json.dumps(state, indent=2)
    _write_text_intermediate_and_final_files(None, directory_intermediate,
                                             None, filename_with_iteration_no_ext,
                                             json_str, '.json')


def _write_domain_pools(domain_pools: Iterable[dc.DomainPool], directories: _Directories) -> None:
    directory_intermediate = directories.domain_pools

    for pool in domain_pools:
        # to avoid filling drive with large files of sequences, only write domain pool when it resets
        try:
            highest_idx = _find_highest_index_in_directory(directories.domain_pools, pool.name, 'json')
        except ValueError:
            highest_idx = -1

        if highest_idx < pool.num_times_sequences_reset:
            filename_no_ext = f'{pool.name}-{pool.num_times_sequences_reset}'
            domain_pools_json_map = pool.to_json_serializable(include_sequences=True)
            domain_pools_json_str = json.dumps(domain_pools_json_map, indent=2)
            _write_text_intermediate_and_final_files(None, directory_intermediate,
                                                     None, filename_no_ext,
                                                     domain_pools_json_str, '.json')


def _write_text_intermediate_and_final_files(directory_final: Optional[str], directory_intermediate: str,
                                             filename_final_no_ext: Optional[str],
                                             filename_with_iteration_no_ext: str,
                                             content: str, ext: str) -> None:
    for directory, filename in zip([directory_intermediate, directory_final],
                                   [filename_with_iteration_no_ext, filename_final_no_ext]):
        if directory is None or filename is None:
            continue
        full_filename = os.path.join(directory, filename + ext)
        with open(full_filename, 'w') as file:
            file.write(content)


def _write_report(design: Design, directory_intermediate: str, directory_final: str,
                  filename_with_iteration: str, filename_final: str,
                  report_only_violations: bool) -> None:
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
                        f'{directory} ([n]/y)? ')
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
class _Directories:
    # Container for various directories and files associated with output from the search.
    # Easier than passing around several strings as parameters/return values.

    # parent director of all output; typically named after script being run
    out: str

    # directories "fully qualified relative to project root": out joined with "subdirectory" strings below
    dsd_design: str = field(init=False)
    rng_state: str = field(init=False)
    domain_pools: str = field(init=False)
    report: str = field(init=False)
    sequence: str = field(init=False)

    # relative to out directory
    dsd_design_subdirectory: str = field(init=False, default='dsd_designs')
    rng_state_subdirectory: str = field(init=False, default='rng_state')
    domain_pools_subdirectory: str = field(init=False, default='domain_pools')
    report_subdirectory: str = field(init=False, default='reports')
    sequence_subdirectory: str = field(init=False, default='sequences')

    # names of files to write (in subdirectories, and also "current-best" versions in out
    dsd_design_filename_no_ext: str = field(init=False, default='dsd-design')
    rng_state_filename_no_ext: str = field(init=False, default='rng-state')
    sequences_filename_no_ext: str = field(init=False, default='sequences')
    report_filename_no_ext: str = field(init=False, default='report')

    debug_file_handler: Optional[logging.FileHandler] = field(init=False, default=None)
    info_file_handler: Optional[logging.FileHandler] = field(init=False, default=None)

    def all_subdirectories(self) -> List[str]:
        return [self.dsd_design, self.rng_state, self.domain_pools, self.report, self.sequence]

    def __init__(self, out: str, debug: bool, info: bool) -> None:
        self.out = out
        self.dsd_design = os.path.join(self.out, self.dsd_design_subdirectory)
        self.rng_state = os.path.join(self.out, self.rng_state_subdirectory)
        self.domain_pools = os.path.join(self.out, self.domain_pools_subdirectory)
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
    # TODO: we don't seem to use the dictionary any more, so we can probably get rid of that

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
                # noinspection PyProtectedMember
                if domain._pool is None and not domain.fixed:
                    raise ValueError(f'for strand {strand.name}, Strand.pool is None, but it has a '
                                     f'non-fixed domain {domain.name} with a DomainPool set to None.\n'
                                     f'For non-fixed domains, exactly one of these must be None.')
                # noinspection PyProtectedMember
                elif domain.fixed and domain._pool is not None:
                    raise ValueError(f'for strand {strand.name}, it has a '
                                     f'domain {domain.name} that is fixed, even though that Domain has a '
                                     f'DomainPool.\nA Domain cannot be fixed and have a DomainPool.')

    return domain_to_strand


@dataclass
class SearchParameters:
    probability_of_keeping_change: Optional[Callable[[float], float]] = None
    """
    Function giving the probability of keeping a change in one
    :any:`Domain`'s DNA sequence, if the new sequence affects the total score of all violated
    :any:`Constraint`'s by `score_delta`, the input to `probability_of_keeping_change`.
    See :py:meth:`default_probability_of_keeping_change_function` for a description of the default
    behavior if this parameter is not specified.
    """

    random_seed: Optional[int] = None
    """
    Integer given as a random seed to the numpy random number generator, used for
    all random choices in the algorithm. Set this to a fixed value to allow reproducibility.
    """

    never_increase_score: Optional[bool] = None
    """
    If specified and True, then it is assumed that the function
    probability_of_keeping_change returns 0 for any negative value of `score_delta` (i.e., the search
    never goes "uphill"), and the search for violations is optimized to quit as soon as the total score
    of violations exceeds that of the current optimal solution. This vastly speeds up the search in later
    stages, when the current optimal solution is low score. If both `probability_of_keeping_change` and
    `never_increase_score` are left unspecified, then `probability_of_keeping_change` uses the default,
    which never goes uphill, and `never_increase_score` is set to True. If
    `probability_of_keeping_change` is specified and `never_increase_score` is not, then
    `never_increase_score` is set to False. If both are specified and `never_increase_score` is set to
    True, then take caution that `probability_of_keeping_change` really has the property that it never
    goes uphill; the optimization will essentially prevent most uphill climbs from occurring.
    """

    out_directory: Optional[str] = None
    """
    Directory in which to write output files (report on constraint violations and DNA sequences)
    whenever a new optimal sequence assignment is found.
    """

    report_delay: float = 60.0
    """
    Every time the design improves, a report on the constraints is written, as long as it has been as
    `report_delay` seconds since the last report was written. Since writing a report requires evaluating
    all constraints, it requires more time than a single iteration, which requires evaluating only those
    constraints involving the :any:`constraints.Domain` whose DNA sequence was changed.
    Thus the default value of 60 seconds avoids spending too much time writing reports, since the
    search finds many new improved designs frequently at the start of the search.
    By setting this to 0, a new report will be written every time the design improves.
    """

    on_improved_design: Callable[[int], None] = lambda _: None
    """
    Function to call whenever the design improves. Takes an integer as input indicating the number
    of times the design has improved.
    """

    restart: bool = False
    """
    If this function was previously called and placed files in `out_directory`, calling with this
    parameter True will re-start the search at that point.
    """

    force_overwrite: bool = False
    """
    If `restart` is False and there are files/subdirectories in `out_directory`,
    then the user will be prompted to confirm that they want to delete these,
    UNLESS force_overwrite is True.
    """

    debug_log_file: bool = False
    """
    If True, a very detailed log of events is written to the file debug.log in the directory
    `out_directory`. If run for several hours, this file can grow to hundreds of megabytes.
    """

    info_log_file: bool = False
    """
    By default, the text written to the screen through logger.info (on the logger instance used in
    dsd.constraints) is written to the file log_info.log in the directory `out_directory`.
    """

    report_only_violations: bool = True
    """
    If True, does not give report on each constraint that was satisfied; only reports violations
    and summary of all constraint checks of a certain type (e.g., how many constraint checks there were).
    """

    max_iterations: Optional[int] = None
    """
    Maximum number of iterations of search to perform.
    """

    max_domains_to_change: int = 1
    """
    Maximum number of :any:`constraints.Domain`'s to change at a time. A number between 1 and
    `max_domains_to_change` is selected uniformly at random, and then that many
    :any:`constraints.Domain`'s are selected proportional to the score of :any:`constraints.Constraint`'s
    that they violated.
    """

    num_digits_update: Optional[int] = None
    """
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

    warn_fixed_sequences: bool = True
    """
    Log warning about sequences that are fixed, indicating they will not be re-assigned during the search.
    """


def search_for_dna_sequences(design: dc.Design, params: SearchParameters) -> None:
    """
    Search for DNA sequences to assign to each :any:`Domain` in `design`, satisfying the various
    :any:`Constraint`'s associated with `design`.

    **Search algorithm:**
    This is a stochastic local search. It determines which :any:`Constraint`'s are violated.
    More precisely, it adds the total score of all violated constraints
    (sum of :py:data:`constraints.Constraint.weight`*score_of_violation over all violated
    :any:`Constraint`'s).
    The goal is to reduce this total score until it is 0 (i.e., no violated constraints).
    Any :any:`Domain` "involved" in the violated :any:`Constraint` is noted as being one of the
    :any:`Domain`'s responsible for the violation. (For example, if a :any:`DomainConstraint` is violated,
    only one :any:`Domain` is blamed, whereas if a :any:`StrandConstraint` is violated, every :any:`Domain`
    in the :any:`Strand` is blamed.) While any :any:`Constraint`'s are violated, a :any:`Domain` is picked
    at random, with probability proportional to the total score of all the :any:`Constraint`'s
    for which the :any:`Domain` was blamed. A new DNA sequence is assigned to this
    :any:`Domain` by calling :py:meth:`constraints.DomainPool.generate_sequence` on the :any:`DomainPool`
    of that :any:`Domain`. The way to decide whether to keep the changed sequence, or revert to the
    old sequence, is to calculate the total score of all violated constraints in the original and changed
    :any:`Design`, calling their difference `score_delta` = `new_total_score` - `old_total_score`.
    The value ``probability_of_keeping_change(score_delta)`` is the probability that the change
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
    every time a new improve assignment is found. This re-evaluates the entire design, so it can be expensive,
    but in practice the design is strictly improved many fewer times than total iterations.

    Whenever a new optimal sequence assignment is found, the following are written to files:
    - DNA sequences of each strand are written to a text file .
    - the whole dsd design
    - a report on the DNA sequences indicating how well they do on constraints.

    :param design:
        The :any:`Design` containing the :any:`Domain`'s to which to assign DNA sequences
        and the :any:`Constraint`'s that apply to them
    :param params:
        A :any:`SearchParameters` object with attributes that can be called within this function
        for flexibility.

    """

    # keys should be the non-independent Domains in this Design, mapping to the unique Strand with a
    # StrandPool that contains them.
    # domain_to_strand: Dict[dc.Domain, dc.Strand] = _check_design(design)
    _check_design(design)

    directories = _setup_directories(
        debug=params.debug_log_file, info=params.info_log_file, force_overwrite=params.force_overwrite,
        restart=params.restart, out_directory=params.out_directory)

    if params.random_seed is not None:
        rng = np.random.default_rng(params.random_seed)
    else:
        rng = dn.default_rng

    if params.probability_of_keeping_change is None:
        params.probability_of_keeping_change = default_probability_of_keeping_change_function(design)
        if params.never_increase_score is None:
            params.never_increase_score = True
    elif params.never_increase_score is None:
        params.never_increase_score = False

    assert params.never_increase_score is not None

    cpu_count = dc.cpu_count()
    logger.info(f'number of processes in system: {cpu_count}')

    if params.random_seed is not None and not params.restart:
        logger.info(f'using random seed of {params.random_seed}; use this same seed to reproduce this run')

    # need to assign to local function variable so it doesn't look like a method call
    on_improved_design: Callable[[int], None] = params.on_improved_design

    try:
        if not params.restart:
            assign_sequences_to_domains_randomly_from_pools(design=design,
                                                            warn_fixed_sequences=params.warn_fixed_sequences,
                                                            rng=rng,
                                                            overwrite_existing_sequences=False)
            num_new_optimal = 0
        else:
            num_new_optimal, rng, design = _restart_from_directory(directories, design)

        violation_set_opt, domains_opt, scores_opt = _find_violations_and_score(
            design=design, never_increase_score=params.never_increase_score, iteration=-1)

        if not params.restart:
            # write initial sequences and report
            _write_intermediate_files(design=design, rng=rng, num_new_optimal=num_new_optimal,
                                      write_report=True, directories=directories,
                                      report_only_violations=params.report_only_violations,
                                      num_digits_update=params.num_digits_update)

        # this helps with logging if we execute no iterations
        violation_set_new = violation_set_opt

        iteration = 0
        time_of_last_improvement: float = -1.0

        while len(violation_set_opt.all_violations) > 0 and \
                (params.max_iterations is None or iteration < params.max_iterations):
            _check_cpu_count(cpu_count)

            domains_changed, original_sequences = _reassign_domains(domains_opt, scores_opt,
                                                                    params.max_domains_to_change, rng)

            # evaluate constraints on new Design with domain_to_change's new sequence
            violation_set_new, domains_new, scores_new = _find_violations_and_score(
                design=design, domains_changed=domains_changed, violation_set_old=violation_set_opt,
                never_increase_score=params.never_increase_score, iteration=iteration)

            # _double_check_violations_from_scratch(design, iteration, params.never_increase_score,
            #                                       violation_set_new, violation_set_opt)

            _log_constraint_summary(design=design,
                                    violation_set_opt=violation_set_opt, violation_set_new=violation_set_new,
                                    iteration=iteration, num_new_optimal=num_new_optimal)

            # based on total score of new constraint violations compared to optimal assignment so far,
            # decide whether to keep the change
            score_delta = violation_set_new.total_score() - violation_set_opt.total_score()
            prob_keep_change = params.probability_of_keeping_change(score_delta)
            keep_change = rng.random() < prob_keep_change if prob_keep_change < 1 else True

            if not keep_change:
                _unassign_domains(domains_changed, original_sequences)
            else:
                # keep new sequence and update information about optimal solution so far
                domains_opt = domains_new
                scores_opt = scores_new
                violation_set_opt = violation_set_new
                if score_delta < 0:  # increment whenever we actually improve the design
                    num_new_optimal += 1
                    on_improved_design(num_new_optimal)  # type: ignore

                    current_time: float = time.time()
                    write_report = False
                    # don't write report unless it is
                    if (time_of_last_improvement < 0  # first iteration
                            or len(violation_set_opt.all_violations) == 0  # last iteration (search is over)
                            or current_time - time_of_last_improvement >= params.report_delay):  # > report_delay seconds since last report
                        time_of_last_improvement = current_time
                        write_report = True
                        logger.debug('writing report')
                    else:
                        logger.debug('skipping report')

                    _write_intermediate_files(design=design, rng=rng, num_new_optimal=num_new_optimal,
                                              write_report=write_report, directories=directories,
                                              report_only_violations=params.report_only_violations,
                                              num_digits_update=params.num_digits_update)

            iteration += 1

        _log_constraint_summary(design=design,
                                violation_set_opt=violation_set_opt, violation_set_new=violation_set_new,
                                iteration=iteration, num_new_optimal=num_new_optimal)

    finally:
        # if sys.platform != 'win32':
        #     _pfunc_killall()
        _process_pool.close()  # noqa
        _process_pool.terminate()

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
        global _process_pool
        _process_pool.close()
        _process_pool.terminate()
        _process_pool = new_process_pool(cpu_count)


def _setup_directories(*, debug: bool, info: bool, force_overwrite: bool, restart: bool,
                       out_directory: str) -> _Directories:
    if out_directory is None:
        out_directory = default_output_directory()
    directories = _Directories(out=out_directory, debug=debug, info=info)
    if not os.path.exists(directories.out):
        os.makedirs(directories.out)
    if not restart:
        _clear_directory(directories.out, force_overwrite)
    for subdir in directories.all_subdirectories():
        if not os.path.exists(subdir):
            os.makedirs(subdir)
    return directories


def _reassign_domains(domains_opt: List[Domain], scores_opt: List[float], max_domains_to_change: int,
                      rng: np.random.Generator) -> Tuple[List[Domain], Dict[Domain, str]]:
    # pick domain to change, with probability proportional to total score of constraints it violates
    probs_opt = np.asarray(scores_opt)
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
        previous_sequence = domain.sequence
        original_sequences[domain] = previous_sequence
        domain.sequence = domain.pool.generate_sequence(rng, previous_sequence)

    dependent_domains = [domain for domain in domains_changed if domain.dependent]
    for domain in dependent_domains:
        original_sequences[domain] = domain.sequence

    return domains_changed, original_sequences


def _unassign_domains(domains_changed: Iterable[Domain], original_sequences: Dict[Domain, str]) -> None:
    for domain_changed in domains_changed:
        domain_changed.sequence = original_sequences[domain_changed]


# used for debugging; early on, the algorithm for quitting early had a bug and was causing the search
# to think a new assignment was better than the optimal so far, but a mistake in score accounting
# from quitting early meant we had simply stopped looking for violations too soon.
def _double_check_violations_from_scratch(design: dc.Design, iteration: int, never_increase_score: bool,
                                          violation_set_new: _ViolationSet, violation_set_opt: _ViolationSet):
    violation_set_new_fs, domains_new_fs, scores_new_fs = _find_violations_and_score(
        design=design, never_increase_score=never_increase_score, iteration=iteration)
    # XXX: we shouldn't check that the actual scores are close if quit_early is enabled, because then
    # the total score found on quitting early will be less than the total score if not.
    # But uncomment this, while disabling quitting early, to test more precisely for "wrong total score".
    # import math
    # if not math.isclose(violation_set_new.total_score(), violation_set_new_fs.total_score()):
    # Instead, we check whether the total score lie on different sides of the opt total score, i.e.,
    # they make different decisions about whether to change to the new assignment
    if (violation_set_new_fs.total_score()
        > violation_set_opt.total_score()
        >= violation_set_new.total_score()) or \
            (violation_set_new_fs.total_score()
             <= violation_set_opt.total_score()
             < violation_set_new.total_score()):
        logger.warning(f'WARNING! There is a bug in dsd.')
        logger.warning(f'total score opt = {violation_set_opt.total_score()}')
        logger.warning(f'from scratch, we found {violation_set_new_fs.total_score()} total score.')
        logger.warning(f'iteratively, we found  {violation_set_new.total_score()} total score.')
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


def _restart_from_directory(directories: _Directories, design: dc.Design) \
        -> Tuple[int, np.random.Generator, dc.Design]:
    # NOTE: restarts from highest index found in dsd_design subdirectory, NOT from "current-best" files,
    # which are ignored. This applies to both the design and the RNG state

    # get DomainPools
    pool_with_name = read_domain_pools(directories)

    # returns highest index found, as well as name of corresponding design file
    highest_idx_design = _find_highest_index_in_directory(directories.dsd_design,
                                                          directories.dsd_design_filename_no_ext, 'json')
    design_filename = os.path.join(directories.dsd_design,
                                   f'{directories.dsd_design_filename_no_ext}-{highest_idx_design}.json')
    with open(design_filename, 'r') as file:
        design_json_str = file.read()
    design_stored = dc.Design.from_json(design_json_str, pool_with_name=pool_with_name)
    dc.verify_designs_match(design_stored, design, check_fixed=False)

    # get RNG state
    rng_filename = os.path.join(directories.rng_state,
                                f'{directories.rng_state_filename_no_ext}-{highest_idx_design}.json')
    with open(rng_filename, 'r') as file:
        rng_state_json = file.read()
    rng_state = json.loads(rng_state_json)
    rng = numpy.random.default_rng()
    rng.bit_generator.state = rng_state

    # this is really ugly how we do this, taking parts of the design from `design`,
    # parts from `design_stored`, and parts from the stored DomainPools, but this seems to be necessary
    # to avoid writing the entire DomainPool (with its 100,000 sequences) every time we write a Design.
    design_stored.copy_constraints_from(design)

    design_json = json.loads(design_json_str)
    stored_pool_idxs = design_json[dc.domain_pools_num_sampled_key]
    for pool in design_stored.domain_pools():
        idx = stored_pool_idxs[pool.name]
        pool.num_sampled = idx

    return highest_idx_design, rng, design_stored


def read_domain_pools(directories: _Directories) -> Dict[str, dc.DomainPool]:
    # return dict mapping name of DomainPool to DomainPool, read from highest index of file with certain
    # name in directories.domain_pool subdirectory

    # first find pool names from dsd_design files
    highest_idx_design = _find_highest_index_in_directory(directories.dsd_design,
                                                          directories.dsd_design_filename_no_ext, 'json')
    design_filename = os.path.join(directories.dsd_design,
                                   f'{directories.dsd_design_filename_no_ext}-{highest_idx_design}.json')
    with open(design_filename, 'r') as file:
        design_json_str = file.read()
    design_json = json.loads(design_json_str)
    domains_json = design_json[dc.domains_key]
    pool_names: Set[str] = set()
    for domain_json in domains_json:
        if dc.domain_pool_name_key in domain_json:
            pool_name = domain_json[dc.domain_pool_name_key]
            pool_names.add(pool_name)

    pool_with_name = {}
    for pool_name in pool_names:
        highest_idx_pool = _find_highest_index_in_directory(directories.domain_pools,
                                                            pool_name, 'json')
        pool_filename = os.path.join(directories.domain_pools, f'{pool_name}-{highest_idx_pool}.json')
        with open(pool_filename, 'r') as file:
            pool_json_str = file.read()
        pool_json = json.loads(pool_json_str)
        pool = dc.DomainPool.from_json_serializable(pool_json)
        assert pool.name == pool_name
        pool_with_name[pool_name] = pool

    return pool_with_name


def _find_highest_index_in_directory(directory: str, filename_start: str, ext: str) -> int:
    # return highest index of filename (name matches "<filename_start>-<index>.<ext>"
    filenames = [filename
                 for filename in os.listdir(directory)
                 if os.path.isfile(os.path.join(directory, filename))]

    pattern = re.compile(filename_start + r'-(\d+)\.' + ext)
    filenames_matching = [filename for filename in filenames if pattern.search(filename)]

    if len(filenames_matching) == 0:
        raise ValueError(f'no files in directory "{directory}" '
                         f'match the pattern "{filename_start}-<index>.{ext}";\n'
                         f'files:\n'
                         f'{filenames}')

    max_index_str = pattern.search(filenames_matching[0]).group(1)
    max_index = int(max_index_str)
    for filename in filenames_matching:
        index_str = pattern.search(filename).group(1)
        index = int(index_str)
        if max_index < index:
            max_index = index

    return max_index


def _write_intermediate_files(*, design: dc.Design, rng: numpy.random.Generator,
                              num_new_optimal: int, write_report: bool,
                              directories: _Directories, report_only_violations: bool,
                              num_digits_update: Optional[int]) -> None:
    num_new_optimal_padded = f'{num_new_optimal}' if num_digits_update is None \
        else f'{num_new_optimal:0{num_digits_update}d}'

    _write_dsd_design_json(design, directories, num_new_optimal_padded)
    _write_rng_state(rng, directories, num_new_optimal_padded)
    _write_domain_pools(design.domain_pools_to_domain_map.keys(), directories)

    _write_sequences(design,
                     directory_intermediate=directories.sequence,
                     directory_final=directories.out,
                     filename_with_iteration_no_ext=f'{directories.sequences_filename_no_ext}'
                                                    f'-{num_new_optimal_padded}',
                     filename_final_no_ext=f'current-best-{directories.sequences_filename_no_ext}')

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


def _find_violations_and_score(design: Design,
                               domains_changed: Optional[Iterable[Domain]] = None,
                               violation_set_old: Optional[_ViolationSet] = None,
                               never_increase_score: bool = False,
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
    :param never_increase_score:
        See _violations_of_constraints for explanation of this parameter.
    :param iteration:
        Current iteration number; useful for debugging (e.g., conditional breakpoints).
    :return:
        Tuple (violations, domains, scores)
            `violations`: dict mapping each domain to list of constraints that they violated
            `domains`:    list of :any:`Domain`'s that caused violations
            `scores`:    list of scores for each :any:`Domain`, in same order the domains appear, giving
                          the total score of :any:`Constraint`'s violated by the corresponding :any:`Domain`
    """
    stopwatch = Stopwatch()

    violation_set: _ViolationSet = _violations_of_constraints(
        design, never_increase_score, domains_changed, violation_set_old, iteration)

    domain_to_score: Dict[Domain, float] = {
        domain: sum(violation.score for violation in domain_violations)
        for domain, domain_violations in violation_set.domain_to_violations.items()
    }
    domains = list(domain_to_score.keys())
    scores = list(domain_to_score.values())

    stopwatch.stop()

    log_time = False
    if log_time:
        _log_time(stopwatch)

    return violation_set, domains, scores


def _flatten(list_of_lists: Iterable[Iterable[Any]]) -> Iterable[Any]:
    #  Flatten one level of nesting
    return itertools.chain.from_iterable(list_of_lists)


def _log_constraint_summary(*, design: Design,
                            violation_set_opt: _ViolationSet,
                            violation_set_new: _ViolationSet,
                            iteration: int,
                            num_new_optimal: int) -> None:
    all_constraints = design.all_constraints()

    score_header = 'iteration|updates|opt score|new score||'
    all_constraints_header = '|'.join(
        f'{constraint.short_description}' for constraint in all_constraints)
    header = score_header + all_constraints_header
    # logger.info('-' * len(header) + '\n')
    logger.info(header)

    score_opt = violation_set_opt.total_score()
    score_new = violation_set_new.total_score()
    dec_opt = max(1, math.ceil(math.log(1 / score_opt, 10)) + 2) if score_opt > 0 else 1
    dec_new = max(1, math.ceil(math.log(1 / score_new, 10)) + 2) if score_new > 0 else 1
    score_str = f'{iteration:9}|{num_new_optimal:7}|' \
                f'{score_opt :9.{dec_opt}f}|' \
                f'{score_new :9.{dec_new}f}|'  # \

    all_constraints_strs = []
    for constraint in all_constraints:
        score = violation_set_new.score_of_constraint(constraint)
        length = len(constraint.short_description)
        num_decimals = max(1, math.ceil(math.log(1 / score, 10)) + 2) if score > 0 else 1
        constraint_str = f'{score:{length}.{num_decimals}f}'
        all_constraints_strs.append(constraint_str)
    all_constraints_str = '|'.join(all_constraints_strs)

    logger.info(score_str + '|' + all_constraints_str)


def assign_sequences_to_domains_randomly_from_pools(design: Design,
                                                    warn_fixed_sequences: bool,
                                                    rng: np.random.Generator = dn.default_rng,
                                                    overwrite_existing_sequences: bool = False) -> None:
    """
    Assigns to each :any:`Domain` in this :any:`Design` a random DNA sequence from its
    :any:`DomainPool`, calling :py:meth:`constraints.DomainPool.generate_sequence` to get the sequence.

    This is step #1 in the search algorithm.

    :param design:
        Design to which to assign DNA sequences.
    :param warn_fixed_sequences:
        Whether to log warning that each :any:`Domain` with :data:`Domain.fixed` = True is not being
        assigned.
    :param rng:
        numpy random number generator (type returned by numpy.random.default_rng()).
    :param overwrite_existing_sequences:
        Whether to overwrite in this initial assignment any existing sequences for :any:`Domain`'s
        that already have a DNA sequence. The DNA sequence of a :any:`Domain` with
        :py:data:`constraints.Domain.fixed` = True are never overwritten, neither here nor later in the
        search. Non-fixed sequences can be skipped for overwriting on this initial assignment, but they
        are subject to change by the subsequent search algorithm.
    """
    at_least_one_domain_unfixed = False
    independent_domains = [domain for domain in design.domains if not domain.dependent]
    for domain in independent_domains:
        skip_nonfixed_msg = skip_fixed_msg = None
        if warn_fixed_sequences and domain.has_sequence():
            skip_nonfixed_msg = f'Skipping assignment of DNA sequence to domain {domain.name}. ' \
                                f'That domain has a NON-FIXED sequence {domain.sequence}, ' \
                                f'which the search will attempt to replace.'
            skip_fixed_msg = f'Skipping assignment of DNA sequence to domain {domain.name}. ' \
                             f'That domain has a FIXED sequence {domain.sequence}.'
        if overwrite_existing_sequences:
            if not domain.fixed:
                at_least_one_domain_unfixed = True
                domain.sequence = domain.pool.generate_sequence(rng, domain.sequence)
                assert len(domain.sequence) == domain.pool.length
            else:
                logger.info(skip_nonfixed_msg)
        else:
            if not domain.fixed:
                # even though we don't assign a new sequence here, we want to record that at least one
                # domain is not fixed so that we know it is eligible to be overwritten during the search
                at_least_one_domain_unfixed = True
            if not domain.fixed and not domain.has_sequence():
                domain.sequence = domain.pool.generate_sequence(rng)
                assert len(domain.sequence) == domain.pool.length
            elif warn_fixed_sequences:
                if domain.fixed:
                    logger.info(skip_fixed_msg)
                else:
                    logger.info(skip_nonfixed_msg)

    if not at_least_one_domain_unfixed:
        raise ValueError('No domains are unfixed, so we cannot do any sequence design. '
                         'Please make at least one domain not fixed.')


_sentinel = object()


def _iterable_is_empty(iterable: abc.Iterable) -> bool:
    iterator = iter(iterable)
    return next(iterator, _sentinel) is _sentinel


def default_probability_of_keeping_change_function(design: dc.Design) -> Callable[[float], float]:
    """
    Returns a function that takes a float input `score_delta` representing a change in score of
    violated constraint, which returns a probability of keeping the change in the DNA sequence assignment.
    The probability is 1 if the change it is at least as good as the previous
    (roughly, the score change is not positive), and the probability is 0 otherwise.

    To mitigate floating-point rounding errors, the actual condition checked is that
    `score_delta` < :py:data:`epsilon`,
    on the assumption that if the same score of constraints are violated,
    rounding errors in calculating `score_delta` could actually make it slightly above than 0
    and result in reverting to the old assignment when we really want to keep the change.
    If all values of :py:data:`Constraint.score` are significantly about :py:data:`epsilon`
    (e.g., 1.0 or higher), then this should be is equivalent to keeping a change in the DNA sequence
    assignment if and only if it is no worse than the previous.

    :param design: :any:`Design` to apply this rule for; `design` is required because the score of
                   :any:`Constraint`'s in the :any:`Design` are used to calculate an appropriate
                   epsilon value for determining when a score change is too small to be significant
                   (i.e., is due to rounding error)
    :return: the "keep change" function `f`: :math:`\\mathbb{R} \\to [0,1]`,
             where :math:`f(w_\\delta) = 1` if :math:`w_\\delta \\leq \\epsilon`
             (where :math:`\\epsilon` is chosen to be 1,000,000 times smaller than
             the smallest :any:`Constraint.weight` for any :any:`Constraint` in `design`),
             and :math:`f(w_\\delta) = 0` otherwise.
    """
    min_weight = min(constraint.weight for constraint in design.all_constraints())
    epsilon_from_min_weight = min_weight / 1000000.0

    def keep_change_only_if_no_worse(score_delta: float) -> float:
        return 1.0 if score_delta <= epsilon_from_min_weight else 0.0

    # def keep_change_only_if_better(score_delta: float) -> float:
    #     return 1.0 if score_delta <= -epsilon_from_min_weight else 0.0

    return keep_change_only_if_no_worse
    # return keep_change_only_if_better
