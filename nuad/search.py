"""
The main export of the search module is the function :meth:`search_for_dna_sequences`,
which is a stochastic local search for finding DNA sequences to assign to
:any:`Domain`'s in a :any:`Design` to satisfy all :any:`Constraint`'s.
Various parameters of the search can be controlled using :any:`SearchParameters`.

Instructions for using the dsd library are available at
https://github.com/UC-Davis-molecular-computing/dsd#data-model
"""

from __future__ import annotations

import json
import math
import itertools
import os
import shutil
import sys
import logging
from collections import defaultdict, deque
import collections.abc as abc
from dataclasses import dataclass, field
from typing import List, Tuple, Sequence, FrozenSet, Optional, Dict, Callable, Iterable, \
    Deque, TypeVar, Union, Generic, Iterator, Any
import statistics
import textwrap
import re
import datetime

from tabulate import tabulate
import numpy.random
from ordered_set import OrderedSet
import numpy as np  # noqa

import nuad.np as nnp

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

from nuad.constraints import Domain, Strand, Design, Constraint, DomainConstraint, StrandConstraint, \
    DomainPairConstraint, StrandPairConstraint, ConstraintWithDomainPairs, ConstraintWithStrandPairs, \
    logger, all_pairs, ConstraintWithDomains, ConstraintWithStrands, \
    ComplexConstraint, ConstraintWithComplexes, Complex, DomainsConstraint, StrandsConstraint, \
    DomainPairsConstraint, StrandPairsConstraint, ComplexesConstraint, DesignPart, DesignConstraint, \
    DomainPair, StrandPair, SingularConstraint, BulkConstraint
import nuad.constraints as nc

from nuad.stopwatch import Stopwatch


def new_process_pool(cpu_count: int) -> pathos.multiprocessing.Pool:
    return pathos.multiprocessing.Pool(processes=cpu_count)


_process_pool: pathos.multiprocessing.Pool = new_process_pool(nc.cpu_count())

log_names_of_domains_and_strands_checked = False
pprint_indent = 4


def default_output_directory() -> str:
    return os.path.join('output', f'{script_name_no_ext()}--{timestamp()}')


_parts_to_check_cache = {}


def find_parts_to_check(constraint: nc.Constraint, design: nc.Design,
                        domains_changed: Optional[Iterable[Domain]]) -> Sequence[nc.DesignPart]:
    cache_key = (constraint, id(design))
    if domains_changed is None and cache_key in _parts_to_check_cache:
        return _parts_to_check_cache[cache_key]

    if domains_changed is not None:
        domains_changed_full: OrderedSet[Domain] = OrderedSet()
        for domain in domains_changed:
            domains_in_tree = domain.all_domains_in_tree()
            domains_changed_full.update(domains_in_tree)
        domains_changed = list(domains_changed_full)
        # = OrderedSet(domain.all_domains_in_tree() for domain in )

    parts_to_check: Sequence[nc.DesignPart]
    if isinstance(constraint, ConstraintWithDomains):
        parts_to_check = _determine_domains_to_check(design.domains, domains_changed, constraint)
    elif isinstance(constraint, ConstraintWithStrands):
        parts_to_check = _determine_strands_to_check(design.strands, domains_changed, constraint)
    elif isinstance(constraint, ConstraintWithDomainPairs):
        parts_to_check = _determine_domain_pairs_to_check(design.domains, domains_changed, constraint)
    elif isinstance(constraint, ConstraintWithStrandPairs):
        parts_to_check = _determine_strand_pairs_to_check(design.strands, domains_changed, constraint)
    elif isinstance(constraint, ConstraintWithComplexes):
        parts_to_check = _determine_complexes_to_check(domains_changed, constraint)
    elif isinstance(constraint, nc.DesignConstraint):
        parts_to_check = []  # not used when checking DesignConstraint
    else:
        raise NotImplementedError()

    if domains_changed is None:
        _parts_to_check_cache[cache_key] = parts_to_check

    return parts_to_check


# XXX: important that this is absolute constant. Sometimes this is called for the total weight of all
# violations, and sometimes just for the difference between old and new (the latter are smaller).
# If using relative epsilon, then those can disagree and trigger the assert statement that
# checks that _violations_of_constraints quit_early agrees with the subroutines it calls.
_epsilon = 0.000001


def _is_significantly_greater(x: float, y: float) -> bool:
    return x > y + _epsilon


def _is_significantly_less(x: float, y: float) -> bool:
    return x < y - _epsilon


def _is_significantly_different(x: float, y: float) -> bool:
    return abs(x - y) > _epsilon


def _at_least_one_domain_unfixed(pair: Tuple[Domain, Domain]) -> bool:
    return not (pair[0].fixed and pair[1].fixed)


def _determine_domains_to_check(all_domains: Iterable[Domain],
                                domains_changed: Optional[Iterable[Domain]],
                                constraint: ConstraintWithDomains) -> Sequence[Domain]:
    """
    Determines domains to check in `all_domains`.
    If `domains_new` is None, then this is all that are not fixed if constraint.domains
    is None, otherwise it is constraint.domains.
    If `domains_new` is not None, then among those domains specified above,
    it is just those in `domains_new` that appear in `all_domains`.
    """
    # either all pairs, or just constraint.pairs if specified
    domains_to_check_if_domain_changed_none = all_domains \
        if constraint.domains is None else constraint.domains

    # filter out those not containing domain_change if specified
    domains_to_check = list(domains_to_check_if_domain_changed_none) if domains_changed is None \
        else [domain for domain in domains_to_check_if_domain_changed_none
              if domain in domains_changed]

    return domains_to_check


def _determine_strands_to_check(all_strands: Iterable[Strand],
                                domains_changed: Optional[Iterable[Domain]],
                                constraint: ConstraintWithStrands) -> Sequence[Strand]:
    """
    Similar to _determine_domains_to_check but for strands.
    """
    # either all pairs, or just constraint.pairs if specified
    strands_to_check_if_domain_changed_none = all_strands \
        if constraint.strands is None else constraint.strands

    # filter out those not containing domain_change if specified
    strands_to_check: List[Strand] = []
    if domains_changed is None:
        strands_to_check = list(strands_to_check_if_domain_changed_none)
    else:
        for strand in strands_to_check_if_domain_changed_none:
            for domain_changed in domains_changed:
                if domain_changed in strand.domains:
                    strands_to_check.append(strand)
                    break

    return strands_to_check


def _determine_domain_pairs_to_check(all_domains: Iterable[Domain],
                                     domains_changed: Optional[Iterable[Domain]],
                                     constraint: ConstraintWithDomainPairs) -> Sequence[DomainPair]:
    """
    Determines domain pairs to check between domains in `all_domains`.
    If `domain_changed` is None, then this is all pairs where they are not both fixed if constraint.pairs
    is None, otherwise it is constraint.pairs.
    If `domain_changed` is not None, then among those pairs specified above,
    it is all pairs where one of the two is `domain_changed`.
    """
    # some code is repeated here, but otherwise it's way too slow on a large design to iterate over
    # all pairs of domains only to filter out most of them that don't intersect domains_new
    if domains_changed is None:
        # either all pairs, or just constraint.pairs if specified
        if constraint.pairs is not None:
            domain_pairs_to_check: List[DomainPair] = \
                [DomainPair(domain1, domain2) for domain1, domain2 in constraint.pairs]
        else:
            # check all pairs of domains unless one is an ancestor of another in a subdomain tree
            def not_subdomain(dom1: Domain, dom2: Domain) -> bool:
                return not dom1.contains_in_subtree(dom2) and not dom2.contains_in_subtree(dom1)

            pairs = all_pairs(all_domains, with_replacement=constraint.check_domain_against_itself,
                              where=not_subdomain)
            domain_pairs_to_check = [DomainPair(domain1, domain2) for domain1, domain2 in pairs
                                     if not (domain1.fixed and domain2.fixed)]

    else:
        # either all pairs, or just constraint.pairs if specified
        if constraint.pairs is not None:
            domain_pairs_to_check: List[DomainPair] = \
                [DomainPair(domain1, domain2) for domain1, domain2 in constraint.pairs
                 if domain1 in domains_changed or domain2 in domains_changed]
        else:
            # check all pairs of domains unless one is an ancestor of another in a subdomain tree
            def not_subdomain(dom1: Domain, dom2: Domain) -> bool:
                return not dom1.contains_in_subtree(dom2) and not dom2.contains_in_subtree(dom1)

            domain_pairs_to_check = []
            for domain_changed in domains_changed:
                for other_domain in all_domains:
                    if domain_changed is not other_domain or constraint.check_domain_against_itself:
                        if not_subdomain(domain_changed, other_domain):
                            domain_pairs_to_check.append(DomainPair(domain_changed, other_domain))

    return domain_pairs_to_check


def _at_least_one_strand_unfixed(pair: Tuple[Strand, Strand]) -> bool:
    return not (pair[0].fixed and pair[1].fixed)


def _determine_strand_pairs_to_check(all_strands: Iterable[Strand],
                                     domains_changed: Optional[Iterable[Domain]],
                                     constraint: ConstraintWithStrandPairs) -> Sequence[StrandPair]:
    """
    Similar to _determine_domain_pairs_to_check but for strands.
    """
    # some code is repeated here, but otherwise it's way too slow on a large design to iterate over
    # all pairs of strands only to filter out most of them that don't intersect domains_new
    if domains_changed is None:
        # either all pairs, or just constraint.pairs if specified
        if constraint.pairs is not None:
            strand_pairs_to_check: List[StrandPair] = \
                [StrandPair(strand1, strand2) for strand1, strand2 in constraint.pairs]
        else:
            pairs = all_pairs(all_strands, with_replacement=constraint.check_strand_against_itself)
            strand_pairs_to_check = [StrandPair(strand1, strand2) for strand1, strand2 in pairs]
    else:
        strand_pairs_to_check = []
        if constraint.pairs is not None:
            for strand1, strand2 in constraint.pairs:
                for domain_changed in domains_changed:
                    if domain_changed in strand1.domains or domain_changed in strand2.domains:
                        strand_pairs_to_check.append(StrandPair(strand1, strand2))
                        break  # ensure we don't add the same strand pair twice
        else:
            for domain_changed in domains_changed:
                strands_with_domain_changed = [strand for strand in all_strands
                                               if domain_changed in strand.domains]
                for strand_with_domain_changed in strands_with_domain_changed:
                    for other_strand in all_strands:
                        if (strand_with_domain_changed is not other_strand or
                                constraint.check_strand_against_itself):
                            strand_pairs_to_check.append(StrandPair(strand_with_domain_changed, other_strand))

    return strand_pairs_to_check


def _determine_complexes_to_check(domains_changed: Optional[Iterable[Domain]],
                                  constraint: ConstraintWithComplexes) -> Tuple[Complex]:
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


_empty_frozen_set: FrozenSet = frozenset()


def _independent_domains_in_part(part: nc.DesignPart, exclude_fixed: bool) -> List[Domain]:
    """
    :param part:
        DesignPart (e.g., :any:`Strand`, :any:`Domani`, Tuple[:any:`Strand`, :any:`Strand`])
    :param exclude_fixed:
        whether to exclude :any:`Domain`'s with :data:`Domain.fixed` == True
    :return:
        independent, non-fixed (if exclude_fixed is True) domains associated with part
        (e.g., all domains in :any:`Strand`), with dependent domains substituted with their
        independent source via Domain.independent_source()
    """
    # first compute "direct" domains that appear directly on strands
    domains: List[Domain]
    if isinstance(part, Domain):
        domains = [part] if not (exclude_fixed and part.fixed) else []
    elif isinstance(part, Strand):
        domains = part.domains if not exclude_fixed else part.unfixed_domains()
    elif isinstance(part, DomainPair):
        domains = list(domain for domain in part.individual_parts() if not (exclude_fixed and domain.fixed))
    elif isinstance(part, (StrandPair, Complex)):
        domains_per_strand = [strand.domains if not exclude_fixed else strand.unfixed_domains()
                              for strand in part.individual_parts()]
        domain_iterable: Iterable[Domain] = _flatten(domains_per_strand)
        domains = list(domain_iterable)
    else:
        raise AssertionError(f'part {part} not recognized as one of Domain, Strand, '
                             f'DomainPair, StrandPair, or Complex; it is type {part.__class__.__name__}')

    # Convert direct domains to independent domains.
    # If multiple dependent domains map to the same indepedent domain d_i, only add d_i once
    independent_domains: List[Domain] = []
    for domain in domains:
        independent_domain = domain.independent_source()
        if independent_domain not in independent_domains:
            independent_domains.append(independent_domain)

    return independent_domains


T = TypeVar('T')


def remove_none_from_list(lst: Iterable[Optional[T]]) -> List[T]:
    return [elt for elt in lst if elt is not None]


def _sequences_fragile_format_output_to_file(design: Design,
                                             include_group: bool = True) -> str:
    return '\n'.join(
        f'{strand.name}  '
        f'{strand.group if include_group else ""}  '
        f'{strand.sequence(delimiter="-")}' for strand in design.strands)


def _write_intermediate_files(*, design: nc.Design, params: SearchParameters, rng: numpy.random.Generator,
                              num_new_optimal: int, directories: _Directories,
                              eval_set: EvaluationSet) -> None:
    num_new_optimal_padded = f'{num_new_optimal}' if params.num_digits_update is None \
        else f'{num_new_optimal:0{params.num_digits_update}d}'

    _write_design(design, params=params, directories=directories,
                  num_new_optimal_padded=num_new_optimal_padded)

    _write_rng_state(rng, params=params, directories=directories,
                     num_new_optimal_padded=num_new_optimal_padded)

    _write_sequences(design, params=params, directories=directories,
                     num_new_optimal_padded=num_new_optimal_padded)

    _write_report(params=params, directories=directories,
                  num_new_optimal_padded=num_new_optimal_padded, eval_set=eval_set)


def _write_design(design: Design, params: SearchParameters, directories: _Directories,
                  num_new_optimal_padded: str) -> None:
    content = design.to_json()

    best_filename = directories.best_design_full_filename_noext()
    idx_filename = directories.indexed_design_full_filename_noext(num_new_optimal_padded) \
        if params.save_design_for_all_updates else None
    _write_text_intermediate_and_final_files(content, best_filename, idx_filename)


def _write_rng_state(rng: numpy.random.Generator, params: SearchParameters, directories: _Directories,
                     num_new_optimal_padded: str) -> None:
    state = rng.bit_generator.state
    content = json.dumps(state, indent=2)

    best_filename = directories.best_rng_full_filename_noext()
    idx_filename = directories.indexed_rng_full_filename_noext(num_new_optimal_padded) \
        if params.save_design_for_all_updates else None
    _write_text_intermediate_and_final_files(content, best_filename, idx_filename)


def _write_sequences(design: Design, params: SearchParameters, directories: _Directories,
                     num_new_optimal_padded: str, include_group: bool = True) -> None:
    content = _sequences_fragile_format_output_to_file(design, include_group)

    best_filename = directories.best_sequences_full_filename_noext()
    idx_filename = directories.indexed_sequences_full_filename_noext(num_new_optimal_padded) \
        if params.save_sequences_for_all_updates else None
    _write_text_intermediate_and_final_files(content, best_filename, idx_filename)


def _write_report(params: SearchParameters, directories: _Directories,
                  num_new_optimal_padded: str, eval_set: EvaluationSet) -> None:
    content = f'''\
Report on constraints
=====================
''' + summary_of_constraints(params.constraints, params.report_only_violations,
                             eval_set=eval_set)

    best_filename = directories.best_report_full_filename_noext()
    idx_filename = directories.indexed_report_full_filename_noext(num_new_optimal_padded) \
        if params.save_report_for_all_updates else None
    _write_text_intermediate_and_final_files(content, best_filename, idx_filename)


def _write_text_intermediate_and_final_files(content: str, best_filename: str,
                                             idx_filename: Optional[str]) -> None:
    with open(best_filename, 'w') as file:
        file.write(content)
    if idx_filename is not None:
        with open(idx_filename, 'w') as file:
            file.write(content)


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
    design: str = field(init=False)
    rng_state: str = field(init=False)
    report: str = field(init=False)
    sequence: str = field(init=False)

    # relative to out directory
    design_subdirectory: str = field(init=False, default='designs')
    rng_state_subdirectory: str = field(init=False, default='rng')
    report_subdirectory: str = field(init=False, default='reports')
    sequence_subdirectory: str = field(init=False, default='sequences')

    # names of files to write (in subdirectories, and also "current-best" versions in out
    design_filename_no_ext: str = field(init=False, default='design')
    rng_state_filename_no_ext: str = field(init=False, default='rng')
    sequences_filename_no_ext: str = field(init=False, default='sequences')
    report_filename_no_ext: str = field(init=False, default='report')

    debug_file_handler: Optional[logging.FileHandler] = field(init=False, default=None)
    info_file_handler: Optional[logging.FileHandler] = field(init=False, default=None)

    def all_subdirectories(self, params: SearchParameters) -> List[str]:
        result = []
        if params.save_design_for_all_updates:
            result.extend([self.design, self.rng_state])
        if params.save_sequences_for_all_updates:
            result.append(self.sequence)
        if params.save_report_for_all_updates:
            result.append(self.report)
        return result

    def __init__(self, out: str, debug: bool, info: bool) -> None:
        self.out = out
        self.design = os.path.join(self.out, self.design_subdirectory)
        self.rng_state = os.path.join(self.out, self.rng_state_subdirectory)
        self.report = os.path.join(self.out, self.report_subdirectory)
        self.sequence = os.path.join(self.out, self.sequence_subdirectory)

        if debug:
            self.debug_file_handler = logging.FileHandler(os.path.join(self.out, 'log_debug.log'))
            self.debug_file_handler.setLevel(logging.DEBUG)
            nc.logger.addHandler(self.debug_file_handler)

        if info:
            self.info_file_handler = logging.FileHandler(os.path.join(self.out, 'log_info.log'))
            self.info_file_handler.setLevel(logging.INFO)
            nc.logger.addHandler(self.info_file_handler)

    @staticmethod
    def indexed_full_filename_noext(filename_no_ext: str, directory: str, idx: Union[int, str],
                                    ext: str) -> str:
        relative_filename = f'{filename_no_ext}-{idx}.{ext}'
        full_filename = os.path.join(directory, relative_filename)
        return full_filename

    def best_full_filename_noext(self, filename_no_ext: str, ext: str) -> str:
        relative_filename = f'{filename_no_ext}_best.{ext}'
        full_filename = os.path.join(self.out, relative_filename)
        return full_filename

    def indexed_design_full_filename_noext(self, idx: Union[int, str]) -> str:
        return self.indexed_full_filename_noext(self.design_filename_no_ext, self.design, idx, 'json')

    def indexed_rng_full_filename_noext(self, idx: Union[int, str]) -> str:
        return self.indexed_full_filename_noext(self.rng_state_filename_no_ext, self.rng_state, idx, 'json')

    def indexed_sequences_full_filename_noext(self, idx: Union[int, str]) -> str:
        return self.indexed_full_filename_noext(self.sequences_filename_no_ext, self.sequence, idx, 'txt')

    def indexed_report_full_filename_noext(self, idx: Union[int, str]) -> str:
        return self.indexed_full_filename_noext(self.report_filename_no_ext, self.report, idx, 'txt')

    def best_design_full_filename_noext(self) -> str:
        return self.best_full_filename_noext(self.design_filename_no_ext, 'json')

    def best_rng_full_filename_noext(self) -> str:
        return self.best_full_filename_noext(self.rng_state_filename_no_ext, 'json')

    def best_sequences_full_filename_noext(self) -> str:
        return self.best_full_filename_noext(self.sequences_filename_no_ext, 'txt')

    def best_report_full_filename_noext(self) -> str:
        return self.best_full_filename_noext(self.report_filename_no_ext, 'txt')


def _check_design(design: nc.Design) -> None:
    # verify design is legal in senses not already checked

    for strand in design.strands:
        for domain in strand.domains:
            # noinspection PyProtectedMember
            if domain._pool is None and not (domain.fixed or domain.dependent):
                raise ValueError(f'for strand {strand.name}, it has a '
                                 f'non-fixed, non-dependent domain {domain.name} '
                                 f'with pool set to None.\n'
                                 f'For domains that are not fixed and not dependent, '
                                 f'exactly one of these must be None.')
            # noinspection PyProtectedMember
            elif domain._pool is not None and domain.fixed:
                raise ValueError(f'for strand {strand.name}, it has a '
                                 f'domain {domain.name} that is fixed, even though that Domain has a '
                                 f'DomainPool.\nA Domain cannot be fixed and have a DomainPool.')
            elif domain._pool is not None and domain.dependent:
                raise ValueError(f'for strand {strand.name}, it has a '
                                 f'domain {domain.name} that is dependent, even though that Domain has a '
                                 f'DomainPool.\nA Domain cannot be dependent and have a DomainPool.')


@dataclass
class SearchParameters:
    """
    This class describes various parameters to give to the search algorithm
    :meth:`search_for_dna_sequences`.
    """

    constraints: List[Constraint] = field(default_factory=list)
    """
    List of :any:`constraints.Constraint`'s to apply to the :any:`Design`.
    """

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
    If True, the text written to the screen through logger.info (on the logger instance used in
    dsd.constraints) is written to the file log_info.log in the directory `out_directory`.
    """

    report_only_violations: bool = True
    """
    NOTE: due to a recent change in how the search is conducted, setting this parameter to False is not
    currently supported. 
    
    If True, does not give report on each constraint that was satisfied; only reports violations
    and summary of all constraint checks of a certain type (e.g., how many constraint checks there were).
    """

    max_iterations: Optional[int] = None
    """
    Maximum number of iterations of search to perform.
    """

    target_score: Optional[float] = None
    """
    Total violation score to attempt to obtain. A score of 0.0 represents that all constraints 
    are satisfied. Often a search can get very close to score 0.0 quickly, but take a very long time
    to reach a score of 0.0. Set this to a small positive value to allow the search to quit before
    all constraints are satisfied, but they are "mostly" satisfied. 
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
    If num_digits_update=3 is specified, for instance, they will be written
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

    save_report_for_all_updates: bool = False
    """
    A report on the most recently updated :any:`Design` is always written to a file 
    `current-best-report.txt`. If this is True, then in the folder `reports`, a file unique to that update
    is also written. Set to False to use less space on disk. 
    """

    save_design_for_all_updates: bool = False
    """
    A serialized (JSON) description of the most recently updated :any:`Design` is always written to 
    a file `current-best-design.json`. If this is True, then in the folder `dsd_designs`, a file unique to 
    that update is also written. Set to False to use less space on disk. 
    """

    save_sequences_for_all_updates: bool = False
    """
    A list of sequences for each :any:`Strand` of most recently updated :any:`Design` is always written to 
    a file `current-best-sequences.txt`. If this is True, then in the folder `sequences`, a file unique to 
    that update is also written. Set to False to use less space on disk. 
    """

    log_time: bool = False
    """
    Whether to log the time taken per iteration to the screen.
    """

    scrolling_output: bool = True
    r"""
    If True, then screen output "scrolls" on the screen, i.e., a newline is printed after each iteration,
    e.g.,
    
    .. code-block:: console
    
        $ python sst_canvas.py
        using random seed of 1; use this same seed to reproduce this run
        number of processes in system: 4
        |-----------|--------|-----------|-----------|----------|---------------|
        | iteration | update | opt score | new score | StrandSS | StrandPairRNA |
        |         0 |      0 |    2555.9 |    2545.9 |    118.2 |        2437.8 |
        |-----------|--------|-----------|-----------|----------|---------------|
        | iteration | update | opt score | new score | StrandSS | StrandPairRNA |
        |         1 |      1 |    2545.9 |    2593.0 |    120.2 |        2425.6 |
        |-----------|--------|-----------|-----------|----------|---------------|
        | iteration | update | opt score | new score | StrandSS | StrandPairRNA |
        |         2 |      1 |    2545.9 |    2563.1 |    120.2 |        2425.6 |
        |-----------|--------|-----------|-----------|----------|---------------|
        | iteration | update | opt score | new score | StrandSS | StrandPairRNA |
        |         3 |      1 |    2545.9 |    2545.0 |    120.2 |        2425.6 |
        |-----------|--------|-----------|-----------|----------|---------------|
        | iteration | update | opt score | new score | StrandSS | StrandPairRNA |
        |         4 |      2 |    2545.0 |    2510.1 |    121.0 |        2423.9 |
    
    If False, then the screen output is updated in place:
    
    .. code-block:: console
        
        $ python sst_canvas.py
        using random seed of 1; use this same seed to reproduce this run
        number of processes in system: 4
        |-----------|--------|-----------|-----------|----------|---------------|
        | iteration | update | opt score | new score | StrandSS | StrandPairRNA |
        |        27 |     14 |    2340.5 |    2320.5 |    109.6 |        2230.9 |

    
    This is done by printing the symbol '\r' (carriage return), which sets the print position
    back to the start of the line. The terminal screen must be wide enough to handle the output or
    this won't work. 
    
    The search also occassionally logs other things to the screen that may disrupt this a bit.
    """

    def __post_init__(self):
        self._check_constraint_types()

    def _check_constraint_types(self) -> None:
        idx = 0
        for constraint in self.constraints:
            if not isinstance(constraint, Constraint):
                raise ValueError('each element of constraints must be an instance of Constraint, '
                                 f'but the element at index {idx} is of type {type(constraint)}')
            idx += 1


def search_for_dna_sequences(design: nc.Design, params: SearchParameters) -> None:
    """
    Search for DNA sequences to assign to each :any:`Domain` in `design`, satisfying the various
    :any:`Constraint`'s in :data:`SearchParameters.constraints`.

    **Search algorithm:**
    This is a stochastic local search. It determines which :any:`Constraint`'s are violated.
    More precisely, it adds the total score of all violated constraints
    (sum of :data:`constraints.Constraint.weight` * score_of_violation over all violated
    :any:`Constraint`'s).
    The goal is to reduce this total score until it is 0 (i.e., no violated constraints).
    Any :any:`Domain` "involved" in the violated :any:`Constraint` is noted as being one of the
    :any:`Domain`'s responsible for the violation, i.e., is "blamed".
    For example, if a :any:`DomainConstraint` is violated,
    only one :any:`Domain` is blamed, whereas if a :any:`StrandConstraint` is violated, every :any:`Domain`
    in the :any:`Strand` is blamed.
    However, fixed domains (those with :data:`constraints.Domain.fixed` = True) are never blamed,
    since their DNA sequences cannot be changed.

    While any :any:`Constraint`'s are violated, a :any:`Domain` is picked
    at random, with probability proportional to the total score of all the :any:`Constraint`'s
    for which the :any:`Domain` was blamed (so probability 0 to pick a :any:`Domain` that is fixed or that
    was involved in no violations).
    A new DNA sequence is assigned to this
    :any:`Domain` by calling :meth:`constraints.DomainPool.generate_sequence` on the :any:`DomainPool`
    of that :any:`Domain`.

    The way to decide whether to keep the changed sequence, or revert to the
    old sequence, can be configured, but the default is to keep the change if and only if it
    does not increase the total score of violations.
    More generally, we calculate the total score of all violated constraints in the original and changed
    :any:`Design`, calling their difference `score_delta` = `new_total_score` - `old_total_score`.
    The value ``probability_of_keeping_change(score_delta)`` is the probability that the change
    is kept. The default function computing this probability is returned by
    :meth:`default_probability_of_keeping_change_function`, which simply assigns probability 0
    to keep the change if `score_delta` is positive (i.e., the score went up) and probability 1
    otherwise.
    In particular, the change is kept if the score is identical (though this would happen only rarely).
    One reason to favor this default is that it allows an optimization that speeds up the search
    significantly in practice: When evaluating constraints, once the total score of violations exceeds
    that of the best design so far, no further constraints need to be evaluated, since we can decide
    immediately that the new design change will not be kept.

    The :any:`Design` is modified in place; each :any:`Domain` is modified to have a DNA sequence.

    If no DNA sequences are assigned to the :any:`Domain`'s initially, they are picked at random
    from the :any:`DomainPool` associated to each :any:`Domain` by calling
    :py:meth:`constraints.DomainPool.generate_sequence`.

    Otherwise, if DNA sequences are already assigned to the :any:`Domain`'s initially, these sequences
    are used as a starting point for finding sequences that satisfy all :any:`Constraint`'s.
    (In this case, those sequences are not checked against any :any:`NumpyConstraint`'s
    or :any:`SequenceConstraint`'s in the :any:`Design`, since those checks are applied prior to
    assigning DNA sequences to any :any:`Domain`.)

    The function has some side effects. It writes a report on the optimal sequence assignment found so far
    every time a new improve assignment is found.

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

    if params.random_seed is not None:
        logger.info(f'using random seed of {params.random_seed}; '
                    f'use this same seed to reproduce this run')

    # keys should be the non-independent Domains in this Design, mapping to the unique Strand with a
    # StrandPool that contains them.
    # domain_to_strand: Dict[dc.Domain, dc.Strand] = _check_design(design)
    design.compute_derived_fields()

    design.check_all_subdomain_graphs_acyclic()
    design.check_all_subdomain_graphs_uniquely_assignable()
    design.check_names_unique()
    _check_design(design)

    directories = _setup_directories(params)

    if params.random_seed is not None:
        rng = np.random.default_rng(params.random_seed)
    else:
        rng = nnp.default_rng

    if params.probability_of_keeping_change is None:
        params.probability_of_keeping_change = default_probability_of_keeping_change_function(params)
        if params.never_increase_score is None:
            params.never_increase_score = True
    elif params.never_increase_score is None:
        params.never_increase_score = False

    assert params.never_increase_score is not None

    cpu_count = nc.cpu_count()
    logger.info(f'number of processes in system: {cpu_count}')

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
            num_new_optimal, rng_restart = _restart_from_directory(directories, design, params)
            if rng_restart is not None:
                rng = rng_restart

        eval_set = EvaluationSet(params.constraints, params.never_increase_score)
        eval_set.evaluate_all(design)

        if not params.restart:
            # write initial sequences and report
            _write_intermediate_files(design=design, params=params, rng=rng, num_new_optimal=num_new_optimal,
                                      directories=directories, eval_set=eval_set)

        iteration = 0
        stopwatch = Stopwatch()

        while not _done(iteration, params, eval_set):
            if params.log_time:
                stopwatch.restart()

            _check_cpu_count(cpu_count)

            domains_new, original_sequences = _reassign_domains(eval_set, params.max_domains_to_change, rng)

            # evaluate constraints on new Design with domain_to_change's new sequence
            eval_set.evaluate_new(design, domains_new=domains_new)

            # _double_check_violations_from_scratch(design=design, params=params, iteration=iteration,
            #                                       eval_set=eval_set)

            _log_constraint_summary(params=params, eval_set=eval_set,
                                    iteration=iteration, num_new_optimal=num_new_optimal)

            # based on total score of new constraint violations compared to optimal assignment so far,
            # decide whether to keep the change
            score_delta = -eval_set.calculate_score_gap()
            prob_keep_change = params.probability_of_keeping_change(score_delta)
            keep_change = rng.random() < prob_keep_change if prob_keep_change < 1 else True

            if not keep_change:
                _unassign_domains(domains_new, original_sequences)
                eval_set.reset_new()
            else:
                # keep new sequence and update information about optimal solution so far
                eval_set.replace_with_new()
                if score_delta < 0:  # increment whenever we actually improve the design
                    num_new_optimal += 1
                    on_improved_design(num_new_optimal)  # type: ignore
                    _write_intermediate_files(design=design, params=params, rng=rng,
                                              num_new_optimal=num_new_optimal, directories=directories,
                                              eval_set=eval_set)

            iteration += 1
            if params.log_time:
                stopwatch.stop()
                _log_time(stopwatch)

        _log_constraint_summary(params=params, eval_set=eval_set,
                                iteration=iteration, num_new_optimal=num_new_optimal)

    finally:
        # if sys.platform != 'win32':
        #     _pfunc_killall()
        _process_pool.close()  # noqa
        _process_pool.terminate()

    if directories.debug_file_handler is not None:
        nc.logger.removeHandler(directories.debug_file_handler)  # noqa

    if directories.info_file_handler is not None:
        nc.logger.removeHandler(directories.info_file_handler)  # noqa


def _done(iteration: int, params: SearchParameters, eval_set: EvaluationSet) -> bool:
    # unconditinoally stop when max_iterations is reached, if specified
    if params.max_iterations is not None and iteration >= params.max_iterations:
        return True

    # otherwise if target_score is specified, check that current score is close to it
    if params.target_score is not None:
        if _is_significantly_greater(eval_set.total_score, params.target_score):
            return False
    else:
        # otherwise just see if any violations remain that are not fixed
        # (i.e., that might be correctable by changing domains; fixed violations are un-solvable)
        if eval_set.has_nonfixed_violations():
            return False

    return True


def create_report(design: nc.Design, constraints: Iterable[Constraint]) -> str:
    """
    Returns string containing report of how well `design` does according to `constraints`, assuming
    `design` has sequences assigned to it, for example, if it was read using
    :meth:`constraints.Design.from_design_file`
    from a design.json file writte as part of a call to :meth:`search_for_dna_sequences`.

    The report is the same format as written to the reports generated when calling
    :meth:`search_for_dna_sequences`.
    Unfortunately this means it suffers the limitation that it currently only prints a summary
    for *violations* of constraints;
    see https://github.com/UC-Davis-molecular-computing/dsd/issues/134

    :param design:
        the :any:`constraints.Design`, with sequences assigned to all :any:`Domain`'s
    :param constraints:
        the list of :any:`constraints.Constraint`'s to evaluate in the report
    :return:
        string describing a report of how well `design` does according to `constraints`
    """
    evaluation_set = EvaluationSet(constraints, False)
    evaluation_set.evaluate_all(design)

    content = f'''\
Report on constraints
=====================
''' + summary_of_constraints(constraints, True, eval_set=evaluation_set)

    return content


def _check_cpu_count(cpu_count: int) -> None:
    # alters number of threads in ThreadPool if cpu count changed. (Lets us hot-swap CPUs, e.g.,
    # in Amazon web services, without stopping the program.)
    if cpu_count != nc.cpu_count():
        logger.info(f'number of processes in system changed from {cpu_count} to {nc.cpu_count()}'
                    f'\nallocating new ThreadPool')
        cpu_count = nc.cpu_count()
        global _process_pool
        _process_pool.close()
        _process_pool.terminate()
        _process_pool = new_process_pool(cpu_count)


def _setup_directories(params: SearchParameters) -> _Directories:
    out_directory = params.out_directory
    if out_directory is None:
        out_directory = default_output_directory()

    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    if not params.restart:
        import time
        time.sleep(0.5)  # for some reason often get file write errors without this pause
        _clear_directory(out_directory, params.force_overwrite)

    directories = _Directories(out=out_directory, debug=params.debug_log_file,
                               info=params.info_log_file)

    for subdir in directories.all_subdirectories(params):
        if not os.path.exists(subdir):
            os.makedirs(subdir)

    return directories


def _reassign_domains(eval_set: EvaluationSet, max_domains_to_change: int,
                      rng: np.random.Generator) -> Tuple[List[Domain], Dict[Domain, str]]:
    # pick domain to change, with probability proportional to total score of constraints it violates
    # first weight scores by domain's weight
    domains = list(eval_set.domain_to_score.keys())
    scores_weighted = [score * domain.weight for domain, score in eval_set.domain_to_score.items()]
    probs_opt = np.asarray(scores_weighted)
    probs_opt /= probs_opt.sum()
    num_domains_to_change = 1 if max_domains_to_change == 1 \
        else rng.choice(a=range(1, max_domains_to_change + 1))
    domains_changed: List[Domain] = list(rng.choice(a=domains, p=probs_opt, replace=False,
                                                    size=num_domains_to_change))

    # fixed Domains should never be blamed for constraint violation
    assert all(not domain_changed.fixed for domain_changed in domains_changed)

    # dependent domains also cannot be blamed, since their independent source should have been blamed
    assert all(not domain_changed.dependent for domain_changed in domains_changed)

    original_sequences: Dict[Domain, str] = {}

    for domain in domains_changed:
        # set sequence of domain_changed to random new sequence from its DomainPool
        assert domain not in original_sequences
        previous_sequence = domain.sequence()
        original_sequences[domain] = previous_sequence
        new_sequence = domain.pool.generate_sequence(rng, previous_sequence)
        domain.set_sequence(new_sequence)

    return domains_changed, original_sequences


def _unassign_domains(domains_changed: Iterable[Domain], original_sequences: Dict[Domain, str]) -> None:
    for domain_changed in domains_changed:
        domain_changed.set_sequence(original_sequences[domain_changed])


# used for debugging; early on, the algorithm for quitting early had a bug and was causing the search
# to think a new assignment was better than the optimal so far, but a mistake in score accounting
# from quitting early meant we had simply stopped looking for violations too soon.
def _double_check_violations_from_scratch(design: nc.Design, params: SearchParameters, iteration: int,
                                          eval_set: EvaluationSet):
    eval_set_from_scratch = EvaluationSet(params.constraints, params.never_increase_score)
    eval_set_from_scratch.evaluate_all(design)
    score_new = eval_set.total_score_new()
    score_opt = eval_set.total_score
    score_fs = eval_set_from_scratch.total_score

    problem = False
    if not params.never_increase_score:
        if _is_significantly_different(score_fs, score_new):
            problem = True
    else:
        # XXX: we shouldn't check that the actual scores are close if quit_early is enabled, because then
        # the total score found on quitting early will be less than the total score if not.
        # But uncomment this, while disabling quitting early, to test more precisely for "wrong total score".
        # import math
        # if not math.isclose(violation_set_new.total_score(), violation_set_new_fs.total_score()):
        # Instead, we check whether the total score lie on different sides of the opt total score, i.e.,
        # they make different decisions about whether to change to the new assignment
        # if ((score_new
        #      <= score_opt
        #      < score_fs) or
        #         (score_fs
        #          <= score_opt
        #          < score_new)):
        if ((_is_significantly_less(score_new, score_opt)
             and _is_significantly_less(score_opt, score_fs)) or
                ((_is_significantly_less(score_fs, score_opt)
                  and _is_significantly_less(score_opt, score_new)))):
            problem = True
    if problem:
        logger.warning(f'''\
WARNING! There is a bug in nuad.
From scratch, we calculated score {score_fs}.
The optimal score so far is       {score_opt}.
Iteratively, we calculated score  {score_new}.
This means the iterative search is saying something different about quitting early than the full search. '
This happened on iteration {iteration}.''')
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


def _restart_from_directory(directories: _Directories, design: nc.Design, params: SearchParameters) \
        -> Tuple[int, np.random.Generator]:
    # NOTE: If the subdirectory design/ exists, then this restarts from highest index found in the
    # subdirectory, NOT from "design_best.json" file, which is ignored in that case.
    # It is only used if the design/ subdirectory is missing.
    # This also dictates whether rng/ subdirectory or rng_best.json is used,
    # so if design/ exists and has a file, e.g., design/design-75.json, then it is assumed that the file
    # rng/rng-75.json also exists.
    # If params.random_seed is defined, then the rng returned is None.

    if os.path.isdir(directories.design):
        # returns highest index found in design subdirectory
        highest_idx = _find_highest_index_in_directory(directories.design,
                                                       directories.design_filename_no_ext, 'json')
        design_filename = directories.indexed_design_full_filename_noext(highest_idx)
        rng_filename = directories.indexed_rng_full_filename_noext(highest_idx)
    else:
        # otherwise we go with contents of "current-best-*.json"
        design_filename = directories.best_design_full_filename_noext()
        rng_filename = directories.best_rng_full_filename_noext()

        # try to find number of updates from other directories
        # so that future written files will have the correct number
        if os.path.isdir(directories.sequence):
            highest_idx = _find_highest_index_in_directory(directories.sequence,
                                                           directories.sequences_filename_no_ext, 'txt')
        elif os.path.isdir(directories.report):
            highest_idx = _find_highest_index_in_directory(directories.report,
                                                           directories.report_filename_no_ext, 'txt')
        else:
            highest_idx = 0

    # read design
    with open(design_filename, 'r') as file:
        design_json_str = file.read()
    design_stored = nc.Design.from_json(design_json_str)
    nc.verify_designs_match(design_stored, design, check_fixed=False)

    rng = None

    if params.random_seed is not None:
        logger.warning(f"""\
When using the restart option, normally I use the stored random seed in
rng_best.json so that the search continues with the same results as if it
had not be stopped. However, you specified a different random seed of
{params.random_seed}, so the results will be different than if the original
run of the search algorithm had been allowed to continue.""")
    else:
        # read RNG state
        with open(rng_filename, 'r') as file:
            rng_state_json = file.read()
        rng_state = json.loads(rng_state_json)
        rng = numpy.random.default_rng()
        rng.bit_generator.state = rng_state
        logger.warning(f"""\
Using stored random seed from file {rng_filename} to produce search results
identical to those that would have happened if the search had not be stopped.""")

    # this is really ugly how we do this, taking parts of the design from `design`,
    # parts from `design_stored`, and parts from the stored DomainPools, but this seems to be necessary
    # to give the user the expected behavior that the Design they passed into search_for_dna_sequences
    # is the Design being modified by the search (not the Design that is read in from the stored .json)
    design.copy_sequences_from(design_stored)

    return highest_idx, (rng if rng is not None else None)


def _find_highest_index_in_directory(directory: str, filename_start: str, ext: str) -> int:
    # return highest index of filename (name matches "<filename_start>-<index>.<ext>"
    # raises ValueError if none exists
    try:
        list_dir = os.listdir(directory)
    except FileNotFoundError:
        list_dir = None
    if list_dir is not None and len(list_dir) > 0:
        filenames = [filename
                     for filename in list_dir
                     if os.path.isfile(os.path.join(directory, filename))]
    else:
        raise ValueError(f'no files in directory "{directory}" '
                         f'match the pattern "{filename_start}-<index>.{ext}";\n')

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


n_in_last_n_calls = 50
time_last_n_calls: Deque = deque(maxlen=n_in_last_n_calls)
time_last_n_calls_available = False


def _log_time(stopwatch: Stopwatch, include_median: bool = False) -> None:
    global time_last_n_calls_available
    if time_last_n_calls_available:
        time_last_n_calls.append(stopwatch.milliseconds())
        ave_time = statistics.mean(time_last_n_calls)
        content = f'| time: {stopwatch.milliseconds_str(1, 6)} ms ' + \
                  f'| last {len(time_last_n_calls)} calls average: {ave_time:.1f} ms |'
        if include_median:
            med_time = statistics.median(time_last_n_calls)
            content += f' median: {med_time:.1f} ms |'
        content_width = len(content)
        logger.info('-' * content_width + '\n' + content)
    else:
        # skip appending first time, since it is much larger and skews the average
        content = f'| time for first call: {stopwatch.milliseconds_str()} ms |'
        logger.info('-' * len(content) + '\n' + content)
        time_last_n_calls_available = True


def _flatten(list_of_lists: Iterable[Iterable[T]]) -> Iterable[T]:
    #  Flatten one level of nesting
    return itertools.chain.from_iterable(list_of_lists)


def _remove_first_lines_from_string(s: str, num_lines: int) -> str:
    return '\n'.join(s.split('\n')[num_lines:])


def _log_constraint_summary(*, params: SearchParameters,
                            eval_set: EvaluationSet,
                            iteration: int,
                            num_new_optimal: int) -> None:
    # If output is not scrolling, only print this once on first iteration.
    if params.scrolling_output or iteration == 0:
        row1 = ['iteration', 'update', 'opt score', 'new score'] + [f'{constraint.short_description}'
                                                                    for constraint in params.constraints]
        header = tabulate([row1], tablefmt='github')
        print(header)

    def _dec(score: float) -> int:
        # how many decimals after decimal point to use given the score
        dec_opt = max(1, math.ceil(math.log(1 / score, 10)) + 2) if score > 0 else 1
        return dec_opt

    score_opt = eval_set.total_score
    score_new = eval_set.total_score_new()

    dec_opt = _dec(score_opt)
    dec_new = _dec(score_new)

    all_constraints_strs = []
    for constraint in params.constraints:
        score = eval_set.score_of_constraint(constraint, True)
        length = len(constraint.short_description)
        num_decimals = _dec(score)
        constraint_str = f'{score:{length}.{num_decimals}f}'
        # round further if this would exceed length
        if len(constraint_str) > length:
            excess = len(constraint_str) > length
            num_decimals -= excess
            if num_decimals < 0:
                num_decimals = 0
            constraint_str = f'{score:{length}.{num_decimals}f}'
        all_constraints_strs.append(constraint_str)
    # all_constraints_str = '|'.join(all_constraints_strs)

    # logger.info(header + '\n' + score_str + all_constraints_str)

    # TODO: use floatfmt per column to adjust decimal places
    row1 = ['iteration', 'update', 'opt score', 'new score'] + [f'{constraint.short_description}'
                                                                for constraint in params.constraints]
    # iteration_str = f'{iteration:9}'
    # num_new_optimal_str = f'{num_new_optimal:6}'
    score_opt_str = f'{score_opt :9.{dec_opt}f}'
    score_new_str = f'{score_new :9.{dec_new}f}'
    row2 = [iteration, num_new_optimal, score_opt_str, score_new_str] + all_constraints_strs  # type:ignore
    table = [row1, row2]
    table_str = tabulate(table, tablefmt='github', numalign='right', stralign='right')
    table_str = _remove_first_lines_from_string(table_str, 2)
    # logger.info(table_str)
    line_end = '\n' if params.scrolling_output else '\r'
    print(table_str, end=line_end)


def assign_sequences_to_domains_randomly_from_pools(design: Design,
                                                    warn_fixed_sequences: bool,
                                                    rng: np.random.Generator = nnp.default_rng,
                                                    overwrite_existing_sequences: bool = False) -> None:
    """
    Assigns to each :any:`Domain` in this :any:`Design` a random DNA sequence from its
    :any:`DomainPool`, calling :py:meth:`constraints.DomainPool.generate_sequence` to get the sequence.

    This is step #1 in the search algorithm.

    :param design:
        Design to which to assign DNA sequences.
    :param warn_fixed_sequences:
        Whether to log warning that each :any:`Domain` with :data:`constraints.Domain.fixed` = True
        is not being assigned.
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
            skip_nonfixed_msg = f'Skipping initial assignment of DNA sequence to domain {domain.name}. ' \
                                f'That domain currently has a non-fixed sequence {domain.sequence()}, ' \
                                f'which the search will attempt to replace.'
            skip_fixed_msg = f'Skipping initial assignment of DNA sequence to domain {domain.name}. ' \
                             f'That domain has a fixed sequence {domain.sequence()}, ' \
                             f'and the search will not replace it.'
        if overwrite_existing_sequences:
            if not domain.fixed:
                at_least_one_domain_unfixed = True
                new_sequence = domain.pool.generate_sequence(rng, domain.sequence())
                domain.set_sequence(new_sequence)
                assert len(domain.sequence()) == domain.pool.length
            else:
                logger.info(skip_nonfixed_msg)
        else:
            if not domain.fixed:
                # even though we don't assign a new sequence here, we want to record that at least one
                # domain is not fixed so that we know it is eligible to be overwritten during the search
                at_least_one_domain_unfixed = True
            if not domain.fixed and not domain.has_sequence():
                new_sequence = domain.pool.generate_sequence(rng)
                domain.set_sequence(new_sequence)
                assert len(domain.sequence()) == domain.get_length()
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


def default_probability_of_keeping_change_function(params: SearchParameters) -> Callable[[float], float]:
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

    :param params: :any:`SearchParameters` to apply this rule for; `params` is required because the score of
                   :any:`Constraint`'s in the :any:`SearchParameters` are used to calculate an appropriate
                   epsilon value for determining when a score change is too small to be significant
                   (i.e., is due to rounding error)
    :return: the "keep change" function `f`: :math:`\\mathbb{R} \\to [0,1]`,
             where :math:`f(w_\\delta) = 1` if :math:`w_\\delta \\leq \\epsilon`
             (where :math:`\\epsilon` is chosen to be 1,000,000 times smaller than
             the smallest :any:`Constraint.weight` for any :any:`Constraint` in `design`),
             and :math:`f(w_\\delta) = 0` otherwise.
    """
    min_weight = min((constraint.weight for constraint in params.constraints), default=0.0)
    epsilon_from_min_weight = min_weight / 1000000.0

    def keep_change_only_if_no_worse(score_delta: float) -> float:
        return 1.0 if score_delta <= epsilon_from_min_weight else 0.0

    # def keep_change_only_if_better(score_delta: float) -> float:
    #     return 1.0 if score_delta <= -epsilon_from_min_weight else 0.0

    return keep_change_only_if_no_worse
    # return keep_change_only_if_better


K1 = TypeVar('K1')
K2 = TypeVar('K2')
V = TypeVar('V')


# convenience methods for iterating over 2D dicts

def keys_2d(dct: Dict[K1, Dict[K2, V]]) -> Iterator[Tuple[K1, K2]]:
    for first_key in dct.keys():
        for second_key in dct[first_key].keys():
            yield first_key, second_key


def values_2d(dct: Dict[K1, Dict[K2, V]]) -> Iterator[V]:
    for first_key in dct.keys():
        for value in dct[first_key].values():
            yield value


def items_2d(dct: Dict[K1, Dict[K2, V]]) -> Iterator[Tuple[Tuple[K1, K2], V]]:
    for first_key in dct.keys():
        for second_key, value in dct[first_key].items():
            yield (first_key, second_key), value


def key_exists_in_2d_dict(dct: Dict[K1, Dict[K2, Any]], key1: K1, key2: K2) -> bool:
    # avoid using the brakcet [] operator (to avoid triggering the default dict behavior
    if key1 not in dct.keys():
        return False
    return key2 in dct[key1][key2]


@dataclass
class EvaluationSet:
    # Represents evaluations of :any:`Constraint`'s in a :any:`Design`.
    #
    # It is designed to be efficiently updateable when a single :any:`Domain` changes,
    # to efficiently update only those evaluations of :any:`Constraint`'s that could have been
    # affected by the changed :any:`Domain`.

    constraints: List[Constraint]
    # list of all constraints

    evaluations: Dict[Constraint, Dict[nc.Part, Evaluation]]
    # "2D dict" mapping each (Constraint, Part) to the list of all Evaluations of it.
    # has keys for every (Constraint, Part) instance.

    evaluations_new: Dict[Constraint, Dict[nc.Part, Evaluation]]
    # "2D dict" mapping some Constraint, Part to the list of all new Evaluation's of it
    # (after changing domain(s)).
    # Unlike evaluations, only has keys for parts affected by the most recent domain changes.

    domain_to_evaluations: Dict[Domain, List[Evaluation]]
    # Dict mapping each Domain to the set of all Evaluations for which it is blamed

    violations: Dict[Constraint, Dict[nc.Part, Evaluation]]
    # "2D dict" mapping each (Constraint, Part) to the list of all violations of it.
    # (evaluation that had a positive score)
    # has keys for every Constraint
    # only has keys for Parts that are violations

    violations_new: Dict[Constraint, Dict[nc.Part, Evaluation]]
    # "2D dict" mapping some Constraint, Part to the list of all new Evaluations of it
    # (after changing domain(s)).
    # Unlike violations, only has keys for parts affected by the most recent domain changes.

    domain_to_evaluations: Dict[Domain, List[Evaluation]]
    # Dict mapping each :any:`constraint.Domain` to the set of all :any:`Evaluation`'s for which it is blamed

    domain_to_evaluations_new: Dict[Domain, List[Evaluation]]

    domain_to_violations: Dict[Domain, List[Evaluation]]
    # Dict mapping each :any:`constraint.Domain` to the set of all :any:`Evaluation`'s for which it is blamed

    domain_to_violations_new: Dict[Domain, List[Evaluation]]

    domain_to_score: Dict[Domain, float]

    domain_to_score_new: Dict[Domain, float]

    total_score: float
    # sum of scores of all evalutions

    total_score_nonfixed: float
    # for evaluations blaming Domains with Domain.fixed = False

    total_score_fixed: float
    # for evaluations blaming Domains with Domain.fixed = True

    num_evaluations: int
    num_evaluations_fixed: int
    num_evaluations_nonfixed: int

    num_violations: int
    num_violations_fixed: int
    num_violations_nonfixed: int

    def __init__(self, constraints: Iterable[Constraint], never_increase_score: bool) -> None:
        self.constraints = list(constraints)
        self.never_increase_score = never_increase_score
        self.reset_all()

    def __repr__(self):
        all_evals: List[Evaluation] = [evaluation
                                       for part_to_eval in self.evaluations.values()
                                       for evaluation in part_to_eval.values()]
        lines = "\n  ".join(map(str, all_evals))
        return f'EvaluationSet(\n  {lines})'

    def __str__(self):
        return repr(self)

    def reset_all(self) -> None:
        self.evaluations = {constraint: {} for constraint in self.constraints}
        self.violations = {constraint: {} for constraint in self.constraints}
        self.domain_to_evaluations = defaultdict(list)
        self.domain_to_violations = defaultdict(list)
        self.domain_to_score = defaultdict(float)
        self.reset_new()

    def reset_new(self) -> None:
        self.evaluations_new = {constraint: {} for constraint in self.constraints}
        self.violations_new = {constraint: {} for constraint in self.constraints}
        self.domain_to_evaluations_new = defaultdict(list)
        self.domain_to_violations_new = defaultdict(list)
        self.domain_to_score_new = defaultdict(float)

    def evaluate_all(self, design: Design) -> None:
        # called on all parts of the design and sets self.evaluations
        self.reset_all()
        for constraint in self.constraints:
            self.evaluate_constraint(constraint, design, None, None)
        self.domain_to_score = EvaluationSet.sum_domain_scores(self.domain_to_violations)
        self.update_scores_and_counts()
        # _assert_violations_are_accurate(self.evaluations, self.violations)

    def evaluate_new(self, design: Design, domains_new: List[Domain]) -> None:
        # called only on changed parts of the design and sets self.evaluations_new
        # does quit early optimization since this is only called when comparing to an existing set of evals
        self.reset_new()
        score_gap = None
        if self.never_increase_score:
            score_gap = self.calculate_initial_score_gap(design, domains_new)
        for constraint in self.constraints:
            score_gap = self.evaluate_constraint(constraint, design, score_gap, domains_new)
            if score_gap is not None and _is_significantly_greater(0.0, score_gap):
                break
        self.domain_to_score_new = EvaluationSet.sum_domain_scores(self.domain_to_violations_new)

    @staticmethod
    def sum_domain_scores(domain_to_violations: Dict[Domain, List[Evaluation]]) -> Dict[Domain, float]:
        # NOTE: this filters out the fixed domains,
        # but we keep them in eval_set for the sake of reports
        domain_to_score = {
            domain: sum(violation.score for violation in domain_violations)
            for domain, domain_violations in domain_to_violations.items()
            if not domain.fixed
        }
        return domain_to_score

    def calculate_score_gap(self, fixed: Optional[bool] = None) -> Optional[float]:
        # fixed is None (all violations), True (fixed violations), or False (nonfixed violations)
        # total score of evaluations - total score of new evaluations
        assert len(self.evaluations) > 0
        total_gap = 0.0
        for ((constraint, part), eval_new) in items_2d(self.evaluations_new):
            eval_old = self.evaluations[constraint][part]
            assert eval_old.part.fixed == eval_new.part.fixed
            assert eval_old.violated == (eval_old.score > 0)
            assert eval_new.violated == (eval_new.score > 0)
            if fixed is None or eval_old.part.fixed == fixed:
                total_gap += eval_old.score - eval_new.score
        return total_gap

    def calculate_initial_score_gap(self, design: Design, domains_new: List[Domain]) -> float:
        # before evaluations_new is populated, we need to calculate the total score of evaluations
        # on parts affected by domains_new, which is the score gap assuming all new evaluations come up 0
        score_gap = 0.0
        for constraint in self.constraints:
            parts = find_parts_to_check(constraint, design, domains_new)
            for part in parts:
                ev = self.evaluations[constraint][part]
                score_gap += ev.score
        return score_gap

    def evaluate_constraint(self,
                            constraint: Constraint[DesignPart],
                            design: Design,  # only used with DesignConstraint
                            score_gap: Optional[float],
                            domains_new: Optional[Iterable[Domain]],
                            ) -> float:
        # returns score gap = score(old evals) - score(new evals);
        # if gap > 0, then new evals haven't added up to
        assert ((score_gap is None and domains_new is None) or
                (score_gap is not None and domains_new is not None))

        parts = find_parts_to_check(constraint, design, domains_new)

        # measure violations of constraints and collect in list of triples (part, score, summary)
        violating_parts_scores_summaries: List[Tuple[DesignPart, float, str]] = []
        if isinstance(constraint, SingularConstraint):
            if not constraint.parallel or len(parts) == 1 or nc.cpu_count() == 1:
                for part in parts:
                    seqs = tuple(indv_part.sequence() for indv_part in part.individual_parts())
                    score, summary = constraint.call_evaluate(seqs, part)
                    violating_parts_scores_summaries.append((part, score, summary))
                    if score > 0.0:
                        if score_gap is not None:
                            score_gap -= score
                            if _is_significantly_greater(0.0, score_gap):
                                break
            else:
                raise NotImplementedError('TODO: implement parallelization')

        elif isinstance(constraint, (BulkConstraint, DesignConstraint)):
            if isinstance(constraint, DesignConstraint):
                violating_parts_scores_summaries = constraint.call_evaluate_design(design, domains_new)
            else:
                # XXX: I don't understand the mypy error on the next line
                violating_parts_scores_summaries = constraint.call_evaluate_bulk(parts)  # type: ignore

            # we can't quit this function early,
            # but we can let the caller know to stop evaluating constraints
            total_score = sum(score for _, score, _ in violating_parts_scores_summaries)
            if score_gap is not None:
                score_gap -= total_score
        else:
            raise AssertionError(
                f'constraint {constraint} of unrecognized type {constraint.__class__.__name__}')

        # assign blame for violations to domains by looking up associated domains in each part
        evals_of_constraint = self.evaluations[constraint]
        viols_of_constraint = self.violations[constraint]
        domain_to_evals = self.domain_to_evaluations
        domain_to_viols = self.domain_to_violations
        if domains_new is not None:
            evals_of_constraint = self.evaluations_new[constraint]
            viols_of_constraint = self.violations_new[constraint]
            domain_to_evals = self.domain_to_evaluations_new
            domain_to_viols = self.domain_to_violations_new

        for part, score, summary in violating_parts_scores_summaries:
            domains = _independent_domains_in_part(part, exclude_fixed=False)
            evaluation = Evaluation(constraint=constraint, part=part, domains=domains,
                                    score=score, summary=summary, violated=score > 0)

            evals_of_constraint[part] = evaluation
            for domain in domains:
                domain_to_evals[domain].append(evaluation)

            if evaluation.violated:
                viols_of_constraint[part] = evaluation
                for domain in domains:
                    domain_to_viols[domain].append(evaluation)

        return score_gap

    def replace_with_new(self) -> None:
        # uses Evaluations in self.evaluations_new to replace equivalent ones in self.evalautions
        # same for violations

        # IMPORTANT: update scores before we start to modify the EvaluationSet
        self.total_score = self.total_score_new()
        self.total_score_fixed = self.total_score_new(True)
        self.total_score_nonfixed = self.total_score_new(False)

        # update all evaluations
        for (constraint, part), evaluation in items_2d(self.evaluations_new):
            # CONSIDER updating everything in this loop by looking up eval.violated
            self.evaluations[constraint][part] = evaluation

            # update dict mapping domain to list of evals/violations for which it is blamed
            for domain in evaluation.domains:
                self.domain_to_evaluations[domain] = self.domain_to_evaluations_new[domain]
                self.domain_to_violations[domain] = self.domain_to_violations_new[domain]

            viols_by_part = self.violations[constraint]
            if evaluation.violated:
                # if was not a violation before, increment total violations
                if part not in viols_by_part:
                    self.num_violations += 1
                    if evaluation.part.fixed:
                        self.num_violations_fixed += 1
                    else:
                        self.num_violations_nonfixed += 1
                # add it to violations, or replace existing violation
                viols_by_part[part] = evaluation
            elif not evaluation.violated and part in viols_by_part.keys():
                # otherwise remove violation if one was there from old EvaluationSet,
                # and decrement total violations
                del viols_by_part[part]
                self.num_violations -= 1
                if evaluation.part.fixed:
                    self.num_violations_fixed -= 1
                else:
                    self.num_violations_nonfixed -= 1

        self.reset_new()

        _assert_violations_are_accurate(self.evaluations, self.violations)

    def update_scores_and_counts(self) -> None:
        """
        :return: Total score of all evaluations.
        """
        self.total_score = self.total_score_fixed = self.total_score_nonfixed = 0.0
        self.num_evaluations = self.num_evaluations_nonfixed = self.num_evaluations_fixed = 0
        self.num_violations = self.num_violations_nonfixed = self.num_violations_fixed = 0

        # count evaluations
        for evaluation in values_2d(self.evaluations):
            self.num_evaluations += 1
            if evaluation.part.fixed:
                self.num_evaluations_fixed += 1
            else:
                self.num_evaluations_nonfixed += 1

        # count violations and score
        for violation in values_2d(self.violations):
            self.total_score += violation.score
            self.num_violations += 1
            if violation.part.fixed:
                self.total_score_fixed += violation.score
                self.num_violations_fixed += 1
            else:
                self.total_score_nonfixed += violation.score
                self.num_violations_nonfixed += 1

    def total_score_new(self, fixed: Optional[bool] = None) -> float:
        # return total score of all evaluations, or only fixed, or only nonfixed
        if fixed is None:
            total_score_old = self.total_score
        elif fixed is True:
            total_score_old = self.total_score_fixed
        elif fixed is False:
            total_score_old = self.total_score_nonfixed
        else:
            raise AssertionError(f'fixed should be None, True, or False, but is {fixed}')

        total_score_new = total_score_old - self.calculate_score_gap(fixed)
        return total_score_new

    def score_of_constraint(self, constraint: Constraint, violations: bool) -> float:
        # :param constraint:
        #     constraint to filter scores on
        # :return:
        #     Total score of all evaluations due to `constraint`.
        return sum(evaluation.score for evaluation in self.evaluations_of_constraint(constraint, violations))

    def score_of_constraint_nonfixed(self, constraint: Constraint, violations: bool) -> float:
        # :param constraint:
        #     constraint to filter scores on
        # :return:
        #     Total score of all nonfixed evaluations due to `constraint`.
        return sum(evaluation.score for evaluation in self.evaluations_of_constraint(constraint, violations)
                   if not evaluation.part.fixed)

    def score_of_constraint_fixed(self, constraint: Constraint, violations: bool) -> float:
        # :param constraint:
        #     constraint to filter scores on
        # :return:
        #     Total score of all fixed violations due to `constraint`.
        return sum(evaluation.score for evaluation in self.evaluations_of_constraint(constraint, violations)
                   if evaluation.part.fixed)

    def has_nonfixed_evaluations(self) -> bool:
        # :return: whether there are any nonfixed Evaluations in this EvaluationSet
        return self.num_evaluations_nonfixed > 0

    def has_nonfixed_violations(self) -> bool:
        # :return: whether there are any nonfixed Evaluations in this EvaluationSet
        return self.num_violations_nonfixed > 0

    def evaluations_of_constraint(self, constraint: Constraint, violations: bool) -> List[Evaluation]:
        dct = self.violations[constraint] if violations else self.evaluations[constraint]
        return list(dct.values())

    def evaluations_nonfixed_of_constraint(self, constraint: Constraint,
                                           violations: bool) -> List[Evaluation]:
        return [ev for ev in self.evaluations_of_constraint(constraint, violations) if not ev.part.fixed]

    def evaluations_fixed_of_constraint(self, constraint: Constraint, violations: bool) -> List[Evaluation]:
        return [ev for ev in self.evaluations_of_constraint(constraint, violations) if ev.part.fixed]

    def num_evaluations_of(self, constraint: Constraint) -> int:
        return len(self.evaluations[constraint])

    def num_violations_of(self, constraint: Constraint) -> int:
        return len(self.violations[constraint])


def _assert_violations_are_accurate(evaluations: Dict[Constraint, Dict[nc.Part, Evaluation]],
                                    violations: Dict[Constraint, Dict[nc.Part, Evaluation]]) -> None:
    # go through all violations and ensure the violations in it are all in evaluations
    for (constraint, part), viol in items_2d(violations):
        assert constraint in evaluations.keys()
        evals_by_part = evaluations[constraint]
        assert part in evals_by_part.keys()
        ev = evals_by_part[part]
        assert viol == ev
        assert viol.violated  # also assert that they really are violated

    # now go in reverse and ensure the keys in evaluations not in violations are all not violated
    for (constraint, part), ev in items_2d(evaluations):
        if part in violations[constraint].keys():
            viol = violations[constraint][part]
            assert viol == ev
            assert ev.violated
        else:
            assert not ev.violated


@dataclass(frozen=True)
class Evaluation(Generic[DesignPart]):
    # Represents a violation of a single :any:`Constraint` in a :any:`Design`.
    # The "part" of the :any:`Design` that was evaluated for the constraint is generic type `DesignPart`
    # (e.g., for :any:`StrandPairConstraint`, DesignPart = :any:`Pair` [:any:`Strand`]).

    constraint: Constraint
    # :any:`Constraint` that was evaluated to result in this :any:`Evaluation`.

    violated: bool
    # whether the :any:`Constraint` was violated last time it was evaluated

    part: DesignPart
    # DesignPart that caused this violation

    domains: FrozenSet[Domain]  # = field(init=False, hash=False, compare=False, default=None)
    # :any:`Domain`'s that were involved in violating :py:data:`Evaluation.constraint`

    summary: str

    score: float

    def __init__(self, constraint: Constraint, violated: bool, part: DesignPart, domains: Iterable[Domain],
                 score: float, summary: str) -> None:
        # :param constraint:
        #     :any:`Constraint` that was violated to result in this
        # :param domains:
        #     :any:`Domain`'s that were involved in violating :py:data:`Evaluation.constraint`
        # :param score:
        #     total "score" of this violation, typically something like an excess energy over a
        #     threshold, squared, multiplied by the :data:`Constraint.weight`
        object.__setattr__(self, 'constraint', constraint)
        object.__setattr__(self, 'violated', violated)
        object.__setattr__(self, 'part', part)
        domains_frozen = frozenset(domains)
        object.__setattr__(self, 'domains', domains_frozen)
        object.__setattr__(self, 'score', score)
        object.__setattr__(self, 'summary', summary)

    def __repr__(self) -> str:
        return f'Evaluation({self.constraint.short_description}, score={self.score:.2f}, ' \
               f'summary={self.summary}, violated={self.violated})'

    def __str__(self) -> str:
        return repr(self)

    # Evaluation equality based on identity; different Evaluations in memory are considered different,
    # even if all data between them matches. Don't create the same Evaluation twice!
    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        return self is other


####################################################################################
# report generating functions

def summary_of_constraints(constraints: Iterable[Constraint], report_only_violations: bool,
                           eval_set: EvaluationSet) -> str:
    summaries: List[str] = []

    # other constraints
    for constraint in constraints:
        summary = summary_of_constraint(constraint, report_only_violations, eval_set)
        summaries.append(summary)

    score = eval_set.total_score
    score_unfixed = eval_set.total_score_nonfixed
    score_total_summary = f'total score of constraint violations: {score:.2f}'
    score_unfixed_summary = f'total score of unfixed constraint violations: {score_unfixed:.2f}'

    summary = (score_total_summary + '\n'
               + (score_unfixed_summary + '\n\n' if score_unfixed != score else '\n')
               + '\n'.join(summaries))

    return summary


def summary_of_constraint(constraint: Constraint, report_only_violations: bool,
                          evaluation_set: EvaluationSet) -> str:
    if isinstance(constraint, (DomainConstraint, StrandConstraint,
                               DomainPairConstraint, StrandPairConstraint, ComplexConstraint,
                               DomainsConstraint, StrandsConstraint,
                               DomainPairsConstraint, StrandPairsConstraint, ComplexesConstraint)):
        summaries = []
        num_evals = evaluation_set.num_evaluations_of(constraint)
        num_viols = evaluation_set.num_violations_of(constraint)
        num_violations = 0
        part_type_name = constraint.part_name()

        evals_nonfixed = evaluation_set.evaluations_nonfixed_of_constraint(constraint, report_only_violations)
        evals_fixed = evaluation_set.evaluations_fixed_of_constraint(constraint, report_only_violations)

        some_fixed_evals = len(evals_fixed) > 0

        for evals, header_name in [(evals_nonfixed, f"unfixed {part_type_name}s"),
                                   (evals_fixed, f"fixed {part_type_name}s")]:
            if len(evals) == 0:
                continue

            max_part_name_length = max(len(violation.part.name) for violation in evals)
            num_violations += len(evals)

            lines_and_scores: List[Tuple[str, float]] = []
            for ev in evals:
                line = f'{part_type_name} {ev.part.name:{max_part_name_length}}: ' \
                       f'{ev.summary};  score: {ev.score:.2f}'
                lines_and_scores.append((line, ev.score))

            lines_and_scores.sort(key=lambda line_and_score: line_and_score[1], reverse=True)

            lines = (line for line, _ in lines_and_scores)
            content = '\n'.join(lines)

            # only put header to distinguish fixed from unfixed violations if there are some fixed
            full_header = _small_header(header_name, "=") if some_fixed_evals else ''
            summary = full_header + f'\n{content}\n'
            summaries.append(summary)

        if report_only_violations and num_violations != num_viols:
            assert num_violations == num_viols

        content = ''.join(summaries)
        report = ConstraintReport(constraint=constraint, content=content,
                                  num_violations=num_viols, num_evaluations=num_evals)

    elif isinstance(constraint, DesignConstraint):
        raise NotImplementedError()
    else:
        content = f'skipping summary of constraint {constraint.description}; ' \
                  f'unrecognized type {type(constraint)}'
        report = ConstraintReport(constraint=constraint, content=content, num_violations=0, num_evaluations=0)

    summary = add_header_to_content_of_summary(report, evaluation_set)
    return summary


def add_header_to_content_of_summary(report: ConstraintReport, eval_set: EvaluationSet) -> str:
    score = eval_set.score_of_constraint(report.constraint, True)
    score_unfixed = eval_set.score_of_constraint_nonfixed(report.constraint, True)

    if score != score_unfixed:
        summary_score_unfixed = f'\n* unfixed score of violations: {score_unfixed:.2f}'
    else:
        summary_score_unfixed = None

    indented_content = textwrap.indent(report.content, '  ')

    summary = f'''
**{"*" * len(report.constraint.description)}
* {report.constraint.description}
# evaluations: {report.num_evaluations}
* violations:  {report.num_violations}
* score of violations: {score:.2f}{"" if summary_score_unfixed is None else summary_score_unfixed}
{indented_content}'''

    return summary


def _small_header(header: str, delim: str) -> str:
    width = len(header)
    return f'\n{header}\n{delim * width}'


@dataclass
class ConstraintReport:
    """
    Represents a report on how well a design did on a constraint.
    """

    constraint: Optional['Constraint']
    """
    The :any:`Constraint` to report on. This can be None if the :any:`Constraint` object is not available
    at the time the :meth:`Constraint.generate_summary` function is defined. If so it will be
    automatically inserted by the report generating code."""

    content: str
    """
    Summary of constraint information on the :any:`Design`.
    """

    num_violations: int
    """
    Total number of "parts" of the :any:`Design` (e.g., :any:`Strand`'s, pairs of :any:`Domain`'s) that
    violated the constraint.
    """

    num_evaluations: int
    """
    Total number of "parts" of the :any:`Design` (e.g., :any:`Strand`'s, pairs of :any:`Domain`'s) that
    were checked against the constraint.
    """
