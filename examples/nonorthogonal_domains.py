from __future__ import annotations

import itertools
from typing import NamedTuple
import argparse
import os
import logging

import nuad.constraints as nc  # type: ignore
import nuad.vienna_nupack as nv  # type: ignore
import nuad.search as ns  # type: ignore


def main() -> None:
    args: CLArgs = parse_command_line_arguments()

    # dc.logger.setLevel(logging.DEBUG)
    nc.logger.setLevel(logging.INFO)

    random_seed = 1

    domain_length = 20

    # many 1-domain strands with no common domains, every domain length = 20

    # num_strands = 3
    # num_strands = 5
    num_strands = 10
    # num_strands = 11
    # num_strands = 50
    # num_strands = 100
    # num_strands = 355

    design = nc.Design()
    #                         di
    # strand i is    [------------------>
    for i in range(num_strands):
        design.add_strand([f'd{i}'])

    some_fixed = False
    # some_fixed = True
    if some_fixed:
        # fix domain of strand 0
        for domain in design.strands[0].domains:
            domain.set_fixed_sequence('A' * domain_length)

    domain_pool_20 = nc.DomainPool(f'length-20_domains', domain_length)

    for i, strand in enumerate(design.strands):
        if some_fixed and i == 0:
            continue
        for domain in strand.domains:
            domain.pool = domain_pool_20

    min_energy = -domain_length / 4
    max_distance = num_strands

    epsilon = 1.0
    thresholds = {}
    # for d1, d2 in itertools.combinations_with_replacement(design.domains, 2):
    for d1, d2 in itertools.combinations(design.domains, 2):
        if d1.name > d2.name:
            d1, d2 = d2, d1
        name1 = d1.name
        name2 = d2.name
        idx1 = int(name1[1:])
        idx2 = int(name2[1:])
        target_distance = abs(idx1 - idx2) if idx1 != idx2 else -min_energy + 4
        target_energy = (target_distance / max_distance) * min_energy
        energy_low = target_energy - epsilon
        energy_high = target_energy
        thresholds[(d1, d2)] = (energy_low, energy_high)

    domain_nupack_pair_nonorth_constraint = nc.nupack_domain_pairs_nonorthogonal_constraint(
        thresholds=thresholds, temperature=37, short_description='dom pairs nonorth nupack')

    domain_rna_plex_pair_nonorth_constraint = nc.rna_plex_domain_pairs_nonorthogonal_constraint(
        thresholds=thresholds, temperature=37, short_description='dom pairs nonorth plex')

    params = ns.SearchParameters(constraints=[
        domain_nupack_pair_nonorth_constraint,
        # domain_rna_plex_pair_nonorth_constraint,
    ],
        out_directory=args.directory,
        restart=args.restart,
        random_seed=random_seed,
        max_iterations=None,
        # save_sequences_for_all_updates=True,
        save_report_for_all_updates=True,
        # save_design_for_all_updates=True,
        force_overwrite=True,
        # log_time=True,
        scrolling_output=False,
    )
    ns.search_for_sequences(design, params)


# command-line arguments
class CLArgs(NamedTuple):
    directory: str
    restart: bool


def parse_command_line_arguments() -> CLArgs:
    default_directory = os.path.join('output', ns.script_name_no_ext())

    parser = argparse.ArgumentParser(  # noqa
        description='Small example using nonorthogonal domains.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output-dir', type=str, default=default_directory,
                        help='directory in which to place output files')
    parser.add_argument('-r', '--restart', action='store_true',
                        help='If true, then assumes output directory contains output of search that was '
                             'cancelled, to restart '
                             'from. Similar to -i option, but will automatically find the most recent design '
                             '(assuming they are numbered with a number such as -84), and will start the '
                             'numbering from there (i.e., the next files to be written upon improving the '
                             'design will have -85).')

    args = parser.parse_args()

    return CLArgs(directory=args.output_dir, restart=args.restart)


if __name__ == '__main__':
    main()
