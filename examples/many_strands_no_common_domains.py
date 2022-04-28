from typing import NamedTuple, Optional
import argparse
import os
import logging
from typing import List

import nuad.constraints as dc  # type: ignore
import nuad.vienna_nupack as dv  # type: ignore
import nuad.search as ds  # type: ignore
from nuad.constraints import NumpyConstraint


# command-line arguments
class CLArgs(NamedTuple):
    directory: str
    restart: bool


def parse_command_line_arguments() -> CLArgs:
    default_directory = os.path.join('output', ds.script_name_no_ext())

    parser = argparse.ArgumentParser(  # noqa
        description='Small example design for testing.',
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


def main() -> None:
    args: CLArgs = parse_command_line_arguments()

    # dc.logger.setLevel(logging.DEBUG)
    dc.logger.setLevel(logging.INFO)

    random_seed = 1

    # many 4-domain strands with no common domains, 4 domains each, every domain length = 10
    # just for testing parallel processing

    # num_strands = 2
    # num_strands = 5
    num_strands = 10
    # num_strands = 50
    # num_strands = 100
    # num_strands = 355

    #                     si         wi         ni         ei
    # strand i is    [----------|----------|----------|---------->
    strands = [dc.Strand([f's{i}', f'w{i}', f'n{i}', f'e{i}']) for i in range(num_strands)]

    # some_fixed = False
    some_fixed = True
    if some_fixed:
        # fix all domains of strand 0 and one domain of strand 1
        for domain in strands[0].domains:
            domain.set_fixed_sequence('ACGTACGTAC')
        strands[1].domains[0].set_fixed_sequence('ACGTACGTAC')

    parallel = False
    # parallel = True

    design = dc.Design(strands)

    numpy_constraints: List[NumpyConstraint] = [
        # dc.NearestNeighborEnergyConstraint(-9.3, -9.0, 52.0),
        # dc.BaseCountConstraint(base='G', high_count=1),
        # dc.BaseEndConstraint(bases=('C', 'G')),
        # dc.RunsOfBasesConstraint(['C', 'G'], 4),
        # dc.RunsOfBasesConstraint(['A', 'T'], 4),
        # dc.BaseEndConstraint(bases=('A', 'T')),
        # dc.BaseEndConstraint(bases=('C', 'G'), distance_from_end=1),
        # dc.BaseAtPositionConstraint(bases='T', position=3),
        # dc.ForbiddenSubstringConstraint(['GGGG', 'CCCC']),
        # dc.RestrictBasesConstraint(bases=['A', 'T', 'C']),
    ]

    # def nupack_binding_energy_in_bounds(seq: str) -> bool:
    #     energy = dv.binding_complement(seq, 52)
    #     dc.logger.debug(f'nupack complement binding energy = {energy}')
    #     return -11 < energy < -9
    #
    # # list of functions:
    # sequence_constraints: List[SequenceConstraint] = [
    #     # nupack_binding_energy_in_bounds,
    # ]

    replace_with_close_sequences = True
    # replace_with_close_sequences = False
    domain_pool_10 = dc.DomainPool(f'length-10_domains', 10,
                                   numpy_constraints=numpy_constraints,
                                   replace_with_close_sequences=replace_with_close_sequences,
                                   )
    domain_pool_11 = dc.DomainPool(f'length-11_domains', 11,
                                   numpy_constraints=numpy_constraints,
                                   replace_with_close_sequences=replace_with_close_sequences,
                                   )

    if some_fixed:
        for strand in strands[1:]:  # skip all domains on strand 0 since all its domains are fixed
            for domain in strand.domains[:2]:
                if domain.name != 's1':  # skip for s1 since that domain is fixed
                    domain.pool = domain_pool_10
            for domain in strand.domains[2:]:
                domain.pool = domain_pool_11
    else:
        for strand in strands:
            for domain in strand.domains[:2]:
                domain.pool = domain_pool_10
            for domain in strand.domains[2:]:
                domain.pool = domain_pool_11

    # have to set nupack_complex_secondary_structure_constraint after DomainPools are set,
    # so that we know the domain lengths
    strand_complexes = [dc.Complex((strand,)) for i, strand in enumerate(strands[2:])]
    strand_base_pair_prob_constraint = dc.nupack_complex_base_pair_probability_constraint(
        strand_complexes=strand_complexes)

    domain_nupack_ss_constraint = dc.nupack_domain_complex_free_energy_constraint(
        threshold=-0.0, temperature=52, short_description='DomainSS')

    domain_pairs_rna_duplex_constraint = dc.rna_duplex_domain_pairs_constraint(
        threshold=-2.0, temperature=52, short_description='DomainPairRNA')

    domain_pair_nupack_constraint = dc.nupack_domain_pair_constraint(
        threshold=-0.5, temperature=52, short_description='DomainPairNUPACK',
        parallel=parallel)

    strand_pairs_rna_duplex_constraint = dc.rna_duplex_strand_pairs_constraint(
        threshold=-1.0, temperature=52, short_description='StrandPairRNA', parallel=parallel)

    strand_individual_ss_constraint = dc.nupack_strand_complex_free_energy_constraint(
        threshold=-1.0, temperature=52, short_description='StrandSS', parallel=parallel)

    strand_pair_nupack_constraint = dc.nupack_strand_pair_constraint(
        threshold=3.0, temperature=52, short_description='StrandPairNUPACK', parallel=parallel, weight=0.1)

    params = ds.SearchParameters(constraints=[
        # domain_nupack_ss_constraint,
        # strand_individual_ss_constraint,
        # strand_pair_nupack_constraint,
        # domain_pair_nupack_constraint,
        # domain_pairs_rna_duplex_constraint,
        # strand_pairs_rna_duplex_constraint,
        # strand_base_pair_prob_constraint,
        dc.domains_not_substrings_of_each_other_constraint(),
    ],
        out_directory=args.directory,
        restart=args.restart,
        random_seed=random_seed,
        max_iterations=None,
        save_sequences_for_all_updates=True,
        save_report_for_all_updates=True,
        save_design_for_all_updates=True,
        force_overwrite=True,
    )
    ds.search_for_dna_sequences(design, params)


if __name__ == '__main__':
    main()
