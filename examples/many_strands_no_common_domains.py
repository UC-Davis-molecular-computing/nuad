from typing import NamedTuple, Optional
import argparse
import os
import logging
from typing import List

import dsd.constraints as dc  # type: ignore
import dsd.vienna_nupack as dv  # type: ignore
import dsd.search as ds  # type: ignore
from dsd.constraints import NumpyConstraint, SequenceConstraint


# command-line arguments
class CLArgs(NamedTuple):
    directory: str
    initial_design_filename: Optional[str]
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
    parser.add_argument('-i', '--initial_design', type=str, default=None,
                        help='(Probably you don\'t want this option, and would prefer -r/--restart.)'
                             'name of JSON filename of initial design. If specified, then the DNA sequences '
                             'of domains will start equal to what they are in the design. This is useful, '
                             'for instance, when starting a DNA sequence design search from a sequence '
                             'assignment that was saved during a prior execution of the dsd sequence '
                             'designer. The strands and domains of the saved design will be compared to '
                             'the scadnano design (though not the DNA sequences).')

    args = parser.parse_args()

    return CLArgs(directory=args.output_dir,
                  initial_design_filename=args.initial_design, restart=args.restart)


def main() -> None:
    args: CLArgs = parse_command_line_arguments()

    # dc.logger.setLevel(logging.DEBUG)
    dc.logger.setLevel(logging.INFO)

    random_seed = 2

    # many 4-domain strands with no common domains, 4 domains each, every domain length = 10
    # just for testing parallel processing

    # num_strands = 2
    # num_strands = 5
    # num_strands = 10
    num_strands = 50
    # num_strands = 100
    # num_strands = 355

    #                     si         wi         ni         ei
    # strand i is    [----------|----------|----------|---------->
    strands = [dc.Strand([f's{i}', f'w{i}', f'n{i}', f'e{i}']) for i in range(num_strands)]

    threaded = False
    # threaded = True

    domain_pairs_rna_duplex_constraint = dc.rna_duplex_domain_pairs_constraint(
        threshold=-2.0, temperature=52, short_description='DomainPairNoCompl')

    domain_pair_nupack_constraint = dc.nupack_domain_pair_constraint(
        threshold=-5.5, temperature=52, short_description='DomainPairNoCompl',
        threaded=threaded)

    strand_pairs_rna_duplex_constraint = dc.rna_duplex_strand_pairs_constraint(
        threshold=-1.0, temperature=52, short_description='StrandPairNoCompl', threaded=threaded)

    # strand_individual_ss_constraint = dc.nupack_strand_secondary_structure_constraint(
    #     threshold=-1.5, temperature=52, short_description='StrandSS', threaded=threaded)

    strand_individual_ss_constraint = dc.nupack_strand_secondary_structure_constraint(
        threshold=-1.5, temperature=52, short_description='StrandSS', threaded=threaded)

    strand_pair_nupack_constraint = dc.nupack_strand_pair_constraint(
        threshold=-5.5, temperature=52, short_description='StrandPairNoCompl', threaded=threaded, weight=0.1)

    design = dc.Design(strands,
                       constraints=[
                           strand_individual_ss_constraint,
                           strand_pair_nupack_constraint,
                           # domain_pair_nupack_constraint,
                           # domain_pairs_rna_duplex_constraint,
                           # dc.domains_not_substrings_of_each_other_domain_pair_constraint(),
                           # strand_pairs_rna_duplex_constraint,
                       ])

    numpy_constraints: List[NumpyConstraint] = [
        # dc.NearestNeighborEnergyConstraint(-9.5, -9.0, 52.0),
        # dc.BaseCountConstraint(base='G', high_count=1),
        # dc.BaseEndConstraint(bases=('C', 'G')),
        dc.RunsOfBasesConstraint(['C', 'G'], 4),
        dc.RunsOfBasesConstraint(['A', 'T'], 4),
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
    domain_pool_10 = dc.DomainPool(f'length-{10}_domains', 10,
                                numpy_constraints=numpy_constraints,
                                replace_with_close_sequences=replace_with_close_sequences,
                                )
    domain_pool_11 = dc.DomainPool(f'length-{11}_domains', 11,
                                numpy_constraints=numpy_constraints,
                                replace_with_close_sequences=replace_with_close_sequences,
                                )

    for strand in strands:
        for domain in strand.domains[:2]:
            domain.pool = domain_pool_10
        for domain in strand.domains[2:]:
            domain.pool = domain_pool_11

    params = ds.SearchParameters(out_directory=args.directory,
                                 restart=args.restart,
                                 report_only_violations=True,
                                 random_seed=random_seed,
                                 max_iterations=None,
                                 # force_overwrite=True,
                                 # report_delay=0.0,
                                 )
    params.force_overwrite = True
    ds.search_for_dna_sequences(design, params)


if __name__ == '__main__':
    main()
