from typing import NamedTuple, Optional
import os
import argparse

import dsd.constraints as dc  # type: ignore
import dsd.vienna_nupack as dv  # type: ignore
import dsd.search as ds  # type: ignore


# command-line arguments
class CLArgs(NamedTuple):
    directory: str
    initial_design_filename: Optional[str]
    weigh_constraint_violations_equally: bool
    restart: bool


def parse_command_line_arguments() -> CLArgs:
    default_directory = os.path.join('output', ds.script_name_no_ext())

    parser = argparse.ArgumentParser(  # noqa
        description='Small example design for testing.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output-dir', type=str, default=default_directory,
                        help='directory in which to place output files')
    parser.add_argument('-we', '--weigh_violations_equally', action='store_true',
                        help='Weigh violations of each constraint equally (only pay attention to whether '
                             'constraint returns 0.0 or a positive value, converting all positive values '
                             'to 1.0).')
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

    return CLArgs(directory=args.output_dir, initial_design_filename=args.initial_design,
                  weigh_constraint_violations_equally=args.weigh_violations_equally, restart=args.restart)


def main() -> None:
    args: CLArgs = parse_command_line_arguments()

    # domain lengths
    #  9: =========
    # 10: ==========
    # 11: ===========
    # 12: ============
    #
    #             /=========--============>
    #             |    n3*         e3*
    #             |        strand 2
    #             |    e1*          n2*
    #             \===========--==========]
    # /==========--===========>/==========--===========>
    # |     n1         e1      |    n2          e2
    # |        strand 0        |        strand 1
    # |     w1           s1    |     w2           s2
    # \============--=========]\============--=========]
    #               /=========--============>
    #               |    s1*         w2*
    #               |        strand 3
    #               |    w4*          s4*
    #               \===========--==========]

    strand0: dc.Strand[str] = dc.Strand(['s1', 'w1', 'n1', 'e1'], name='strand 0')
    strand1: dc.Strand[str] = dc.Strand(['s2', 'w2', 'n2', 'e2'], name='strand 1')
    strand2: dc.Strand[None] = dc.Strand(['n2*', 'e1*', 'n3*', 'e3*'], name='strand 2')
    strand3: dc.Strand[str] = dc.Strand(['s4*', 'w4*', 's1*', 'w2*'], name='strand 3')
    strands = [strand0, strand1, strand2, strand3]

    strand_pairs_no_comp_constraint = dc.rna_duplex_strand_pairs_constraint(
        threshold=-1.0, temperature=52, short_description='StrandPairNoCompl',
        pairs=((strand0, strand1), (strand2, strand3))
    )
    strand_pairs_comp_constraint = dc.rna_duplex_strand_pairs_constraint(
        threshold=-7.0, temperature=52, short_description='StrandPairCompl',
        pairs=[(strand0, strand2), (strand0, strand3), (strand1, strand2), (strand1, strand3)]
    )
    strand_individual_ss_constraint = dc.nupack_strand_secondary_structure_constraint(
        threshold=-0.0, temperature=52, short_description='StrandSS')

    initial_design = dc.Design(strands,
                               constraints=[strand_pairs_no_comp_constraint,
                                            # strand_pairs_comp_constraint,
                                            # strand_individual_ss_constraint,
                                            # dc.domains_not_substrings_of_each_other_domain_pair_constraint(),
                                            ])

    if args.initial_design_filename is not None:
        with open(args.initial_design_filename, 'r') as file:
            design_json_str: str = file.read()
        design = dc.Design.from_json(design_json_str)
    else:
        design = initial_design

    numpy_constraints = [
        dc.NearestNeighborEnergyConstraint(-9.5, -9.0, 52.0),
        dc.BaseCountConstraint(base='G', high_count=1),
        dc.RunsOfBasesConstraint(['C', 'G'], 4),
        dc.RunsOfBasesConstraint(['A', 'T'], 4),
    ]

    def nupack_binding_energy_in_bounds(seq: str) -> bool:
        energy = dv.binding_complement_3(seq, 52)
        dc.logger.debug(f'nupack complement binding energy = {energy}')
        return -12 < energy < -8

    sequence_constraints = [
        # nupack_binding_energy_in_bounds,
    ]

    lengths = [9, 10, 11, 12]
    domain_pools = {
        length:
            dc.DomainPool(f'length-{length} domains', length,
                          numpy_constraints=numpy_constraints,
                          sequence_constraints=sequence_constraints) for length in lengths
    }

    for strand in [strand0, strand1]:
        strand.domains[0].pool = domain_pools[lengths[0]]
        strand.domains[1].pool = domain_pools[lengths[3]]
        strand.domains[2].pool = domain_pools[lengths[1]]
        strand.domains[3].pool = domain_pools[lengths[2]]

    strand2.domains[2].pool = domain_pools[lengths[0]]
    strand2.domains[3].pool = domain_pools[lengths[3]]
    strand3.domains[0].pool = domain_pools[lengths[1]]
    strand3.domains[1].pool = domain_pools[lengths[2]]

    ds.search_for_dna_sequences(design=design,
                                # weigh_violations_equally=True,
                                report_delay=0.0,
                                out_directory=args.directory,
                                restart=args.restart,
                                # force_overwrite=True,
                                report_only_violations=False,
                                random_seed=1,
                                )

    # for strand in design.strands:
    #     print(f'strand seq = {strand.sequence(spaces_between_domains=True)}')

    # print('violations of constraints:')
    # pprint(ds.violations_of_constraints(design))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
