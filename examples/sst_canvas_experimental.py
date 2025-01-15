from dataclasses import dataclass
from typing import Optional
import argparse
import os
from typing import List, Tuple

import nuad.constraints as nc
import nuad.search as ns


# NOTE: This is an "experimental" version of the file that I use to test new ideas.
# Look at sst_canvas.py for an example of a DNA sequence designer that shows recommended use of nuad.
# DNA sequence designer for a 2D canvas of single-stranded tiles (SSTs).
# See docstring for create_design for a detailed description of the design.

def main() -> None:
    args: CLArgs = parse_command_line_arguments()

    design = create_design(width=args.width, height=args.height)

    constraints = create_constraints(design)

    params = ns.SearchParameters(
        constraints=constraints,
        out_directory=args.directory,
        restart=args.restart,
        random_seed=args.seed,
        scrolling_output=False,
        save_report_for_all_updates=True,
        force_overwrite=args.force_overwrite,
        report_only_violations=False,
        score_transfer_function=lambda score: score**5,
        # log_time=True,
    )
    ns.search_for_sequences(design, params)


# command-line arguments
@dataclass
class CLArgs:
    directory: str
    """output directory for search"""

    restart: bool
    """whether to restart a stopped search"""

    width: int
    """width of SST canvas"""

    height: int
    """height of SST canvas"""

    seed: Optional[int] = None
    """seed for random number generator; set to fixed integer for reproducibility"""

    force_overwrite: bool = False
    """whether to overwrite output files without prompting the user"""


def parse_command_line_arguments() -> CLArgs:
    default_directory = os.path.join('output', ns.script_name_no_ext())

    parser = argparse.ArgumentParser(
        description='Designs DNA sequences for a canvas of single-stranded tiles (SSTs).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-o', '--output-dir', type=str, default=default_directory,
                        help='directory in which to place output files')

    parser.add_argument('-s', '--seed', type=int,
                        help='seed for random number generator')

    parser.add_argument('-w', '--width', type=int, required=True,
                        help='width of canvas (number of tiles)')

    parser.add_argument('-ht', '--height', type=int, required=True,
                        help='height of canvas (number of tiles)')

    parser.add_argument('-r', '--restart', action='store_true',
                        help='If true, then assumes output directory contains output of search that was '
                             'cancelled, to restart from. Will automatically find the most recent design '
                             '(assuming they are indexed with a number such as 84), and will start the '
                             'numbering from there (i.e., the next files to be written upon improving the '
                             'design will have index 85).')

    parser.add_argument('-f', '--force', action='store_true',
                        help='If true, then overwrites the output files without prompting the user.')

    args = parser.parse_args()

    return CLArgs(
        directory=args.output_dir,
        width=args.width,
        height=args.height,
        seed=args.seed,
        restart=args.restart,
        force_overwrite=args.force,
    )


def create_design(width: int, height: int) -> nc.Design:
    """
    Creates an SST canvas `width` tiles width and `height` tiles high.

    For instance a width=4 x height=3 canvas looks like below, with each tile named t_x_y.
    x goes from 0 up to `width`-1.
    y goes from 0 up to `height`-1.

    Domain lengths are listed at the top in the next figure.

    .. code-block:: none

        |     10    |     11      |     10     |     11      |     10     |     11      |     10     |

                                   +==========---===========>
                                   |         t_0_2
                                   +==========---===========]
                     +===========---==========> +===========---==========>
                     |          t_0_1           |          t_1_2
                     +===========---==========] +===========---==========]
        +==========---===========> +==========---===========> +==========---===========>
        |         t_0_0            |         t_1_1            |         t_2_2
        +==========---===========] +==========---===========] +==========---===========]
                     +===========---==========> +===========---==========> +===========---==========>
                     |          t_1_0           |          t_2_1           |          t_3_2
                     +===========---==========] +===========---==========] +===========---==========]
                                   +==========---===========> +==========---===========>
                                   |         t_2_0            |         t_3_1
                                   +==========---===========] +==========---===========]
                                                +===========---==========>
                                                |          t_3_0
                                                +===========---==========]

    :param width:
        number of tiles wide to make canvas
    :param height:
        number of tiles high to make canvas
    :return:
        design with `width` x `height` canvas of SSTs
    """
    numpy_filters = [
        nc.NearestNeighborEnergyFilter(-9.3, -9.0, 52.0),  # energies should all be "close"
        nc.RunsOfBasesFilter(['C', 'G'], 4),  # forbid substrings of form {C,G}^4
        nc.ForbiddenSubstringFilter(['AAAAA', 'TTTTT']),  # forbid 5 A's in a row or 5 T's in a row
    ]

    domain_pool_10 = nc.DomainPool(f'length-10_domains', 10, numpy_filters=numpy_filters)
    domain_pool_11 = nc.DomainPool(f'length-11_domains', 11, numpy_filters=numpy_filters)

    design = nc.Design()

    for x in range(width):
        for y in range(height):
            # domains are named after the strand for which they are on the bottom,
            # so the two domains on top are starred and named after the tiles to which they bind
            # If you tilt your head 45 degrees left, then glues are
            # "north" (n), "south" (s), "west" (w), "east" (e),
            # so start with either ns_ ("north-south") or we_ ("west-east")
            # From 5' ] to 3' >, they are in the order s, w, n, e.
            #
            # Parity of x+y determines whether first and last domain (s and e) are length 11 or 10.
            #
            # e.g. this is tile t_3_5:
            #
            #          10           11
            #     +==========--===========>
            #     |  ns_2_5*     we_3_6*
            #     |
            #     |  we_3_5      ns_3_5
            #     +==========--===========]
            #
            # and this is tile t_6_1:
            #
            #          11           10
            #     +===========--==========>
            #     |  ns_5_1*      we_6_2*
            #     |
            #     |  we_6_1       ns_6_1
            #     +===========--==========]

            s_domain_name = f'ns_{x}_{y}'
            w_domain_name = f'we_{x}_{y}'
            n_domain_name = f'ns_{x - 1}_{y}*'
            e_domain_name = f'we_{x}_{y + 1}*'
            tile = design.add_strand(
                domain_names=[s_domain_name, w_domain_name, n_domain_name, e_domain_name], name=f't_{x}_{y}')

            if (x + y) % 2 == 0:
                outer_pool = domain_pool_11
                inner_pool = domain_pool_10
            else:
                outer_pool = domain_pool_10
                inner_pool = domain_pool_11

            s_domain, w_domain, n_domain, e_domain = tile.domains
            if not s_domain.has_pool():
                s_domain.pool = outer_pool
            if not e_domain.has_pool():
                e_domain.pool = outer_pool
            if not n_domain.has_pool():
                n_domain.pool = inner_pool
            if not w_domain.has_pool():
                w_domain.pool = inner_pool

    return design


@dataclass
class Thresholds:
    temperature: float = 52.0
    """Temperature in Celsius"""

    tile_ss: float = -3.0
    """NUPACK complex free energy threshold for individual tiles."""

    tile_pair_0comp: float = -4.0
    """RNAduplex complex free energy threshold for pairs tiles with no complementary domains."""

    tile_pair_1comp: float = -6.5
    """RNAduplex complex free energy threshold for pairs tiles with 1 complementary domain."""


def create_constraints(design: nc.Design) -> List[nc.Constraint]:
    thresholds = Thresholds()

    strand_individual_ss_constraint = nc.nupack_strand_free_energy_constraint(
        threshold=thresholds.tile_ss, temperature=thresholds.temperature, short_description='StrandSS')

    strand_pairs_rna_duplex_constraint_0comp, strand_pairs_rna_duplex_constraint_1comp = \
        nc.rna_duplex_strand_pairs_constraints_by_number_matching_domains(
            thresholds={0: thresholds.tile_pair_0comp, 1: thresholds.tile_pair_1comp},
            temperature=thresholds.temperature,
            short_descriptions={0: 'StrandPairRNA0Comp', 1: 'StrandPairRNA1Comp'},
            strands=design.strands,
        )

    strand_pairs_nupack_constraint_0comp, strand_pairs_nupack_constraint_1comp = \
        nc.nupack_strand_pair_constraints_by_number_matching_domains(
            thresholds={0: thresholds.tile_pair_0comp, 1: thresholds.tile_pair_1comp},
            temperature=thresholds.temperature,
            short_descriptions={0: 'StrandPairNUPACK0Comp', 1: 'StrandPairNUPACK1Comp'},
            strands=design.strands,
        )

    no_gggg_constraint = create_tile_no_gggg_constraint(weight=100)

    return [
        strand_individual_ss_constraint,
        strand_pairs_rna_duplex_constraint_0comp,
        strand_pairs_rna_duplex_constraint_1comp,
        # strand_pairs_nupack_constraint_0comp,
        # strand_pairs_nupack_constraint_1comp,
        # no_gggg_constraint,
    ]


def create_tile_no_gggg_constraint(weight: float) -> nc.StrandConstraint:
    # This shows how one might make a custom constraint, in case those in dsd.constraints are not
    # sufficient. See also source code of provided constraints in dsd/constraints.py for more examples,
    # particularly for examples that call NUPACK or ViennaRNA.

    # We already forbid GGGG in any domain, but let's also ensure we don't get GGGG in any strand
    # i.e., forbid GGGG that comes from concatenating domains, e.g.,
    #
    #               *  ***
    #      ACGATCGATG  GGGATGCATGA
    #     +==========--===========>
    #     |
    #     +==========--===========]

    def evaluate(seqs: Tuple[str, ...], strand: Optional[nc.Strand]) -> nc.Result:  # noqa
        sequence = seqs[0]
        if 'GGGG' in sequence:
            result = nc.Result(excess=1.0, summary=f'GGGG found in {sequence}', value=sequence.count('GGGG'))
        else:
            result = nc.Result(excess=0.0, summary=f'', value=0)
        return result

    description = "No GGGG allowed in strand's sequence"

    return nc.StrandConstraint(description=description, short_description='NoGGGG',
                               weight=weight, evaluate=evaluate)


if __name__ == '__main__':
    main()