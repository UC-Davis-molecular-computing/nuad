import sys
import os
import logging
from typing import List

import dsd.constraints as dc  # type: ignore
import dsd.vienna_nupack as dv  # type: ignore
import dsd.search as ds  # type: ignore
from dsd.constraints import NumpyConstraint, SequenceConstraint


def main() -> None:
    # dc.logger.setLevel(logging.DEBUG)
    dc.logger.setLevel(logging.INFO)

    random_seed = 1

    threaded_domain_constraints = False
    # threaded_domain_constraints = True

    threaded_strand_constraints = False
    # threaded_strand_constraints = True

    threaded_domain_pair_constraints = False
    # threaded_domain_pair_constraints = True

    # threaded_strand_pair_constraints = False
    threaded_strand_pair_constraints = True

    never_increase_weight = True

    # threaded_nupack_pairs = False
    threaded_nupack_pairs = True

    # threaded_vienna_pairs = False
    threaded_vienna_pairs = True

    # many 4-domain strands with no common domains, 4 domains each, every domain length = 10
    # just for testing parallel processing

    num_strands = 10
    # num_strands = 100
    # num_strands = 355

    strands = [dc.Strand([f's{i}', f'w{i}', f'n{i}', f'e{i}']) for i in range(num_strands)]
    # strands = [dc.Strand([f's{i}', f'w{i}']) for i in range(num_strands)]

    domain_pairs_rna_duplex_constraint = dc.rna_duplex_domain_pairs_constraint(
        threshold=-1.0, temperature=52, short_description='DomainPairNoCompl')

    domain_pair_nupack_constraint = dc.nupack_domain_pair_constraint(
        threshold=-4.5, temperature=52, short_description='DomainPairNoCompl',
        threaded4=threaded_nupack_pairs, threaded=threaded_domain_pair_constraints)

    strand_pair_nupack_constraint = dc.nupack_strand_pair_constraint(
        threshold=-5.5, temperature=52, short_description='StrandPairNoCompl')

    strand_pairs_no_comp_constraint = dc.rna_duplex_strand_pairs_constraint(
        threshold=-1.0, temperature=52, short_description='StrandPairNoCompl', threaded=threaded_vienna_pairs)

    strand_individual_ss_constraint = dc.nupack_strand_secondary_structure_constraint(
        threshold=-1.5, temperature=52, short_description='StrandSS', threaded=threaded_strand_constraints)

    design = dc.Design(strands,
                       # domain_pair_constraints=[
                       #     dc.domains_not_substrings_of_each_other_domain_pair_constraint()],
                       # domain_pairs_constraints=[domain_pairs_rna_duplex_constraint],
                       # domain_pair_constraints=[domain_pair_nupack_constraint],
                       # strand_pair_constraints=[strand_pair_nupack_constraint],
                       # strand_constraints=[strand_individual_ss_constraint],
                       strand_pairs_constraints=[strand_pairs_no_comp_constraint]
                       )

    numpy_constraints: List[NumpyConstraint] = [
        # dc.NearestNeighborEnergyConstraint(-9.5, -9.0, 52.0),
        dc.BaseCountConstraint(base='G', high_count=1),
        # dc.BaseEndConstraint(bases=('C', 'G')),
        dc.RunsOfBasesConstraint(['C', 'G'], 4),
        dc.RunsOfBasesConstraint(['A', 'T'], 4),
        # dc.BaseEndConstraint(bases=('A', 'T')),
        # dc.BaseEndConstraint(bases=('C', 'G'), distance_from_end=1),
        # dc.BaseAtPositionConstraint(bases='T', position=3),
        # dc.ForbiddenSubstringConstraint(['GGGG', 'CCCC']),
        # dc.RestrictBasesConstraint(bases=['A', 'T', 'C']),
    ]

    def nupack_binding_energy_in_bounds(seq: str) -> bool:
        energy = dv.binding_complement(seq, 52)
        dc.logger.debug(f'nupack complement binding energy = {energy}')
        return -11 < energy < -9

    # list of functions:
    sequence_constraints: List[SequenceConstraint] = [
        # nupack_binding_energy_in_bounds,
    ]

    length = 10
    domain_pool = dc.DomainPool(f'length-{length} domains', length,
                                numpy_constraints=numpy_constraints,
                                sequence_constraints=sequence_constraints)

    for strand in strands:
        for domain in strand.domains:
            domain.pool = domain_pool

    ds.search_for_dna_sequences(design=design,
                                never_increase_weight=never_increase_weight,
                                random_seed=random_seed,
                                directory_output_files=f'output/many_strands_no_common_domains')


if __name__ == '__main__':
    main()
