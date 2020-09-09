import dsd.constraints as dc  # type: ignore
import dsd.vienna_nupack as dv  # type: ignore
import dsd.search as ds  # type: ignore
import logging


# from pprint import pprint


def main():
    # dc.logger.setLevel(logging.DEBUG)
    dc.logger.setLevel(logging.INFO)

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

    strand0: dc.Strand[str] = dc.Strand(['s1', 'w1', 'n1', 'e1'], label='strand 0')
    strand1: dc.Strand[str] = dc.Strand(['s2', 'w2', 'n2', 'e2'], label='strand 1')
    strand2: dc.Strand[None] = dc.Strand(['n2*', 'e1*', 'n3*', 'e3*'], label=None)
    strand3: dc.Strand[str] = dc.Strand(['s4*', 'w4*', 's1*', 'w2*'], label='strand 3')
    strands = [strand0, strand1, strand2, strand3]

    strand3.label = 'abc'
    # strand2.label = 123

    print(len(strand3.label))
    print(len(strand2.label))

    strand_pairs_no_comp_constraint = dc.rna_duplex_strand_pairs_constraint(
        threshold=-1.5, temperature=52, short_description='StrandPairNoCompl',
        pairs=((strand0, strand1), (strand2, strand3))
    )
    strand_pairs_comp_constraint = dc.rna_duplex_strand_pairs_constraint(
        threshold=-6.0, temperature=52, short_description='StrandPairCompl',
        pairs=[(strand0, strand2), (strand0, strand3), (strand1, strand2), (strand1, strand3)]
    )
    strand_individual_ss_constraint = dc.nupack_strand_secondary_structure_constraint(
        threshold=-1.0, temperature=52, short_description='StrandSS')

    design = dc.Design(strands,
                       domain_pair_constraints=[
                           dc.domains_not_substrings_of_each_other_domain_pair_constraint()],
                       strand_constraints=[strand_individual_ss_constraint],
                       strand_pairs_constraints=[strand_pairs_no_comp_constraint,
                                                 strand_pairs_comp_constraint]
                       )

    numpy_constraints = [
        dc.NearestNeighborEnergyConstraint(-9.5, -9.0, 52.0),
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
        return -12 < energy < -8

    sequence_constraints = [
        # nupack_binding_energy_in_bounds,
    ]

    lengths = [9, 10, 11, 12]
    # lengths = [4, 5, 6, 7]
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

    ds.search_for_dna_sequences(design)

    for strand in design.strands:
        print(f'strand seq = {strand.sequence(spaces_between_domains=True)}')

    # print('violations of constraints:')
    # pprint(ds.violations_of_constraints(design))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
