import nuad.search as ns  # type: ignore
import nuad.constraints as nc  # type: ignore
import nuad.vienna_nupack as nv  # type: ignore


def main():
    domain_length = 15
    # energy_constraint = dc.NearestNeighborEnergyConstraint(low_energy=-9.2, high_energy=-7)
    numpy_constraints = [  # energy_constraint,
        nc.RunsOfBasesConstraint(['C', 'G'], 4),
        nc.RunsOfBasesConstraint(['A', 'T'], 4)
    ]
    domain_pool = nc.DomainPool(f'length-{domain_length} domains', domain_length,
                                numpy_constraints=numpy_constraints, replace_with_close_sequences=True)

    random_seed = 0
    strands = [nc.Strand([f'{i}' for i in range(1, 50)])]
    for strand in strands:
        for domain in strand.domains:
            domain.pool = domain_pool
    params = ns.SearchParameters(
        constraints=[nc.nupack_strand_complex_free_energy_constraint(
            threshold=-1.5, temperature=52, short_description='StrandSS', parallel=False)],
        out_directory='output/hamming_dist_test',
        # weigh_violations_equally=True
        report_only_violations=False,
        random_seed=random_seed,
        max_iterations=None)
    params.force_overwrite = True  # directly deletes output folder contents w/o user input
    # params.report_delay = 0.0

    design = nc.Design(strands)
    ns.search_for_dna_sequences(design, params)


if __name__ == '__main__':
    main()
