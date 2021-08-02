import dsd.search as ds  # type: ignore
import dsd.constraints as dc  # type: ignore
import dsd.vienna_nupack as dv  # type: ignore


def main():
    domain_length = 15
    # energy_constraint = dc.NearestNeighborEnergyConstraint(low_energy=-9.2, high_energy=-7)
    numpy_constraints = [  # energy_constraint,
        dc.RunsOfBasesConstraint(['C', 'G'], 4),
        dc.RunsOfBasesConstraint(['A', 'T'], 4)
    ]
    domain_pool = dc.DomainPool(f'length-{domain_length} domains', domain_length,
                                numpy_constraints=numpy_constraints, replace_with_close_sequences=True)

    random_seed = 0
    strands = [dc.Strand([f'{i}' for i in range(1, 50)])]
    for strand in strands:
        for domain in strand.domains:
            domain.pool = domain_pool
    params = ds.SearchParameters(out_directory='output/hamming_dist_test',
                                 # weigh_violations_equally=True
                                 report_only_violations=False,
                                 random_seed=random_seed,
                                 max_iterations=None)
    params.force_overwrite = True  # directly deletes output folder contents w/o user input
    # params.report_delay = 0.0
    design = dc.Design(strands, constraints=[dc.nupack_strand_secondary_structure_constraint(
        threshold=-1.5, temperature=52, short_description='StrandSS', threaded=False)])
    ds.search_for_dna_sequences(design, params)


if __name__ == '__main__':
    main()
