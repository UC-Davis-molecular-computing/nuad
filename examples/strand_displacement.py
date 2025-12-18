from typing import List, Optional, Tuple

import nuad.constraints as nc
import nuad.search as ns  # type: ignore

import numpy as np  # noqa
import numpy.random


FULL_DOMAIN_LENGTH = 15
TOEHOLD_DOMAIN_LENGTH = 5
ALMOST_FULL_DOMAINS_LENGTH = FULL_DOMAIN_LENGTH - TOEHOLD_DOMAIN_LENGTH
CLAMP_LENGTH = 2

FULL_LONG_DOMAIN_POOL: nc.DomainPool = nc.DomainPool(
    'full_domain_pool',
    FULL_DOMAIN_LENGTH,
)
ALMOST_FULL_DOMAINS_POOL: nc.DomainPool = nc.DomainPool(
    'almost_full_domain_pool',
    ALMOST_FULL_DOMAINS_LENGTH,
)
TOEHOLD_DOMAIN_POOL: nc.DomainPool = nc.DomainPool(
    'toehold_domain_pool',
    TOEHOLD_DOMAIN_LENGTH,
)
CLAMP_DOMAIN_POOL: nc.DomainPool = nc.DomainPool(
    'clamp_domain_pool',
    CLAMP_LENGTH,
)
NON_CLAMP_POOL: nc.DomainPool = nc.DomainPool(
    'non_clamp_domain_pool',
    FULL_DOMAIN_LENGTH-CLAMP_LENGTH
)

pool_list = [FULL_LONG_DOMAIN_POOL, ALMOST_FULL_DOMAINS_POOL, TOEHOLD_DOMAIN_POOL, CLAMP_DOMAIN_POOL, NON_CLAMP_POOL]

def dependency_function(sequence: str, rng: numpy.random.Generator = numpy.random.default_rng()) -> str:

    if sequence[2] == 'A' or sequence[2] == 'T':
        return sequence[0:2]+'C'+sequence[3:]
    elif rng.random() < 0.5:
        return sequence[0:2] + 'A' + sequence[3:]
    else:
        return sequence[0:2] + 'T' + sequence[3:]



design = nc.Design()

# input strand
# X
#        X3               X2               X1
# [===============--===============--===============>

# Xi :
#         Delta           Toehold
# [= = = = = = = = = = | = = = = =>


design.add_subdomains(domain_name='x1',
                      subdomain_names_and_lengths=[('x1_delta', ALMOST_FULL_DOMAINS_LENGTH), ('x1_toe', TOEHOLD_DOMAIN_LENGTH)],
                      domain_type=nc.DomainType.ASSIGNABLE)
design.add_subdomains(domain_name='x2',
                      subdomain_names_and_lengths=[('x2_delta', ALMOST_FULL_DOMAINS_LENGTH), ('x2_toe', TOEHOLD_DOMAIN_LENGTH)],
                      domain_type=nc.DomainType.ASSIGNABLE)
design.add_subdomains(domain_name='x3',
                      subdomain_names_and_lengths=[('x3_delta', ALMOST_FULL_DOMAINS_LENGTH), ('x3_toe', TOEHOLD_DOMAIN_LENGTH)],
                      domain_type=nc.DomainType.ASSIGNABLE)
X: nc.Strand = design.add_strand(domain_names=['x3', 'x2', 'x1'], name=f'input_strand')
# print(X.domains)
X1, X2, X3 = X.domains[2], X.domains[1], X.domains[0]
# print(X1.subdomains)
X1.pool, X2.pool, X3.pool = FULL_LONG_DOMAIN_POOL, FULL_LONG_DOMAIN_POOL, FULL_LONG_DOMAIN_POOL
X1_Delta, X1_TOE = X1.subdomains[0], X1.subdomains[1]
X2_Delta, X2_TOE = X2.subdomains[0], X2.subdomains[1]
X3_Delta, X3_TOE = X3.subdomains[0], X3.subdomains[1]


# let y1_delta be dependent on x1_delta
Y1_Delta = X1_Delta.create_domain_with_mismatches(name='y1_delta', pick_dependent_seq=dependency_function)
design.domains_by_name['y1_delta'] = Y1_Delta

# Y1
#         Delta               Toe
#                               Clamp
# [= = = = = = = = = = | = = = | = =>
#                          |
#                          |
#                    Toe_non_clamp

# universal domain clamp:
clamp = nc.Domain(name='clamp')
clamp.set_fixed_sequence('GC')
design.domains_by_name['clamp'] = clamp

design.add_subdomains('y1',
                      [('y1_delta', ALMOST_FULL_DOMAINS_LENGTH), ('y1_toe', TOEHOLD_DOMAIN_LENGTH)])
design.add_subdomains('y1_toe',
                      [('y1_toe_non_clamp', TOEHOLD_DOMAIN_LENGTH-CLAMP_LENGTH), ('clamp', CLAMP_LENGTH)])
design.domains_by_name['y1_toe_non_clamp'].pool = nc.DomainPool('toe_non_clamp_domain_pool',
                                                                TOEHOLD_DOMAIN_LENGTH-CLAMP_LENGTH)
# Y2, Y3
#         non_clamp           clamp
# [= = = = = = = = = = = = = | = =>

design.add_subdomains('y2',
                      [('y2_non_clamp', FULL_DOMAIN_LENGTH-CLAMP_LENGTH), ('clamp', CLAMP_LENGTH)])
design.add_subdomains('y3',
                      [('y3_non_clamp', FULL_DOMAIN_LENGTH-CLAMP_LENGTH), ('clamp', CLAMP_LENGTH)])

design.domains_by_name['y2_non_clamp'].pool = NON_CLAMP_POOL
design.domains_by_name['y3_non_clamp'].pool = NON_CLAMP_POOL

# output strand
# Y
#        Y3               Y2               Y1          X3_Delta
# [===============--===============--===============--==========>

Y: nc.Strand = design.add_strand(domain_names=['y3', 'y2', 'y1', 'x3_delta'], name=f'output_strand')

design.compute_derived_fields()

# F1
#         X1_Delta          X2              X3               Y1
#       <==========--===============--===============--===============]
#        X1*               X2*            X3*        clamp*
#  [===============--===============--===============--==>

starred_X1 = nc.Domain.complementary_domain_name('x1')
starred_X2 = nc.Domain.complementary_domain_name('x2')
starred_X3 = nc.Domain.complementary_domain_name('x3')
starred_clamp = nc.Domain.complementary_domain_name(clamp.name)
fuel_1_top = design.add_strand(name='fuel_1_top', domain_names=['y1', 'x3', 'x2', 'x1_delta'])
fuel_1_bottom = design.add_strand(name='fuel_1_bottom',
                                  domain_names=[starred_X1, starred_X2, starred_X3, starred_clamp])

# F2
#         X2_Delta         X3             Y1                 Y2
#       <==========--===============--===============--===============]
#         X2*                X3*            Y1*      clamp*
#  [===============--===============--===============--==>

starred_Y1 = nc.Domain.complementary_domain_name('y1')
fuel_2_top = design.add_strand(name='fuel_2_top',domain_names=['y2', 'y1', 'x3', 'x2_delta'])
fuel_2_bottom = design.add_strand(name='fuel_2_bottom',
                                 domain_names=[starred_X2, starred_X3, starred_Y1, starred_clamp])

# F3
#         X3_Delta         Y1               Y2               Y3
#       <==========--===============--===============--===============]
#        X3*                Y1*            Y2*       clamp*
#  [===============--===============--===============--==>
starred_Y2 = nc.Domain.complementary_domain_name('y2')
fuel_3_top = design.add_strand(name='fuel_3_top',domain_names=['y3', 'y2', 'y1', 'x3_delta'])
fuel_3_bottom = design.add_strand(name='fuel_3_bottom',
                                 domain_names=[starred_X3, starred_Y1, starred_Y2, starred_clamp])

# reporter
#        Y1_Delta    c      Y2        c      Y3
#       <==========--===============--===============]
#        Y1*         c*      Y2*      c*       Y3*
#  [===============--===============--===============>
starred_Y3 = nc.Domain.complementary_domain_name('y3')
reporter_top = design.add_strand(name='reporter_top',domain_names=['y3', 'y2', 'y1_delta'])
reporter_bottom = design.add_strand(name='reporter_bottom',
                                 domain_names=[starred_Y1, starred_Y2, starred_Y3])

# All strands
strands = [
    X,
    fuel_1_top, fuel_1_bottom,
    fuel_2_top, fuel_2_bottom,
    fuel_3_top, fuel_3_bottom,
    reporter_top, reporter_bottom
]


X_complex = nc.Complex(X)
X_complex_constraint = nc.nupack_complex_base_pair_probability_constraint(
    strand_complexes=[X_complex])

fuel_1_complex = nc.Complex(fuel_1_top, fuel_1_bottom)
fuel_2_complex = nc.Complex(fuel_2_top, fuel_2_bottom)
fuel_3_complex = nc.Complex(fuel_3_top, fuel_3_bottom)
fuel_complex_constraint = nc.nupack_complex_base_pair_probability_constraint(
    strand_complexes=[fuel_1_complex,fuel_2_complex, fuel_3_complex])

reporter_complex = nc.Complex(reporter_top, reporter_bottom)
reporter_complex_constraint = nc.nupack_complex_base_pair_probability_constraint(
    strand_complexes=[reporter_complex])

def four_g_constraint_evaluate(seqs: Tuple[str, ...], strand: Optional[nc.Strand]) -> nc.Result:
    seq = seqs[0]
    excess = 1000 if 'GGGG' in seq else 0
    violation_str = '' if 'GGGG' not in strand.sequence() else '** violation**'
    return nc.Result(excess=excess, summary=f'{strand.name}: {strand.sequence()}{violation_str}')

four_g_constraint = nc.StrandConstraint(
    description='4GConstraint',
    short_description='4GConstraint',
    evaluate=four_g_constraint_evaluate,
    strands=tuple(strands),
)

# Constraints
constraints = [
    X_complex_constraint,
    fuel_complex_constraint,
    reporter_complex_constraint,
    four_g_constraint
]

params = ns.SearchParameters(  # weigh_violations_equally=True,
    constraints=constraints,
    out_directory='output/strand_displacement',
    report_only_violations=False,
)

ns.search_for_sequences(design=design, params=params)
