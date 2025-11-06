from typing import List, Optional, Tuple

import nuad.constraints as nc
import nuad.search as ns  # type: ignore

import numpy as np  # noqa
import numpy.random


FULL_DOMAIN_LENGTH = 15
TOEHOLD_DOMAIN_LENGTH = 5
ALMOST_FULL_DOMAINS = FULL_DOMAIN_LENGTH-TOEHOLD_DOMAIN_LENGTH
CLAMP_LENGTH = 2

FULL_LONG_DOMAIN_POOL: nc.DomainPool = nc.DomainPool(
    'full_domain_pool',
    FULL_DOMAIN_LENGTH,
)

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


domain_to_subdomains = {}
for x in ['x3', 'x2', 'x1']:
    #         Delta         non_Delta
    # [= = = = = = = = = = | = = = = =>

    Delta = nc.Domain(name=f'Delta_{x}')
    non_Delta = nc.Domain(name=f'non_Delta_{x}')
    Delta.set_locked()
    non_Delta.set_locked()
    Delta.length = ALMOST_FULL_DOMAINS
    non_Delta.length = TOEHOLD_DOMAIN_LENGTH
    domain_to_subdomains[x] = [Delta, non_Delta]

X: nc.Strand = design.add_strand(domain_names=['x3', 'x2', 'x1'],
                                 domain_name_to_subdomains=domain_to_subdomains, name=f'input_strand')

for domain in X.domains:
    domain.pool = FULL_LONG_DOMAIN_POOL

X1, X2, X3 = X.domains[2], X.domains[1], X.domains[0]
non_Delta_X1, Delta_X1 = X1.subdomains[0], X1.subdomains[1]
non_Delta_X2, Delta_X2 = X2.subdomains[0], X2.subdomains[1]
non_Delta_X3, Delta_X3 = X3.subdomains[0], X3.subdomains[1]

# to include the subdomains (Delta and non_Delta) in the design
design.compute_derived_fields()

# output strand
# Y
#        Y3               Y2               Y1          Delta_X3
# [===============--===============--===============--==========>

Y1 = X1.create_domain_with_mismatches(name='y1', pick_dependent_seq=dependency_function)
Y2 = X2.create_domain_with_mismatches(name='y2', pick_dependent_seq=dependency_function)
Y3 = X3.create_domain_with_mismatches(name='y3', pick_dependent_seq=dependency_function)
Y: nc.Strand = design.add_strand(domains=[Y3, Y2, Y1, Delta_X3], starred_domain_indices=[], name=f'output_strand')


# Y1
#         Delta
#                               clamp
# [= = = = = = = = = = | = = = | = =>
#                          |
#                          |
#                    non_clamp_and_Delta
Delta_Y1 = nc.Domain(name=f'Delta_y1', locked=True)
clamp_Y1 = nc.Domain(name=f'clamp_y1', locked=True)
non_clamp_and_Delta_Y1 = nc.Domain(name=f'non_clamp_y1', locked=True)
clamp_Y1.length = CLAMP_LENGTH
non_clamp_and_Delta_Y1.length = TOEHOLD_DOMAIN_LENGTH - CLAMP_LENGTH
Delta_Y1.length = ALMOST_FULL_DOMAINS

Y1.subdomains = [Delta_Y1, non_clamp_and_Delta_Y1, clamp_Y1]

# Y2, Y3
#         non_clamp           clamp
# [= = = = = = = = = = = = = | = =>

for domain in [Y2, Y3]:
    clamp = nc.Domain(name=f'clamp_{domain.name}', locked=True)
    non_clamp = nc.Domain(name=f'non_clamp_{domain.name}', locked=True)
    clamp.length = CLAMP_LENGTH
    non_clamp.length = FULL_DOMAIN_LENGTH - CLAMP_LENGTH

    domain.subdomains = [non_clamp, clamp]

design.compute_derived_fields()

# F1
#           Delta_X1        X2             X3               Y1
#       <==========--===============--===============--===============]
#        X1*               X2*            X3*        clamp_Y1*
#  [===============--===============--===============--==>

starred_X1 = nc.Domain.complementary_domain_name(X1.name)
starred_X2 = nc.Domain.complementary_domain_name(X2.name)
starred_X3 = nc.Domain.complementary_domain_name(X3.name)
starred_clamp_Y1 = nc.Domain.complementary_domain_name(clamp_Y1.name)
fuel_1_top = design.add_strand(name='fuel_1_top', domain_names=[Y1.name, X3.name, X2.name, Delta_X1.name])
fuel_1_bottom = design.add_strand(name='fuel_1_bottom',
                                  domain_names=[starred_X1, starred_X2, starred_X3, starred_clamp_Y1])

# F2
#         Delta_X2         X3             Y1                 Y2
#       <==========--===============--===============--===============]
#         X2*                X3*            Y1*      clamp_Y2*
#  [===============--===============--===============--==>

starred_Y1 = nc.Domain.complementary_domain_name(Y1.name)
starred_clamp_Y2 = nc.Domain.complementary_domain_name(Y2.subdomains[1].name)
fuel_2_top = design.add_strand(name='fuel_2_top',domain_names=[Y2.name, Y1.name, X3.name, Delta_X2.name])
fuel_2_bottom = design.add_strand(name='fuel_2_bottom',
                                 domain_names=[starred_X2, starred_X3, starred_Y1, starred_clamp_Y2])

# F3
#         Delta_X3          Y1              Y2              Y3
#       <==========--===============--===============--===============]
#        X3*                Y1*            Y2*       clamp_Y3*
#  [===============--===============--===============--==>
starred_Y2 = nc.Domain.complementary_domain_name(Y2.name)
starred_clamp_Y3 = nc.Domain.complementary_domain_name(Y3.subdomains[1].name)
fuel_3_top = design.add_strand(name='fuel_3_top',domain_names=[Y3.name, Y2.name, Y1.name, Delta_X3.name])
fuel_3_bottom = design.add_strand(name='fuel_3_bottom',
                                 domain_names=[starred_X3, starred_Y1, starred_Y2, starred_clamp_Y3])

# reporter
#        Delta_Y1          Y2              Y3
#      <==========--===============--===============]
#        Y1*                Y2*            Y3*
#  [===============--===============--===============>
starred_Y3 = nc.Domain.complementary_domain_name(Y3.name)
reporter_top = design.add_strand(name='reporter_top',domain_names=[Y3.name, Y2.name,  Delta_Y1.name])
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
