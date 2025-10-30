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

design = nc.Design()

def input_strand() -> nc.Strand:  #X

    #        X1               X2               X3
    # [===============--===============--===============>

    x1, x2, x3 = 'x1', 'x2', 'x3'
    s: nc.Strand = design.add_strand([x1, x2, x3], name=f'input_strand')
    for domain in s.domains:
        domain.pool = s.domains[0].pool = FULL_LONG_DOMAIN_POOL

    return s

def arbitrary_strand(name: str, domains: List[str]) -> nc.Strand: # fuel or reporter
    s: nc.Strand = design.add_strand(domain_names=domains, name=name)

    return s

def output_strand(almost_long_domain: nc.Domain) -> nc.Strand:   #Y

    #        Delta_X3          Y1              Y2              Y3
    #      <==========--===============--===============--===============]

    y1, y2, y3 = 'y1', 'y2', 'y3'
    s: nc.Strand = design.add_strand([almost_long_domain.name, y1, y2, y3], name=f'output_strand')
    for d in s.domains[1:]:
        d.state = nc.State.DEPENDENT
    # s.domains[1].pool= FULL_LONG_DOMAIN_POOL
    # s.domains[2].pool= FULL_LONG_DOMAIN_POOL
    # s.domains[3].pool = FULL_LONG_DOMAIN_POOL

    return s

def create_subdomains(domain: nc.Domain) -> None:

    #         Delta          non_Delta
    #                               clamp
    # [= = = = = = = = = = | = = = | = =>
    #                      non_clamp
    non_Delta = nc.Domain(name=f'non_Delta_{domain.name}', parents=[domain], state=nc.State.LOCKED)
    clamp = nc.Domain(name=f'clamp_{domain.name}', parents=[non_Delta], state=nc.State.LOCKED)
    non_clamp = nc.Domain(name=f'non_clamp_{domain.name}', parents=[non_Delta], state=nc.State.LOCKED)

    Delta = nc.Domain(name=f'Delta_{domain.name}', parents=[domain], state=nc.State.LOCKED)

    non_Delta.length = TOEHOLD_DOMAIN_LENGTH
    clamp.length = CLAMP_LENGTH
    non_clamp.length = TOEHOLD_DOMAIN_LENGTH - CLAMP_LENGTH
    Delta.length = ALMOST_FULL_DOMAINS

    domain.subdomains = [Delta, non_Delta]
    non_Delta.subdomains = [non_clamp, clamp]

    design.compute_derived_fields()


def dependency_function(sequence: str, rng: numpy.random.Generator = numpy.random.default_rng()) -> str:

    if sequence[2] == 'A' or sequence[2] == 'T':
        return sequence[0:2]+'C'+sequence[3:]
    elif rng.random() < 0.5:
        return sequence[0:2] + 'A' + sequence[3:]
    else:
        return sequence[0:2] + 'T' + sequence[3:]



# X
#        X1               X2               X3
# [===============--===============--===============>
X = input_strand()
for domain in X.domains:
    create_subdomains(domain)

X1, X2, X3 = X.domains[0], X.domains[1], X.domains[2]
non_Delta_X1, Delta_X1 = X1.subdomains[0], X1.subdomains[1]
non_Delta_X2, Delta_X2 = X2.subdomains[0], X2.subdomains[1]
non_Delta_X3, Delta_X3 = X3.subdomains[0], X3.subdomains[1]

# Y
#    Delta_X3        Y1               Y2               Y3
# [==========--===============--===============--===============>
Y = output_strand(Delta_X3)
for Y_domain in Y.domains[1:]:
    create_subdomains(Y_domain)
Y1, Y2, Y3 = Y.domains[1], Y.domains[2], Y.domains[3]
non_Delta_Y1, Delta_Y1 = Y1.subdomains[0], Y1.subdomains[1]
non_Delta_Y2, Delta_Y2 = Y2.subdomains[0], Y2.subdomains[1]
non_Delta_Y3, Delta_Y3 = Y3.subdomains[0], Y3.subdomains[1]

X1.dependents.append((Y1, dependency_function))
Y1.length = X1.get_length()
X2.dependents.append((Y2, dependency_function))
Y2.length = X2.get_length()
X3.dependents.append((Y3, dependency_function))
Y3.length = X3.get_length()

# F1
#          Delta_X1        X2             X3               Y1
#      <==========--===============--===============--===============]
#        X1*               X2*            X3*
#  [===============--===============--===============>
fuel_1_top = arbitrary_strand('fuel_1_top',[Delta_X1.name, X2.name, X3.name, Y1.name])
fuel_1_bottom = arbitrary_strand('fuel_1_bottom',
                                 [nc.Domain.complementary_domain_name(X1.name),
                                  nc.Domain.complementary_domain_name(X2.name),
                                  nc.Domain.complementary_domain_name(X3.name)])

# F2
#          Delta_X2         X3             Y1                 Y2
#      <==========--===============--===============--===============]
#        X2*                X3*            Y1*
#  [===============--===============--===============>
fuel_2_top = arbitrary_strand('fuel_2_top',[Delta_X2.name, X3.name, Y1.name, Y2.name])
fuel_2_bottom = arbitrary_strand('fuel_2_bottom',
                                 [nc.Domain.complementary_domain_name(X2.name),
                                  nc.Domain.complementary_domain_name(X3.name),
                                  nc.Domain.complementary_domain_name(Y1.name)])

# F3
#        Delta_X3          Y1              Y2              Y3
#      <==========--===============--===============--===============]
#        X3*                Y1*            Y2*
#  [===============--===============--===============>
fuel_3_top = arbitrary_strand('fuel_3_top',[Delta_X3.name, Y1.name, Y2.name, Y3.name])
fuel_3_bottom = arbitrary_strand('fuel_3_bottom',
                                 [nc.Domain.complementary_domain_name(X3.name),
                                  nc.Domain.complementary_domain_name(Y1.name),
                                  nc.Domain.complementary_domain_name(Y2.name)])

# reporter
#        Delta_Y1          Y2              Y3
#      <==========--===============--===============]
#        Y1*                Y2*            Y3*
#  [===============--===============--===============>
reporter_top = arbitrary_strand('reporter_top',[Delta_Y1.name, Y2.name, Y3.name])
reporter_bottom = arbitrary_strand('reporter_bottom',
                                 [nc.Domain.complementary_domain_name(Y1.name),
                                  nc.Domain.complementary_domain_name(Y2.name),
                                  nc.Domain.complementary_domain_name(Y3.name)])

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