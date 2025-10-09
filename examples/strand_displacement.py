from typing import List

import nuad.constraints as nc
import nuad.search as ns  # type: ignore

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
    s: nc.Strand = design.add_strand([almost_long_domain.name, y1, y2, y3], name=f'input_strand')
    s.domains[1].pool= FULL_LONG_DOMAIN_POOL
    s.domains[2].pool= FULL_LONG_DOMAIN_POOL
    s.domains[3].pool = FULL_LONG_DOMAIN_POOL

    return s

def create_subdomains(domain: nc.Domain) -> None:

    #   non_Delta          Delta
    #  clamp
    # [= = | = = = | = = = = = = = = = =>

    non_Delta = nc.Domain(name='non_Delta', state=nc.State.LOCKED)
    clamp = nc.Domain(name='clamp', state=nc.State.LOCKED)
    non_clamp = nc.Domain(name='non_clamp', state=nc.State.LOCKED)

    Delta = nc.Domain(name='Delta', state=nc.State.LOCKED)

    non_Delta.length = TOEHOLD_DOMAIN_LENGTH
    clamp.length = CLAMP_LENGTH
    non_clamp.length = TOEHOLD_DOMAIN_LENGTH - CLAMP_LENGTH
    Delta.length = ALMOST_FULL_DOMAINS

    domain.subdomains = [non_Delta, Delta]
    non_Delta.subdomains = [clamp, non_clamp]


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
for i in range(1,3):
    create_subdomains(Y.domains[i])
Y1, Y2, Y3 = Y.domains[1], Y.domains[2], Y.domains[3]
non_Delta_Y1, Delta_Y1 = Y1.subdomains[0], Y1.subdomains[1]
non_Delta_Y2, Delta_Y2 = Y2.subdomains[0], Y2.subdomains[1]
non_Delta_Y3, Delta_X3 = Y3.subdomains[0], Y3.subdomains[1]

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
reporter_top = arbitrary_strand('fuel_3_top',[Delta_Y1.name, Y2.name, Y3.name])
reporter_bottom = arbitrary_strand('fuel_3_bottom',
                                 [nc.Domain.complementary_domain_name(Y1.name),
                                  nc.Domain.complementary_domain_name(Y2.name),
                                  nc.Domain.complementary_domain_name(Y3.name)])

