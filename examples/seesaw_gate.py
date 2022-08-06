from typing import List, Optional, Tuple

# Test ComplexConstraint evaluate
import nuad.constraints as nc
import nuad.search as ns  # type: ignore

LONG_DOMAIN_LENGTH = 15
SUB_LONG_DOMAIN_LENGTH = 2
NON_SUB_LONG_DOMAIN_LENGTH = LONG_DOMAIN_LENGTH - SUB_LONG_DOMAIN_LENGTH

TOEHOLD_LENGTH = 5

SIGNAL_DOMAIN_PREFIX = 'S'
SIGNAL_DOMAIN_SUB_PREFIX = 's'
TOEHOLD_DOMAIN = 'T'
COMPLEMENT_SUFFIX = '*'
TOEHOLD_COMPLEMENT = f'{TOEHOLD_DOMAIN}{COMPLEMENT_SUFFIX}'

# Domain pools
forbidden_substring_constraints = [
    nc.ForbiddenSubstringConstraint(['G' * 4, 'C' * 4]),
]

non_sub_long_domain_constraints = [
    nc.RestrictBasesConstraint(('A', 'C', 'T')),
    *forbidden_substring_constraints
]
sub_long_domain_constraints: List[nc.NumpyConstraint] = [
    nc.RestrictBasesConstraint(('A', 'C', 'T')),
]

if SUB_LONG_DOMAIN_LENGTH > 3:
    sub_long_domain_constraints.extend(forbidden_substring_constraints)

SUB_LONG_DOMAIN_POOL: nc.DomainPool = nc.DomainPool('sub_long_domain_pool', SUB_LONG_DOMAIN_LENGTH,
                                                    numpy_constraints=sub_long_domain_constraints)
NON_SUB_LONG_DOMAIN_POOL: nc.DomainPool = nc.DomainPool('non_sub_long_domain_pool',
                                                        NON_SUB_LONG_DOMAIN_LENGTH,
                                                        numpy_constraints=non_sub_long_domain_constraints)

toehold_domain_contraints = [
    nc.ForbiddenSubstringConstraint('G' * 4),
    nc.ForbiddenSubstringConstraint('C' * 4)
]
TOEHOLD_DOMAIN_POOL: nc.DomainPool = nc.DomainPool('toehold_domain_pool', 5)

design = nc.Design()


#  s2       S2          T    s1      S1
# [==--=============--=====--==-=============>
def seesaw_signal_strand(gate1: int, gate2: int) -> nc.Strand:
    d1 = f'{SIGNAL_DOMAIN_PREFIX}{gate1}'
    d1_sub = f'{SIGNAL_DOMAIN_SUB_PREFIX}{gate1}'
    d2 = f'{SIGNAL_DOMAIN_PREFIX}{gate2}'
    d2_sub = f'{SIGNAL_DOMAIN_SUB_PREFIX}{gate2}'
    s: nc.Strand = design.add_strand([d2_sub, d2, TOEHOLD_DOMAIN, d1_sub, d1], name=f'signal {gate1} {gate2}')
    s.domains[0].pool = SUB_LONG_DOMAIN_POOL
    s.domains[1].pool = NON_SUB_LONG_DOMAIN_POOL
    s.domains[2].pool = TOEHOLD_DOMAIN_POOL
    s.domains[3].pool = SUB_LONG_DOMAIN_POOL
    s.domains[4].pool = NON_SUB_LONG_DOMAIN_POOL

    return s


#    T*         S1*      s1*   T*
# [=====--=============--==--=====>
def gate_base_strand(gate: int) -> nc.Strand:
    d = f'{SIGNAL_DOMAIN_PREFIX}{gate}{COMPLEMENT_SUFFIX}'
    d_sub = f'{SIGNAL_DOMAIN_SUB_PREFIX}{gate}{COMPLEMENT_SUFFIX}'
    s: nc.Strand = design.add_strand(
        [TOEHOLD_COMPLEMENT, d, d_sub, TOEHOLD_COMPLEMENT], name=f'gate {gate}')
    s.domains[0].pool = TOEHOLD_DOMAIN_POOL
    s.domains[1].pool = NON_SUB_LONG_DOMAIN_POOL
    s.domains[2].pool = SUB_LONG_DOMAIN_POOL
    s.domains[3].pool = TOEHOLD_DOMAIN_POOL
    return s


#  s1      S1
# [==--=============>
def waste_strand(gate: int) -> nc.Strand:
    d = f'{SIGNAL_DOMAIN_PREFIX}{gate}'
    d_sub = f'{SIGNAL_DOMAIN_SUB_PREFIX}{gate}'
    s: nc.Strand = design.add_strand([d_sub, d], name=f'waste {gate}')
    s.domains[0].pool = SUB_LONG_DOMAIN_POOL
    s.domains[1].pool = NON_SUB_LONG_DOMAIN_POOL
    return s


#  s1*   T*        S2*       s2*
# [==--=====--=============--==>
def threshold_base_strand(gate1: int, gate2: int) -> nc.Strand:
    d1_sub = f'{SIGNAL_DOMAIN_SUB_PREFIX}{gate1}{COMPLEMENT_SUFFIX}'

    d2 = f'{SIGNAL_DOMAIN_PREFIX}{gate2}{COMPLEMENT_SUFFIX}'
    d2_sub = f'{SIGNAL_DOMAIN_SUB_PREFIX}{gate2}{COMPLEMENT_SUFFIX}'

    s: nc.Strand = design.add_strand(
        [d1_sub, TOEHOLD_COMPLEMENT, d2, d2_sub], name=f'threshold {gate1} {gate2}')
    s.domains[0].pool = SUB_LONG_DOMAIN_POOL

    s.domains[1].pool = TOEHOLD_DOMAIN_POOL

    s.domains[2].pool = NON_SUB_LONG_DOMAIN_POOL
    s.domains[3].pool = SUB_LONG_DOMAIN_POOL

    return s


#    T*        S1*       s1*
# [=====--=============--==>
def reporter_base_strand(gate) -> nc.Strand:
    d = f'{SIGNAL_DOMAIN_PREFIX}{gate}{COMPLEMENT_SUFFIX}'
    d_sub = f'{SIGNAL_DOMAIN_SUB_PREFIX}{gate}{COMPLEMENT_SUFFIX}'

    s: nc.Strand = design.add_strand(
        [TOEHOLD_COMPLEMENT, d, d_sub], name=f'reporter {gate}')
    s.domains[0].pool = TOEHOLD_DOMAIN_POOL
    s.domains[1].pool = NON_SUB_LONG_DOMAIN_POOL
    s.domains[2].pool = SUB_LONG_DOMAIN_POOL
    return s


# Signals

#  s6       S6          T     s5      S5
# [==--=============--=====--===-=============>
#
#       S5        s5    T          S6       s6
# <=============-===--=====--=============--==]
signal_5_6_strand = seesaw_signal_strand(5, 6)
signal_5_6_toehold_addr = signal_5_6_strand.address_of_first_domain_occurence(TOEHOLD_DOMAIN)

#  s5       S5          T    s2      S2
# [==--=============--=====--==-=============>
#
#       S2       s2    T          S5       s5
# <=============-==--=====--=============--==]
signal_2_5_strand = seesaw_signal_strand(2, 5)
signal_2_5_toehold_addr = signal_2_5_strand.address_of_first_domain_occurence(TOEHOLD_DOMAIN)

#  s7       S7          T    s5      S5
# [==--=============--=====--==-=============>
#
#       S5       s5    T          S7       s7
# <=============-==--=====--=============--==]
signal_5_7_strand = seesaw_signal_strand(5, 7)
signal_5_7_toehold_addr = signal_5_7_strand.address_of_first_domain_occurence(TOEHOLD_DOMAIN)

# Gate Bases
#    T*         S5*      s5*   T*
# [=====--=============--==--=====>
#
#    T*   s5*      S5*         T*
# <=====--==--=============--=====]
gate_5_base_strand = gate_base_strand(5)
gate_5_bound_toehold_3p_addr = gate_5_base_strand.address_of_last_domain_occurence(TOEHOLD_COMPLEMENT)
gate_5_bound_toehold_5p_addr = gate_5_base_strand.address_of_first_domain_occurence(TOEHOLD_COMPLEMENT)

# Waste Strands
#  s5      S5
# [==--=============>
#
#        S5       s5
# <=============--==]
waste_5_strand = waste_strand(5)
waste_6_strand = waste_strand(6)

# Threshold Base Strands
#  s2*   T*        S5*       s5*
# [==--=====--=============--==>
threshold_2_5_base_strand = threshold_base_strand(2, 5)

# Reporter Base Strands
#    T*        S6*       s6*
# [=====--=============--==>
reporter_6_base_strand = reporter_base_strand(6)

# Collect all strands
strands = [
    # signals
    signal_5_6_strand,
    signal_2_5_strand,
    signal_5_7_strand,
    # gate
    gate_5_base_strand,
    # waste
    waste_5_strand,
    waste_6_strand,
    # threshold
    threshold_2_5_base_strand,
    # reporter
    reporter_6_base_strand,
]

# Complexes (with NUPACK indexing)
#
#
#                 S5         s5    T        S6 / S7  s6 / s7
#                            21
#             34          22 |20 19  15 14          2  10
#             |           |  ||  |   |  |           |  ||
#            <=============--==--=====--=============--==]
#             |||||||||||||  ||  |||||
#     [=====--=============--==--=====>
#      |   |  |           |  ||  |   |
#      35  39 40          52 |54 55  59
#                            53
#        T*         S5*      s5*   T*
#
# Debugging base pair types:
#
#                 S5         s5    T       S6 / S7    s6 / s7
#             DANGLE_5P   INTERIOR_TO_STRAND
#             |              |   |
#            <=============--==--=====--=============--==]
#             |||||||||||||  ||  |||||
#     [=====--=============--==--=====>
#                         |   |      |
#                INTERIOR_TO_STRAND  DANGLE_5P
#        T*         S5*      s5*   T*
g_5_s_5_6_complex = nc.Complex(signal_5_6_strand, gate_5_base_strand)
g_5_s_5_7_complex = nc.Complex(signal_5_7_strand, gate_5_base_strand)

g_5_s_5_6_nonimplicit_base_pairs = [(signal_5_6_toehold_addr, gate_5_bound_toehold_3p_addr)]
g_5_s_5_6_complex_constraint = nc.nupack_complex_base_pair_probability_constraint(
    strand_complexes=[g_5_s_5_6_complex, g_5_s_5_7_complex],
    nonimplicit_base_pairs=g_5_s_5_6_nonimplicit_base_pairs,
    # base_pair_probs={
    #     dc.BasePairType.DANGLE_5P: 0.4
    # }
)

#       S2        s2    T          S5       s5
#                21
#  34          22|20 19  15 14          2  10
#  |           | ||  |   |  |           |  ||
# <=============-==--=====--=============--==]
#                    |||||  |||||||||||||  ||
#                   [=====--=============--==--=====>
#                    |   |  |           |  ||  |   |
#                    35  39 40          52 |54 55  59
#                                          53
#                      T*         S5*      s5*   T*
#
# Debugging base pair types:
#
#
#      S2        s2    T          S5       s5
#            DANGLE_3P     INTERIOR_TO_STRAND
#                    |      |              |
# <=============-==--=====--=============--==]
#                    |||||  |||||||||||||  ||
#                   [=====--=============--==--=====>
#                        |              |   |
#                       INTERIOR_TO_STRAND  DANGLE_3P
#                      T*         S5*      s5*   T*
g_5_s_2_5_complex = nc.Complex(signal_2_5_strand, gate_5_base_strand)
g_5_s_2_5_nonimplicit_base_pairs = [(signal_2_5_toehold_addr, gate_5_bound_toehold_5p_addr)]
g_5_s_2_5_complex_constraint = nc.nupack_complex_base_pair_probability_constraint(
    strand_complexes=[g_5_s_2_5_complex],
    nonimplicit_base_pairs=g_5_s_2_5_nonimplicit_base_pairs,
    # base_pair_probs={
    #     dc.BasePairType.DANGLE_5P: 0.4
    # }
)

#        S5
# [===============>
waste_5_strand = waste_strand(5)

#                   S5       s5
#             14          2  10
#             |           |  ||
#            <=============--==]
#             |||||||||||||  ||
# [==--=====--=============--==>
#  ||  |   |  |           |  ||
# 15|  17  21 22          34 |36
#   16                      35
#  s2*   T*        S5*       s5*
#
# Debugging base pair types:
#
#
#                   S5       s5
#     DANGLE_5P              INTERIOR_TO_STRAND
#             |              |
#            <=============--==]
#             |||||||||||||  ||
# [==--=====--=============--==>
#                         |   |
#        INTERIOR_TO_STRAND   BLUNT_END
#  s2*   T*        S5*       s5*
t_2_5_w_5_complex = nc.Complex(waste_5_strand, threshold_2_5_base_strand)
t_2_5_w_5_complex_constraint = nc.nupack_complex_base_pair_probability_constraint(
    strand_complexes=[t_2_5_w_5_complex])

#
#      S2        s2    T          S5       s5
#
#                21
#  34          22|20 19  15 14          2  10
#  |           | ||  |   |  |           |  ||
# <=============-==--=====--=============--==]
#                ||  |||||  |||||||||||||  ||
#               [==--=====--=============--==>
#                ||  |   |  |           |  ||
#               35|  37  41 42          54 |56
#                 36                       55
#
#                s2*   T*        S5*       s5*
#
# Debugging base pair types:
#
#      S2        s2    T          S5       s5
#        DANGLE_3P INTERIOR_TO_STRAND      INTERIOR_TO_STRAND
#                |   |      |              |
# <=============-==--=====--=============--==]
#                ||  |||||  |||||||||||||  ||
#               [==--=====--=============--==>
#                 |      |              |   |
# INTERIOR_TO_STRAND     INTERIOR_TO_STRAND BLUNT_END
#                s2*   T*        S5*       s5*

waste_2_5_complex = nc.Complex(signal_2_5_strand, threshold_2_5_base_strand)
waste_2_5_complex_constraint = nc.nupack_complex_base_pair_probability_constraint(
    strand_complexes=[waste_2_5_complex])

#               S6       s6
#         14          2  10
#         |           |  ||
#        <=============--==]
#         |||||||||||||  ||
# [=====--=============--==>
#  |   |  |           |  ||
#  15  19 20          32 |34
#                        33
#    T*        S6*       s6*
#
# Base Pair Types:
#
#               S6       s6
# DANGLE_5P              INTERIOR_TO_STRAND
#         |              |
#        <=============--==]
#         |||||||||||||  ||
# [=====--=============--==>
#                     |   |
#    INTERIOR_TO_STRAND   BLUNT_END
#    T*        S6*       s6*
reporter_6_complex = nc.Complex(waste_6_strand, reporter_6_base_strand)
reporter_6_complex_constraint = nc.nupack_complex_base_pair_probability_constraint(
    strand_complexes=[reporter_6_complex])

#       S5        s5    T          S6       s6
#                 21
#  34          22 |20 19  15 14          2  10
#  |           |  ||  |   |  |           |  ||
# <=============--==--=====--=============--==]
#                     |||||  |||||||||||||  ||
#                    [=====--=============--==>
#                     |   |  |           |  ||
#                     35  39 40          52 |54
#                                           53
#                       T*        S6*       s6*
#
# Base Pair Types:
#
#       S5        s5    T          S6       s6
#             DANGLE_3P     INTERIOR_TO_STRAND
#                     |      |              |
# <=============--==--=====--=============--==]
#                     |||||  |||||||||||||  ||
#                    [=====--=============--==>
#                         |              |   |
#                        INTERIOR_TO_STRAND  BLUNT_END
#                       T*        S6*       s6*
f_waste_6_complex = nc.Complex(signal_5_6_strand, reporter_6_base_strand)
f_waste_6_complex_constraint = nc.nupack_complex_base_pair_probability_constraint(
    strand_complexes=[f_waste_6_complex])


def four_g_constraint_evaluate(seqs: Tuple[str, ...], strand: Optional[nc.Strand]) -> Tuple[float, str]:
    seq = seqs[0]
    score = 1000 if 'GGGG' in seq else 0
    violation_str = "" if 'GGGG' not in strand.sequence() else "** violation**"
    return score, f"{strand.name}: {strand.sequence()}{violation_str}"


def four_g_constraint_summary(strand: nc.Strand):
    violation_str = "" if 'GGGG' not in strand.sequence() else "** violation**"
    return f"{strand.name}: {strand.sequence()}{violation_str}"


four_g_constraint = nc.StrandConstraint(description="4GConstraint",
                                        short_description="4GConstraint",
                                        evaluate=four_g_constraint_evaluate,
                                        strands=tuple(strands), )

# Constraints
constraints = [
    g_5_s_5_6_complex_constraint,
    g_5_s_2_5_complex_constraint,
    t_2_5_w_5_complex_constraint,
    waste_2_5_complex_constraint,
    reporter_6_complex_constraint,
    f_waste_6_complex_constraint,
]
constraints.append(four_g_constraint)

seesaw_design = nc.Design(strands=strands)

params = ns.SearchParameters(  # weigh_violations_equally=True,
    constraints=constraints,
    # report_delay=0.0,
    out_directory='output/seesaw_gate',
    report_only_violations=False, )

ns.search_for_sequences(design=seesaw_design, params=params)
