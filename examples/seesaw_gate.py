import dsd.vienna_nupack as dv

# Test ComplexConstraint evaluate
import dsd.constraints as dc
import dsd.search as ds  # type: ignore

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
    dc.ForbiddenSubstringConstraint(['G'*4, 'C'*4]),
]

non_sub_long_domain_constraints = [
    dc.RestrictBasesConstraint(('A', 'C', 'T')),
    *forbidden_substring_constraints
]
sub_long_domain_constraints = [
    dc.RestrictBasesConstraint(('A', 'C', 'T')),
]

if SUB_LONG_DOMAIN_LENGTH > 3:
    sub_long_domain_constraints.extend(forbidden_substring_constraints)

SUB_LONG_DOMAIN_POOL: dc.DomainPool = dc.DomainPool('sub_long_domain_pool', SUB_LONG_DOMAIN_LENGTH, numpy_constraints=sub_long_domain_constraints)
NON_SUB_LONG_DOMAIN_POOL: dc.DomainPool = dc.DomainPool('non_sub_long_domain_pool', NON_SUB_LONG_DOMAIN_LENGTH, numpy_constraints=non_sub_long_domain_constraints)


toehold_domain_contraints = [
    dc.ForbiddenSubstringConstraint('G'*4),
    dc.ForbiddenSubstringConstraint('C'*4)
]
TOEHOLD_DOMAIN_POOL: dc.DomainPool = dc.DomainPool('toehold_domain_pool', 5)

#  s2       S2          T    s1      S1
# [==--=============--=====--==-=============>
def seesaw_signal_strand(gate1: int, gate2: int) -> dc.Strand:
    d1 = f'{SIGNAL_DOMAIN_PREFIX}{gate1}'
    d1_sub = f'{SIGNAL_DOMAIN_SUB_PREFIX}{gate1}'
    d2 = f'{SIGNAL_DOMAIN_PREFIX}{gate2}'
    d2_sub = f'{SIGNAL_DOMAIN_SUB_PREFIX}{gate2}'
    s = dc.Strand([d2_sub, d2, TOEHOLD_DOMAIN, d1_sub, d1], name=f'signal {gate1} {gate2}')
    s.domains[0].pool = SUB_LONG_DOMAIN_POOL
    s.domains[1].pool = NON_SUB_LONG_DOMAIN_POOL
    s.domains[2].pool = TOEHOLD_DOMAIN_POOL
    s.domains[3].pool = SUB_LONG_DOMAIN_POOL
    s.domains[4].pool = NON_SUB_LONG_DOMAIN_POOL

    return s

#    T*         S1*      s1*   T*
# [=====--=============--==--=====>
def gate_base_strand(gate: int) -> dc.Strand:
    d = f'{SIGNAL_DOMAIN_PREFIX}{gate}{COMPLEMENT_SUFFIX}'
    d_sub = f'{SIGNAL_DOMAIN_SUB_PREFIX}{gate}{COMPLEMENT_SUFFIX}'
    s: dc.Strand = dc.Strand(
        [TOEHOLD_COMPLEMENT, d, d_sub, TOEHOLD_COMPLEMENT], name=f'gate {gate}')
    s.domains[0].pool = TOEHOLD_DOMAIN_POOL
    s.domains[1].pool = NON_SUB_LONG_DOMAIN_POOL
    s.domains[2].pool = SUB_LONG_DOMAIN_POOL
    s.domains[3].pool = TOEHOLD_DOMAIN_POOL
    return s

#  s1      S1
# [==--=============>
def waste_strand(gate: int) -> dc.Strand:
    d = f'{SIGNAL_DOMAIN_PREFIX}{gate}'
    d_sub = f'{SIGNAL_DOMAIN_SUB_PREFIX}{gate}'
    s = dc.Strand([d_sub, d], name=f'waste {gate}')
    s.domains[0].pool = SUB_LONG_DOMAIN_POOL
    s.domains[1].pool = NON_SUB_LONG_DOMAIN_POOL
    return s

#  s1*   T*        S2*       s2*
# [==--=====--=============--==>
def threshold_base_strand(gate1: int, gate2: int) -> dc.Strand:
    # TODO: account for subdomain for gate1
    d1_sub = f'{SIGNAL_DOMAIN_SUB_PREFIX}{gate1}{COMPLEMENT_SUFFIX}'

    d2 = f'{SIGNAL_DOMAIN_PREFIX}{gate2}{COMPLEMENT_SUFFIX}'
    d2_sub = f'{SIGNAL_DOMAIN_SUB_PREFIX}{gate2}{COMPLEMENT_SUFFIX}'


    s: dc.Strand = dc.Strand(
        [d1_sub, TOEHOLD_COMPLEMENT, d2, d2_sub], name=f'threshold {gate1} {gate2}')
    s.domains[0].pool = SUB_LONG_DOMAIN_POOL

    s.domains[1].pool = TOEHOLD_DOMAIN_POOL

    s.domains[2].pool = NON_SUB_LONG_DOMAIN_POOL
    s.domains[3].pool = SUB_LONG_DOMAIN_POOL

    return s

#    T*        S1*       s1*
# [=====--=============--==>
def reporter_base_strand(gate) -> dc.Strand:
    d = f'{SIGNAL_DOMAIN_PREFIX}{gate}{COMPLEMENT_SUFFIX}'
    d_sub = f'{SIGNAL_DOMAIN_SUB_PREFIX}{gate}{COMPLEMENT_SUFFIX}'

    s: dc.Strand = dc.Strand(
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
#       S2        s2    T          S5       s5
# <=============-===--=====--=============--==]
signal_2_5_strand = seesaw_signal_strand(2, 5)
signal_2_5_toehold_addr = signal_2_5_strand.address_of_first_domain_occurence(TOEHOLD_DOMAIN)

#  s7       S7          T    s5      S5
# [==--=============--=====--==-=============>
#
#       S5        s5    T          S7       s7
# <=============-===--=====--=============--==]
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
threshold_2_5_base_strand = threshold_base_strand(2,5)

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
#                 S5         s5    T          S6       s6
#                            21                         1
#             34          22 |20 19  15 14          2  |0
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
#                 S5         s5    T          S6       s6
#             DANGLE_5P      I   I
#             |              |   |
#            <=============--==--=====--=============--==]
#             |||||||||||||  ||  |||||
#     [=====--=============--==--=====>
#                         |   |      |
#                         I   I      DANGLE_5P
#        T*         S5*      s5*   T*
g_5_s_5_6_complex = (signal_5_6_strand, gate_5_base_strand)
g_5_s_5_6_nonimplicit_base_pairs = [(signal_5_6_toehold_addr, gate_5_bound_toehold_3p_addr)]
g_5_s_5_6_complex_constraint = dc.nupack_4_complex_secondary_structure_constraint(
    strand_complexes=[g_5_s_5_6_complex],
    nonimplicit_base_pairs=g_5_s_5_6_nonimplicit_base_pairs,
    # base_pair_probs={
    #     dc.BasePairType.DANGLE_5P: 0.4
    # }
)

#        S2           T           S5
#  34            20 19  15 14            0
#  |             |  |   |  |             |
# <===============--=====--===============]
#                   |||||  |||||||||||||||
#                  [=====--===============--=====>
#                   |   |  |             |  |   |
#                   35  39 40            54 55  59
#                     T*         S5*          T*
#
# Debugging base pair types:
#
#                         INTERIOR_TO_STRAND
#           DANGLE_3P      |
#                   |      |
#                   |      |
#        S2         | T    |      S5
# <===============--=====--===============]
#                   |||||  |||||||||||||||
#                  [=====--===============--=====>
#                       |                |
#                     T*|        S5*     |    T*
#                       |                |
#                       |                |
#      INTERIOR_TO_STRAND                DANGLE_3P
#
#
g_5_s_2_5_complex = (signal_2_5_strand, gate_5_base_strand)
g_5_s_2_5_nonimplicit_base_pairs = [(signal_2_5_toehold_addr, gate_5_bound_toehold_5p_addr)]
g_5_s_2_5_complex_constraint = dc.nupack_4_complex_secondary_structure_constraint(
    strand_complexes=[g_5_s_2_5_complex],
    nonimplicit_base_pairs=g_5_s_2_5_nonimplicit_base_pairs,
    # base_pair_probs={
    #     dc.BasePairType.DANGLE_5P: 0.4
    # }
)

#               S5           T           S7
#         34            20 19  15 14            0
#         |             |  |   |  |             |
#        <===============--=====--===============]
#         |||||||||||||||  |||||
# [=====--===============--=====>
#  |   |  |             |  |   |
#  35  39 40            54 55  59
#    T*         S5*          T*
#
# Debugging base pair types:
#
#      DANGLE_5P           INTERIOR_TO_STRAND
#         |                |
#         |                |
#         |                |
#         |     S5         | T           S7
#        <===============--=====--===============]
#         |||||||||||||||  |||||
# [=====--===============--=====>
#    T*         S5*     |    T*|
#                       |      |
#                       |      |
#                       |      |
#      INTERIOR_TO_STRAND      DANGLE_5P
g_5_s_5_7_complex = (signal_5_7_strand, gate_5_base_strand)
g_5_s_5_7_nonimplicit_base_pairs = [(signal_5_7_toehold_addr, gate_5_bound_toehold_3p_addr)]
g_5_s_5_7_complex_constraint = dc.nupack_4_complex_secondary_structure_constraint(
    strand_complexes=[g_5_s_5_7_complex],
    nonimplicit_base_pairs=g_5_s_5_7_nonimplicit_base_pairs,
    # base_pair_probs={
    #     dc.BasePairType.DANGLE_5P: 0.4
    # }
)

#        S5
# [===============>
waste_5_strand = waste_strand(5)

#                                S5
#                          14            0
#                          |             |
#                         <===============]
#                          |||||||||||||||
# [===============--=====--===============>
#  |             |  |   |  |             |
#  15           29  30  34 35            49
#        S2*         T*          S5*
#
# Debugging base pair types:
#                          DANGLE_5P
#                          |     S5
#                         <===============]
#                          |||||||||||||||
# [===============--=====--===============>
#        S2*         T*          S5*     |
#                                        BLUNT_END
t_2_5_w_5_complex = (waste_5_strand, threshold_2_5_base_strand)
t_2_5_w_5_complex_constraint = dc.nupack_4_complex_secondary_structure_constraint(strand_complexes=[t_2_5_w_5_complex])

#        S2           T           S5
#  34            20 19  15 14            0
#  |             |  |   |  |             |
# <===============--=====--===============]
#  |||||||||||||||  |||||  |||||||||||||||
# [===============--=====--===============>
#  |             |  |   |  |             |
#  35           49  50  54 55            69
#        S2*         T*          S5*
#
# Debugging base pair types:
#
#  BLUNT_END   INTERIOR_TO_STRAND
#  |     S2         | T    |     S5
# <===============--=====--===============]
#  |||||||||||||||  |||||  |||||||||||||||
# [===============--=====--===============>
#        S2*     |    T*|        S5*     |
#           INTERIOR_TO_STRAND           BLUNT_END
waste_2_5_complex = (signal_2_5_strand, threshold_2_5_base_strand)
waste_2_5_complex_constraint = dc.nupack_4_complex_secondary_structure_constraint(strand_complexes=[waste_2_5_complex])

#                                        S6
#                                 14            0
#                                 |             |
#                                <===============]
#                                 |||||||||||||||
#                         [=====--===============>
#                          |   |  |             |
#                          15  19 20            34
#                            T*           S6*
#
# Base Pair Types:
#                                 DANGLE_5P
#                                 |
#                                 |      S6
#                                <===============]
#                                 |||||||||||||||
#                         [=====--===============>
#                            T*           S6*   |
#                                               |
#                                               BLUNT_END
reporter_6_complex = (waste_6_strand, reporter_6_base_strand)
reporter_6_complex_constraint = dc.nupack_4_complex_secondary_structure_constraint(strand_complexes=[reporter_6_complex])

#               S5           T           S6
#         34            20 19  15 14            0
#         |             |  |   |  |             |
#        <===============--=====--===============]
#                          |||||  |||||||||||||||
#                         [=====--===============>
#                          |   |  |             |
#                          35  39 40            54
#                            T*           S6*
#
# Base Pair Types:
#                  DANGLE_3P      INTERIOR_TO_STRAND
#                          |      |
#               S5         | T    |      S6
#        <===============--=====--===============]
#                          |||||  |||||||||||||||
#                         [=====--===============>
#                            T*|          S6*   |
#                              |                |
#             INTERIOR_TO_STRAND                BLUNT_END
f_waste_6_complex = (signal_5_6_strand, reporter_6_base_strand)
f_waste_6_complex_constraint = dc.nupack_4_complex_secondary_structure_constraint(strand_complexes=[f_waste_6_complex])

# TODO: Strand Constraint check for 4 G's (don't need to check for 4 C's)


# Constraints
complex_constraints = [
    g_5_s_5_6_complex_constraint,
    g_5_s_2_5_complex_constraint,
    g_5_s_5_7_complex_constraint,
    t_2_5_w_5_complex_constraint,
    waste_2_5_complex_constraint,
    reporter_6_complex_constraint,
    f_waste_6_complex_constraint,
]

seesaw_design = dc.Design(strands=strands, complex_constraints=complex_constraints)

ds.search_for_dna_sequences(design=seesaw_design,
                            # weigh_violations_equally=True,
                            report_delay=0.0,
                            out_directory='output/seesaw_gate',
                            report_only_violations=False,
                            )
