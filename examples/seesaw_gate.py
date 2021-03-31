import dsd.vienna_nupack as dv

# Test ComplexConstraint evaluate
import dsd.constraints as dc
import dsd.search as ds  # type: ignore

LONG_DOMAIN_LENGTH = 15
TOEHOLD_LENGTH = 5

SIGNAL_DOMAIN_PREFIX = 'S'
TOEHOLD_DOMAIN = 'T'
COMPLEMENT_SUFFIX = '*'
TOEHOLD_COMPLEMENT = f'{TOEHOLD_DOMAIN}{COMPLEMENT_SUFFIX}'

LONG_DOMAIN_POOL: dc.DomainPool = dc.DomainPool('long_domain_pool', 15)
TOEHOLD_DOMAIN_POOL: dc.DomainPool = dc.DomainPool('toehold_domain_pool', 5)


def seesaw_signal_strand(gate1: int, gate2: int) -> dc.Strand:
    d1 = f'{SIGNAL_DOMAIN_PREFIX}{gate1}'
    d2 = f'{SIGNAL_DOMAIN_PREFIX}{gate2}'
    s = dc.Strand([d2, TOEHOLD_DOMAIN, d1], name=f'signal {gate1} {gate2}')
    s.domains[0].pool = LONG_DOMAIN_POOL
    s.domains[1].pool = TOEHOLD_DOMAIN_POOL
    s.domains[2].pool = LONG_DOMAIN_POOL
    return s


def gate_base_strand(gate: int) -> dc.Strand:
    d = f'{SIGNAL_DOMAIN_PREFIX}{gate}{COMPLEMENT_SUFFIX}'
    s: dc.Strand = dc.Strand(
        [TOEHOLD_COMPLEMENT, d, TOEHOLD_COMPLEMENT], name=f'gate {gate}')
    s.domains[0].pool = TOEHOLD_DOMAIN_POOL
    s.domains[1].pool = LONG_DOMAIN_POOL
    s.domains[2].pool = TOEHOLD_DOMAIN_POOL
    return s


def waste_strand(gate: int) -> dc.Strand:
    d = f'{SIGNAL_DOMAIN_PREFIX}{gate}'
    s = dc.Strand([d], name=f'waste {gate}')
    s.domains[0].pool = LONG_DOMAIN_POOL
    return s

def threshold_base_strand(gate1: int, gate2: int) -> dc.Strand:
    # TODO: account for subdomain for gate1
    d1 = f'{SIGNAL_DOMAIN_PREFIX}{gate1}{COMPLEMENT_SUFFIX}'
    d2 = f'{SIGNAL_DOMAIN_PREFIX}{gate2}{COMPLEMENT_SUFFIX}'

    s: dc.Strand = dc.Strand(
        [d1, TOEHOLD_COMPLEMENT, d2], name=f'threshold {gate1} {gate2}')
    s.domains[0].pool = LONG_DOMAIN_POOL
    s.domains[1].pool = TOEHOLD_DOMAIN_POOL
    s.domains[2].pool = LONG_DOMAIN_POOL
    return s

def reporter_base_strand(gate) -> dc.Strand:
    d = f'{SIGNAL_DOMAIN_PREFIX}{gate}{COMPLEMENT_SUFFIX}'
    s: dc.Strand = dc.Strand(
        [TOEHOLD_COMPLEMENT, d], name=f'reporter {gate}')
    s.domains[0].pool = TOEHOLD_DOMAIN_POOL
    s.domains[1].pool = LONG_DOMAIN_POOL
    return s

# Comments show resulting sequence (6862 iterations).

# Signals

#         S6          T           S5
#  AGTGGGTGGTTTTAT  GGAGC  ATCCTGGTCTGGGCT
# [===============--=====--===============>
signal_5_6_strand = seesaw_signal_strand(5, 6)
# TODO: make this a method of Strand instead of a StrandDomainAddress method
signal_5_6_toehold_addr = dc.StrandDomainAddress.address_of_first_domain_occurence(signal_5_6_strand, TOEHOLD_DOMAIN)

#         S5          T           S2
# [===============--=====--===============>
signal_2_5_strand = seesaw_signal_strand(2, 5)
signal_2_5_toehold_addr = dc.StrandDomainAddress.address_of_first_domain_occurence(signal_2_5_strand, TOEHOLD_DOMAIN)

#         S7          T           S5
# [===============--=====--===============>
signal_5_7_strand = seesaw_signal_strand(5, 7)
signal_5_7_toehold_addr = dc.StrandDomainAddress.address_of_first_domain_occurence(signal_5_7_strand, TOEHOLD_DOMAIN)


# Gate Bases
#    T*         S5*          T*
#  GCTCC  AGCCCAGACCAGGAT  GCTCC
# [=====--===============--=====>
gate_5_base_strand = gate_base_strand(5)
gate_5_bound_toehold_3p_addr = dc.StrandDomainAddress.address_of_last_domain_occurence(gate_5_base_strand, TOEHOLD_COMPLEMENT)
gate_5_bound_toehold_5p_addr = dc.StrandDomainAddress.address_of_first_domain_occurence(gate_5_base_strand, TOEHOLD_COMPLEMENT)

# Waste Strands
#        S5
# [===============>
waste_5_strand = waste_strand(5)
waste_6_strand = waste_strand(6)

# Threshold Base Strands
#         S2*         T*           S5*
# [===============--=====--===============>
threshold_2_5_base_strand = threshold_base_strand(2,5)

# Reporter Base Strands
#    T*           S6*
# [=====--===============>
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
#               S5           T           S6
#         34            20 19  15 14            0
#         |             |  |   |  |             |
#         GAACCGAGCCGAGGG  CCCGA  ACCCACCTTCACTCT
#        <===============--=====--===============]
#         |||||||||||||||  |||||
# [=====--===============--=====>
#  GGGCT  CTTGGCTCGGCTCCC  GGGCT
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
#         |     S5         | T           S6
#        <===============--=====--===============]
#         |||||||||||||||  |||||
# [=====--===============--=====>
#    T*         S5*     |    T*|
#                       |      |
#                       |      |
#                       |      |
#      INTERIOR_TO_STRAND      DANGLE_5P
g_5_s_5_6_complex = (signal_5_6_strand, gate_5_base_strand)
g_5_s_5_6_nonimplicit_base_pairs = [(signal_5_6_toehold_addr, gate_5_bound_toehold_3p_addr)]
g_5_s_5_6_complex_constraint = dc.nupack_4_complex_secondary_structure_constraint(
    strand_complex=g_5_s_5_6_complex,
    nonimplicit_base_pairs=g_5_s_5_6_nonimplicit_base_pairs,
    # base_pair_probs={
    #     dc.BasePairType.DANGLE_5P: 0.4
    # }
)

#        S2           T           S5
#  34            20 19  15 14            0
#  |             |  |   |  |             |
#  TTTGGTGGGTTGTTT  CCCGA  GAACCGAGCCGAGGG'
# <===============--=====--===============]
#                   |||||  |||||||||||||||
#                  [=====--===============--=====>
#                   GGGCT  CTTGGCTCGGCTCCC  GGGCT
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
    strand_complex=g_5_s_2_5_complex,
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
    strand_complex=g_5_s_5_7_complex,
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
t_2_5_w_5_complex_constraint = dc.nupack_4_complex_secondary_structure_constraint(strand_complex=t_2_5_w_5_complex)

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
waste_2_5_complex_constraint = dc.nupack_4_complex_secondary_structure_constraint(strand_complex=waste_2_5_complex)

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
reporter_6_complex_constraint = dc.nupack_4_complex_secondary_structure_constraint(strand_complex=reporter_6_complex)

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
f_waste_6_complex_constraint = dc.nupack_4_complex_secondary_structure_constraint(strand_complex=f_waste_6_complex)



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
