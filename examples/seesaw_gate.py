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
    s = dc.Strand([d1, TOEHOLD_DOMAIN, d2], name=f'signal {gate1},{gate2}')
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


# Signals

#         S6          T           S5
# [===============--=====--===============>
signal_6_5_strand = seesaw_signal_strand(6, 5)
signal_toehold_addr = dc.StrandDomainAddress.address_of_first_domain_occurence(signal_6_5_strand, TOEHOLD_DOMAIN)

# Gate Bases
#    T*         S5*          T*
# [=====--===============--=====>
gate_5_base_strand = gate_base_strand(5)
gate_5_bound_toehold_3p_addr = dc.StrandDomainAddress.address_of_last_domain_occurence(gate_5_base_strand, TOEHOLD_COMPLEMENT)
gate_5_bound_toehold_5p_addr = dc.StrandDomainAddress.address_of_first_domain_occurence(gate_5_base_strand, TOEHOLD_COMPLEMENT)

# Collect all strands
strands = [
    signal_6_5_strand,
    gate_5_base_strand
]

# Complexes (with NUPACK indexing)
#
#               S5           T           S6
#         34            20 19  15 14            0
#         |             |  |   |  |             |
#        <===============--=====--===============]
#         |||||||||||||||  |||||
# [=====--===============--=====>
#  |   |  |             |  |   |
#  35  39 40            54 55  59
#    T*         S5*          T*


#                                S5
# [===============--=====--===============>
#                   =====--===============--=====
#                                S5*

#    0          1           2
#             S5*
# [=====--===============--=====>
#        <===============--=====--===============]
#                2           1            0
#             S5
#
g_5_s_5_6_complex = (signal_6_5_strand, gate_5_base_strand)
g_5_s_5_6_nonimplicit_base_pairs = [(signal_toehold_addr, gate_5_bound_toehold_3p_addr)]
g_5_s_5_6_complex_constraint = dc.nupack_4_complex_secondary_structure_constraint(
    complexes=[g_5_s_5_6_complex],
    nonimplicit_base_pairs=g_5_s_5_6_nonimplicit_base_pairs
)

# Constraints
complex_constraints = [
    g_5_s_5_6_complex_constraint,
]

seesaw_design = dc.Design(strands=strands, complex_constraints=complex_constraints)

ds.search_for_dna_sequences(design=seesaw_design,
                            # weigh_violations_equally=True,
                            report_delay=0.0,
                            out_directory='test_output',
                            report_only_violations=False,
                            )
