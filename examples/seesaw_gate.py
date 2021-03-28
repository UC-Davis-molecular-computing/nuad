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

# Gate Bases
#    T*         S5*          T*
# [=====--===============--=====>
gate_5_base_strand = gate_base_strand(5)

# Collect all strands
strands = [
    signal_6_5_strand,
    gate_5_base_strand
]

# Complexes
#
#               S5           T           S6
#        <===============--=====--===============]
# [=====--===============--=====>
#    T*         S5*          T*
g_5_s_5_6_complex = (signal_6_5_strand, gate_5_base_strand)
signal_toehold_addr = dc.StrandDomainAddress.address_of_first_domain_occurence(signal_6_5_strand, TOEHOLD_DOMAIN)
gate_bound_toehold_addr = dc.StrandDomainAddress.address_of_last_domain_occurence(gate_5_base_strand, 'T*')

g_5_s_5_6_complex_constraint = dc.nupack_4_complex_secondary_structure_constraint(
    complexes=[g_5_s_5_6_complex],
    nonimplicit_base_pairs=[dc.StrandDomainAddress.address_of_first_domain_occurence()]
)



# Constraints
complex_constraints = [
]


seesaw_design = dc.Design(
    strands=strands, complex_constraints=complex_constraints)

ds.search_for_dna_sequences(design=seesaw_design,
                            # weigh_violations_equally=True,
                            report_delay=0.0,
                            out_directory='test_output',
                            report_only_violations=False,
                            )
