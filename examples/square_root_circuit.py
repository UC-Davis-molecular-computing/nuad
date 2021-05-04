from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Union

import dsd.search as ds  # type: ignore
import dsd.constraints as dc

# Constants

# Constants -- Recognition domain
#
# A recognition domain is made of two domains, a sub-domain for extending
# toeholds for thresholding reaction and the rest of the domain (called a
# sup-domain). The 3' end of the domain is located on the sup-domain while
# the 5' end of the domain is located on the sub-domain
#
# Assuming a total length of 15, and the sub-domain is length 2, then
# the sup-domain length is 13.
#
#        sup-domain  sub-domain
#           |         |
# 3'--=============--==--5'
SUP_REG_DOMAIN_PREFIX = 'S'
SUB_REG_DOMAIN_PREFIX = SUP_REG_DOMAIN_PREFIX.lower()
REG_DOMAIN_LENGTH = 15
SUB_REG_DOMAIN_LENGTH = 2
SUP_REG_DOMAIN_LENGTH = REG_DOMAIN_LENGTH - SUB_REG_DOMAIN_LENGTH

# Constants -- Toehold domain
TOEHOLD_DOMAIN = 'T'
TOEHOLD_COMPLEMENT = f'{TOEHOLD_DOMAIN}*'
TOEHOLD_LENGTH = 5

# Constants -- Fuel domain
FUEL_DOMAIN = 'f'

# NumpyConstraints
three_letter_code_constraint = dc.RestrictBasesConstraint(('A', 'C', 'T'))
no_gggg_constraint = dc.ForbiddenSubstringConstraint(['G'*4, 'C'*4])

# Domain pools
sup_reg_domain_constraints = [
    no_gggg_constraint,
    three_letter_code_constraint
]
SUP_REG_DOMAIN_POOL: dc.DomainPool = dc.DomainPool(
    'SUP_REG_DOMAIN_POOL', SUP_REG_DOMAIN_LENGTH,
    numpy_constraints=sup_reg_domain_constraints)

sub_reg_domain_constraints: List[dc.NumpyConstraint] = [
    three_letter_code_constraint
]
SUB_REG_DOMAIN_POOL: dc.DomainPool = dc.DomainPool(
    'SUB_REG_DOMAIN_POOL', SUB_REG_DOMAIN_LENGTH,
    numpy_constraints=sub_reg_domain_constraints)

toehold_domain_contraints: List[dc.NumpyConstraint] = [
    no_gggg_constraint,
]
TOEHOLD_DOMAIN_POOL: dc.DomainPool = dc.DomainPool(
    'TOEHOLD_DOMAIN_POOL', TOEHOLD_LENGTH,
    numpy_constraints=toehold_domain_contraints)

FUEL_DOMAIN_POOL: dc.DomainPool = dc.DomainPool(
    'FUEL_DOMAIN_POOL', REG_DOMAIN_LENGTH, [three_letter_code_constraint])


def signal_strand(
        gate3p: Union[int, str],
        gate5p: Union[int, str],
        name: str = '') -> dc.Strand:
    """Returns a signal strand with recognition domains
    gate3p and gate5p on the 3' and 5' respectively

    .. code-block:: none

          S{g3p}      s{g3p}  T       S{g5p}    s{g5p}
            |           |     |          |         |
        <=============--==--=====--=============--==]

    :param gate3p: Gate to be identified by the recognition domain on the 3'
                   end
    :type gate3p: Union[int, str]
    :param gate5p: Gate to be identified by the recognition domain on the 5'
                   end
    :type gate5p: Union[int, str]
    :param name: Name of the strand, defaults to 'signal {gate3p} {gate5p}'
    :type name: str, optional
    :return: Strand
    :rtype: dc.Strand
    """
    d3p_sup = f'{SUP_REG_DOMAIN_PREFIX}{gate3p}'
    d3p_sub = f'{SUB_REG_DOMAIN_PREFIX}{gate3p}'
    d5p_sup = f'{SUP_REG_DOMAIN_PREFIX}{gate5p}'
    d5p_sub = f'{SUB_REG_DOMAIN_PREFIX}{gate5p}'
    if name == '':
        name = f'signal {gate3p} {gate5p}'
    s: dc.Strand = dc.Strand(
        [d5p_sub, d5p_sup, TOEHOLD_DOMAIN, d3p_sub, d3p_sup], name=name)
    s.domains[0].pool = SUB_REG_DOMAIN_POOL
    s.domains[1].pool = SUP_REG_DOMAIN_POOL
    s.domains[2].pool = TOEHOLD_DOMAIN_POOL
    s.domains[3].pool = SUB_REG_DOMAIN_POOL
    s.domains[4].pool = SUP_REG_DOMAIN_POOL

    return s


def fuel_strand(gate: int) -> dc.Strand:
    """Returns a fuel strand with recognition domain `gate`.

    .. code-block:: none

          S{g3p}      s{g3p}  T         Sf         sf
            |           |     |          |         |
        <=============--==--=====--=============--==]

    :param gate: The name of the gate that this fuel strand will fuel.
    :type gate: int
    :return: Fuel strand
    :rtype: dc.Strand
    """
    return signal_strand(gate3p=gate, gate5p=FUEL_DOMAIN, name=f'fuel {gate}')


def gate_base_strand(gate: int) -> dc.Strand:
    """Returns a gate base strand with recognition domain `gate`.

    .. code-block:: none

           T*      S{gate}* s{gate}* T*
           |          |        |     |
        [=====--=============--==--=====>

    :param gate: Gate to be identified by the recognition domain
    :type gate: int
    :return: Gate base strand
    :rtype: dc.Strand
    """
    dsup = f'{SUP_REG_DOMAIN_PREFIX}{gate}*'
    dsub = f'{SUB_REG_DOMAIN_PREFIX}{gate}*'
    s: dc.Strand = dc.Strand(
        [TOEHOLD_COMPLEMENT, dsup, dsub, TOEHOLD_COMPLEMENT],
        name=f'gate base {gate}')
    s.domains[0].pool = TOEHOLD_DOMAIN_POOL
    s.domains[1].pool = SUP_REG_DOMAIN_POOL
    s.domains[2].pool = SUB_REG_DOMAIN_POOL
    s.domains[3].pool = TOEHOLD_DOMAIN_POOL
    return s


def threshold_base_strand(input: int, gate: int) -> dc.Strand:
    """Returns a threshold base strand for seesaw gate labeled `gate` that
    thresholds `input`

    .. code-block:: none

     s{input}* T*      S{gate}*    s{gate}*
         |     |          |        |
        [==--=====--=============--==>

    :param input: Name of input that is being thresholded
    :type input: int
    :param gate: Name of gate
    :type gate: int
    :return: Threshold base strand
    :rtype: dc.Strand
    """
    d_input_sub = f'{SUB_REG_DOMAIN_PREFIX}{input}*'
    d_gate_sup = f'{SUP_REG_DOMAIN_PREFIX}{gate}*'
    d_gate_sub = f'{SUB_REG_DOMAIN_PREFIX}{gate}*'
    s: dc.Strand = dc.Strand(
        [d_input_sub, TOEHOLD_COMPLEMENT, d_gate_sup, d_gate_sub],
        name=f'threshold base {input} {gate}')
    s.domains[0].pool = TOEHOLD_DOMAIN_POOL
    s.domains[1].pool = SUP_REG_DOMAIN_POOL
    s.domains[2].pool = SUB_REG_DOMAIN_POOL
    s.domains[3].pool = TOEHOLD_DOMAIN_POOL
    return s


def waste_strand(gate: int) -> dc.Strand:
    """Returns a waste strand for a thresholding/reporting reaction involving
    the seesaw gate labeled `gate`

    .. code-block:: none

            S{gate}   s{gate}
               |        |
        <=============--==]

    :param gate: Name of gate
    :type gate: int
    :return: Waste strand
    :rtype: dc.Strand
    """
    d_sup = f'{SUP_REG_DOMAIN_PREFIX}{gate}'
    d_sub = f'{SUB_REG_DOMAIN_PREFIX}{gate}'
    s: dc.Strand = dc.Strand([d_sub, d_sup], name=f'waste {gate}')
    s.domains[0].pool = SUB_REG_DOMAIN_POOL
    s.domains[1].pool = SUP_REG_DOMAIN_POOL
    return s


def reporter_base_strand(gate) -> dc.Strand:
    """Returns a reporter base strand for seesaw gate labeled `gate`

    .. code-block:: none

           T*     S{gate}*   s{gate}*
           |          |        |
        [=====--=============--==>

    :param gate: Name of gate
    :type gate: [type]
    :return: Reporter base strand
    :rtype: dc.Strand
    """
    d_sup = f'{SUP_REG_DOMAIN_POOL}{gate}*'
    d_sub = f'{SUB_REG_DOMAIN_POOL}{gate}*'

    s: dc.Strand = dc.Strand(
        [TOEHOLD_COMPLEMENT, d_sup, d_sub], name=f'reporter {gate}')
    s.domains[0].pool = TOEHOLD_DOMAIN_POOL
    s.domains[1].pool = SUP_REG_DOMAIN_POOL
    s.domains[2].pool = SUB_REG_DOMAIN_POOL
    return s


def gggg_constraint(strands: List[dc.Strand]) -> dc.StrandConstraint:
    """Returns a StrandConstraint that prevents a run of four Gs on a DNA
    strand sequence.

    :param strands: List of strands to check
    :type strands: List[dc.Strand]
    :return: StrandConstraint
    :rtype: dc.StrandConstraint
    """
    def gggg_constraint_evaluate(strand: dc.Strand):
        if 'GGGG' in strand.sequence():
            return 1000
        else:
            return 0

    def gggg_constraint_summary(strand: dc.Strand):
        violation_str: str
        if 'GGGG' not in strand.sequence():
            violation_str = ''
        else:
            violation_str = "** violation**"
        return f"{strand.name}: {strand.sequence()}{violation_str}"

    return dc.StrandConstraint(description="GGGGConstraint",
                               short_description="GGGGConstraint",
                               evaluate=gggg_constraint_evaluate,
                               strands=tuple(strands),
                               summary=gggg_constraint_summary)


@dataclass
class SeesawCircuit:
    """Class for keeping track of a seesaw circuit and its DNA representation.
    """
    seesaw_gates: List['SeesawGate']
    strands: List[dc.Strand] = field(init=False)
    constraints: List[dc.ComplexConstraint] = field(init=False)

    def __post_init__(self) -> None:
        signal_strands: Dict[Tuple[int, int], dc.Strand] = {}
        fuel_strands: Dict[int, dc.Strand] = {}
        gate_base_strands: Dict[int, dc.Strand] = {}
        threshold_base_strands: Dict[Tuple[int, int], dc.Strand] = {}
        waste_strands: Dict[int, dc.Strand] = {}
        reporter_base_strands: Dict[int, dc.Strand] = {}

        input_gate_complexes: List[Tuple[dc.Strand, ...]] = []
        output_gate_complexes: List[Tuple[dc.Strand, ...]] = []
        threshold_waste_complexes: List[Tuple[dc.Strand, ...]] = []
        threshold_signal_complexes: List[Tuple[dc.Strand, ...]] = []
        reporter_waste_complexes: List[Tuple[dc.Strand, ...]] = []
        reporter_signal_complexes: List[Tuple[dc.Strand, ...]] = []

        # Set of all input, gate pairs
        signal_strand_gates: Set[Tuple[int, int]] = set()
        # Set of all gates with fuel
        gates_with_fuel: Set[int] = set()
        # Set of all gates
        all_gates: Set[int] = set()
        # Set of all input, gate pairs with threshold
        input_gates_with_threshold: Set[Tuple[int, int]] = set()
        # Set of all gates with threshold
        gates_with_threshold: Set[int] = set()
        # Set of all reporter gates
        all_reporter_gates: Set[int] = set()

        # Populate sets
        for seesaw_gate in self.seesaw_gates:
            gate_name = seesaw_gate.gate_name

            if gate_name in all_gates:
                raise ValueError(f'Invalid seesaw circuit: '
                                 'Multiple gates labeled {gate_name} found')
            all_gates.add(gate_name)
            for input in seesaw_gate.inputs:
                assert (input, gate_name) not in signal_strand_gates
                signal_strand_gates.add((input, gate_name))

                if seesaw_gate.has_threshold:
                    assert gate_name not in input_gates_with_threshold
                    input_gates_with_threshold.add((input, gate_name))

            if seesaw_gate.has_threshold:
                assert gate_name not in gates_with_threshold
                gates_with_threshold.add(gate_name)

            if seesaw_gate.has_fuel:
                assert gate_name not in gates_with_fuel
                gates_with_fuel.add(gate_name)

            if seesaw_gate.is_reporter:
                assert gate_name not in all_reporter_gates
                assert seesaw_gate.has_threshold
                all_reporter_gates.add(gate_name)

        signal_strands = {(input, gate): signal_strand(input, gate)
                          for input, gate in signal_strand_gates}

        fuel_strands = {gate: fuel_strand(gate) for gate in gates_with_fuel}

        gate_base_strands = {gate: gate_base_strand(gate)
                             for gate in all_gates}

        threshold_base_strands = {(input, gate): threshold_base_strand(
            input, gate) for input, gate in input_gates_with_threshold}

        waste_strands = {gate: waste_strand(gate)
                         for gate in gates_with_threshold}

        reporter_base_strands = {gate: reporter_base_strand(gate)
                                 for gate in all_reporter_gates}

        self.strands = (list(signal_strands.values())
                        + list(fuel_strands.values())
                        + list(gate_base_strands.values())
                        + list(threshold_base_strands.values())
                        + list(waste_strands.values())
                        + list(reporter_base_strands.values()))
        func = dc.nupack_4_complex_secondary_structure_constraint
        self.constraints = list(map(func, filter(lambda c: len(c) > 0, [
            input_gate_complexes,
            output_gate_complexes,
            threshold_waste_complexes,
            threshold_signal_complexes,
            reporter_waste_complexes,
            reporter_signal_complexes,
        ])))


@dataclass(frozen=True)
class SeesawGate:
    """Class for keeping track of seesaw gate and its input."""
    gate_name: int
    inputs: List[int]
    has_threshold: bool
    has_fuel: bool
    is_reporter: bool = False

    def get_signal_strands(self) -> List[dc.Strand]:
        raise NotImplemented

    def get_gate_strand(self) -> dc.Strand:
        raise NotImplemented

    def get_strands(self) -> List[dc.Strand]:
        raise NotImplemented


def and_or_gate(integrating_gate_name: int, amplifying_gate_name: int,
                inputs: List[int]) -> Tuple[SeesawGate, SeesawGate]:
    """Returns two SeesawGate objects (the integrating gate and amplifying
    gate) that implements the AND or OR gate

    :param integrating_gate_name: Name for integrating gate
    :type integrating_gate_name: int
    :param amplifying_gate_name: Name for amplifying gate
    :type amplifying_gate_name: int
    :param inputs: Inputs into the AND or OR gate
    :type inputs: List[int]
    :return: An integrating gate and an amplifying gate
    :rtype: Tuple[SeesawGate, SeesawGate]
    """
    integrating_gate = SeesawGate(
        gate_name=integrating_gate_name, inputs=inputs, has_threshold=False,
        has_fuel=False)
    amplifying_gate = SeesawGate(
        gate_name=amplifying_gate_name, inputs=[integrating_gate_name],
        has_threshold=True, has_fuel=True)
    return (integrating_gate, amplifying_gate)


def reporter_gate(gate_name: int, input: int) -> SeesawGate:
    """Returns a SeesawGate for a reporter

    :param gate_name: Name of the reporter
    :type gate_name: int
    :param input: Input
    :type input: int
    :return: SeesawGate for a reporter
    :rtype: SeesawGate
    """
    return SeesawGate(
        gate_name=gate_name, inputs=[input],
        has_threshold=True, has_fuel=False, is_reporter=True)


def input_gate(gate_name: int, input: int) -> SeesawGate:
    """Returns a SeesawGate for an input

    :param gate_name: Name of the gate
    :type gate_name: int
    :param input: Input
    :type input: int
    :return: SeesawGate
    :rtype: SeesawGate
    """
    return SeesawGate(
        gate_name=gate_name, inputs=[input],
        has_threshold=True, has_fuel=True)


def main() -> None:
    seesaw_gates = [
        *and_or_gate(integrating_gate_name=10,
                     amplifying_gate_name=1, inputs=[21, 27]),
        *and_or_gate(integrating_gate_name=53,
                     amplifying_gate_name=5, inputs=[18, 22]),
        reporter_gate(gate_name=6, input=5),
        *and_or_gate(integrating_gate_name=20,
                     amplifying_gate_name=8, inputs=[35, 38]),
        *and_or_gate(integrating_gate_name=26,
                     amplifying_gate_name=13, inputs=[33, 37]),
        *and_or_gate(integrating_gate_name=34,
                     amplifying_gate_name=18, inputs=[28, 33, 37]),
        *and_or_gate(integrating_gate_name=36,
                     amplifying_gate_name=21, inputs=[29, 35, 38]),
        reporter_gate(gate_name=23, input=1),
        reporter_gate(gate_name=24, input=13),
        reporter_gate(gate_name=25, input=8),
        *and_or_gate(integrating_gate_name=39,
                     amplifying_gate_name=22, inputs=[29, 31]),
        *and_or_gate(integrating_gate_name=40,
                     amplifying_gate_name=27, inputs=[30, 28]),
        *and_or_gate(integrating_gate_name=41,
                     amplifying_gate_name=28, inputs=[46, 48]),
        *and_or_gate(integrating_gate_name=42,
                     amplifying_gate_name=29, inputs=[45, 47]),
        *and_or_gate(integrating_gate_name=43,
                     amplifying_gate_name=30, inputs=[33, 38]),
        *and_or_gate(integrating_gate_name=44,
                     amplifying_gate_name=31, inputs=[35, 37]),
        input_gate(gate_name=33, input=49),
        input_gate(gate_name=35, input=50),
        input_gate(gate_name=37, input=51),
        input_gate(gate_name=38, input=52),
    ]

    seesaw_circuit = SeesawCircuit(seesaw_gates=seesaw_gates)
    strands = seesaw_circuit.strands

    for s in sorted(strands, key=lambda s: s.name):
        print(s)
    exit(0)

    design = dc.Design(strands=strands,
                       complex_constraints=seesaw_circuit.constraints,
                       strand_constraints=[gggg_constraint(strands)],
                       )

    ds.search_for_dna_sequences(design=design,
                                # weigh_violations_equally=True,
                                report_delay=0.0,
                                out_directory='output/square_root_circuit',
                                report_only_violations=False,
                                )


if __name__ == '__main__':
    main()
