from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import dsd.search as ds  # type: ignore
import dsd.constraints as dc

def four_g_constraint(strand: List[dc.Strand]) -> dc.StrandConstraint:
    def four_g_constraint_evaluate(strand: dc.Strand):
        if 'GGGG' in strand.sequence():
            return 1000
        else:
            return 0

    def four_g_constraint_summary(strand: dc.Strand):
        violation_str = "" if 'GGGG' not in strand.sequence() else "** violation**"
        return f"{strand.name}: {strand.sequence()}{violation_str}"


    return dc.StrandConstraint(description="4GConstraint",
                                short_description="4GConstraint",
                                evaluate=four_g_constraint_evaluate ,
                                strands=tuple(strands),
                                summary=four_g_constraint_summary)


@dataclass
class SeesawCircuit:
    """Class for keeping track of a seesaw circuit and its DNA representation."""
    seesaw_gates: List['SeesawGate']
    strands: List[dc.Strand] = field(init=False)
    constraints: List[dc.ComplexConstraint] = field(init=False)

    def __post_init__(self) -> None:
        signal_strands: Dict[Tuple[int, int], dc.Strand] = {}
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

        self.strands = (list(signal_strands.values())
                        + list(gate_base_strands.values())
                        + list(threshold_base_strands.values())
                        + list(waste_strands.values())
                        + list(reporter_base_strands.values()))
        self.constraints = [dc.nupack_4_complex_secondary_structure_constraint(strand_complexes=cs) for cs in [
            input_gate_complexes,
            output_gate_complexes,
            threshold_waste_complexes,
            threshold_signal_complexes,
            reporter_waste_complexes,
            reporter_signal_complexes,
        ]]


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

def and_or_gate(integrating_gate_name: int, amplifying_gate_name: int, inputs: List[int]) -> Tuple[SeesawGate, SeesawGate]:
    integrating_gate = SeesawGate(gate_name=integrating_gate_name, inputs=inputs, has_threshold=False, has_fuel=False)
    amplifying_gate = SeesawGate(gate_name=amplifying_gate_name, inputs=[integrating_gate_name], has_threshold=True, has_fuel=True)
    return (integrating_gate, amplifying_gate)

def reporter_gate(gate_name: int, input: int) -> SeesawGate:
    return SeesawGate(gate_name=gate_name, inputs=[input], has_threshold=True, has_fuel=False, is_reporter=True)

def input_gate(gate_name: int, input) -> SeesawGate:
    return SeesawGate(gate_name=gate_name, inputs=[input], has_threshold=True, has_fuel=True)

seesaw_gates = [
    *and_or_gate(integrating_gate_name=10, amplifying_gate_name=1, inputs=[21, 27]),
    *and_or_gate(integrating_gate_name=53, amplifying_gate_name=5, inputs=[18, 22]),
    reporter_gate(gate_name=6, input=5),
    *and_or_gate(integrating_gate_name=20, amplifying_gate_name=8, inputs=[35, 38]),
    *and_or_gate(integrating_gate_name=26, amplifying_gate_name=13, inputs=[33, 37]),
    *and_or_gate(integrating_gate_name=34, amplifying_gate_name=18, inputs=[28, 33, 37]),
    *and_or_gate(integrating_gate_name=36, amplifying_gate_name=21, inputs=[29, 35, 38]),
    reporter_gate(gate_name=23, input=1),
    reporter_gate(gate_name=24, input=13),
    reporter_gate(gate_name=25, input=8),
    *and_or_gate(integrating_gate_name=39, amplifying_gate_name=22, inputs=[29, 31]),
    *and_or_gate(integrating_gate_name=40, amplifying_gate_name=27, inputs=[30, 28]),
    *and_or_gate(integrating_gate_name=41, amplifying_gate_name=28, inputs=[46, 48]),
    *and_or_gate(integrating_gate_name=42, amplifying_gate_name=29, inputs=[45, 47]),
    *and_or_gate(integrating_gate_name=43, amplifying_gate_name=30, inputs=[33, 38]),
    *and_or_gate(integrating_gate_name=44, amplifying_gate_name=31, inputs=[35, 37]),
    input_gate(gate_name=33, input=49),
    input_gate(gate_name=35, input=50),
    input_gate(gate_name=37, input=51),
    input_gate(gate_name=38, input=52),
]


seesaw_circuit = SeesawCircuit(seesaw_gates=seesaw_gates)
strands = seesaw_circuit.strands

design = dc.Design(strands=strands,
                   complex_constraints=seesaw_circuit.constraints,
                   strand_constraints=[four_g_constraint(strands)],
                  )


ds.search_for_dna_sequences(design=design,
                            # weigh_violations_equally=True,
                            report_delay=0.0,
                            out_directory='output/square_root_circuit',
                            report_only_violations=False,
                            )
