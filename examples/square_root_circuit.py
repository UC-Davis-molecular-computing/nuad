from dataclasses import dataclass
from typing import List, Tuple

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

def convert_seesaw_gates_to_strands_and_constraints(seesaw_gates: List[SeesawGate]) -> Tuple[List[dc.Strand], List[dc.ComplexConstraint]]:
    strands: List[dc.Strand] = []
    for seesaw_gate in seesaw_gates:
        strands += seesaw_gate.get_strands()

    input_gate_cssc: dc.ComplexConstraint
    output_gate_cssc: dc.ComplexConstraint
    threshold_waste_cssc: dc.ComplexConstraint
    threshold_signal_cssc: dc.ComplexConstraint
    reporter_waste_cssc: dc.ComplexConstraint
    reporter_signal_cssc: dc.ComplexConstraint

    constraints = [
        input_gate_cssc,
        output_gate_cssc,
        threshold_waste_cssc,
        threshold_signal_cssc,
        reporter_waste_cssc,
        reporter_signal_cssc,
    ]
    return (strands, constraints)

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

print(len(seesaw_gates))
exit


strands: List[dc.Strand] = []
complex_constraints: List[dc.ComplexConstraint] = []

(s, c) = convert_seesaw_gates_to_strands_and_constraints(seesaw_gates)


design = dc.Design(strands=strands,
                   complex_constraints=complex_constraints,
                   strand_constraints=[four_g_constraint(strands)],
                  )


ds.search_for_dna_sequences(design=design,
                            # weigh_violations_equally=True,
                            report_delay=0.0,
                            out_directory='output/square_root_circuit',
                            report_only_violations=False,
                            )
