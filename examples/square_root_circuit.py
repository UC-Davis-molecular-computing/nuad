from dataclasses import dataclass
from typing import List

@dataclass
class SeesawGate:
    """Class for keeping track of seesaw gate and its input."""
    gate_name: int
    inputs: List[int]

seesaw_gates = [
    SeesawGate(gate_name=1, inputs=[10]),
    SeesawGate(gate_name=5, inputs=[53]),
    SeesawGate(gate_name=8, inputs=[20]),
    SeesawGate(gate_name=10, inputs=[21, 27]),
    SeesawGate(gate_name=13, inputs=[26]),
    SeesawGate(gate_name=18, inputs=[34]),
    SeesawGate(gate_name=20, inputs=[35, 38]),
    SeesawGate(gate_name=21, inputs=[36]),
    SeesawGate(gate_name=22, inputs=[39]),
    SeesawGate(gate_name=26, inputs=[33, 37]),
    SeesawGate(gate_name=27, inputs=[40]),
    SeesawGate(gate_name=28, inputs=[41]),
    SeesawGate(gate_name=29, inputs=[42]),
    SeesawGate(gate_name=30, inputs=[43]),
    SeesawGate(gate_name=31, inputs=[44]),
    SeesawGate(gate_name=33, inputs=[49]),
    SeesawGate(gate_name=34, inputs=[28, 33, 37]),
    SeesawGate(gate_name=35, inputs=[50]),
    SeesawGate(gate_name=36, inputs=[29, 35, 38]),
    SeesawGate(gate_name=37, inputs=[51]),
    SeesawGate(gate_name=38, inputs=[52]),
    SeesawGate(gate_name=39, inputs=[29, 31]),
    SeesawGate(gate_name=40, inputs=[30, 28]),
    SeesawGate(gate_name=41, inputs=[46, 48]),
    SeesawGate(gate_name=42, inputs=[45, 47]),
    SeesawGate(gate_name=43, inputs=[33, 38]),
    SeesawGate(gate_name=44, inputs=[35, 37]),
    SeesawGate(gate_name=53, inputs=[18, 22]),
]

print(len(seesaw_gates))