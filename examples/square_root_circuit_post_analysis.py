import argparse
from dataclasses import dataclass
from square_root_circuit import reporter_top_strand
from typing import List, Iterable, Dict, Tuple
from itertools import chain, combinations
from functools import reduce
from collections import defaultdict
from dsd.vienna_nupack import pfunc, binding
from nupack import Model, Complex, ComplexSet, SetSpec, complex_analysis
from nupack import Strand as NupackStrand


@dataclass
class Strand:
    name: str
    sequence: str


@dataclass
class SignalStrand(Strand):
    name: str
    gate_3p: str
    gate_5p: str
    sequence: str


@dataclass
class FuelStrand(Strand):
    name: str
    gate_3p: str
    sequence: str


@dataclass
class GateBaseStrand(Strand):
    gate: str
    sequence: str
    name: str


@dataclass
class ThresholdBottomStrand(Strand):
    input: str
    gate: str
    sequence: str


@dataclass
class ThresholdTopStrand(Strand):
    gate: str
    sequence: str
    name: str


@dataclass
class ReporterBottomStrand(Strand):
    gate: str
    sequence: str
    name: str


@dataclass
class ReporterTopStrand(Strand):
    gate: str
    sequence: str
    name: str


signal_strands: List[SignalStrand] = []
fuel_strands: List[FuelStrand] = []
gate_base_strands: List[GateBaseStrand] = []
threshold_bottom_strands: List[ThresholdBottomStrand] = []
threshold_top_strands: List[ThresholdTopStrand] = []
reporter_bottom_strands: List[ReporterBottomStrand] = []
reporter_top_strands: List[ReporterTopStrand] = []
strand_iterators: List[Iterable[Strand]] = [signal_strands, fuel_strands, gate_base_strands, threshold_bottom_strands,
                                            threshold_top_strands, reporter_bottom_strands, reporter_top_strands]
longest_strand_name_length: int


def process_line(line: str) -> None:
    """Read and parse line.

    :param line: Line.
    :type line: str
    """
    line_split = line.split()
    strand_name = line_split[0]
    strand_name_split = strand_name.split('_')
    sequence = ''.join(line_split[2:])
    if strand_name.startswith('signal'):
        assert len(strand_name_split) == 3
        gate_3p = strand_name_split[1]
        gate_5p = strand_name_split[2]
        assert len(line_split) == 5
        signal_strands.append(SignalStrand(gate_3p=gate_3p, gate_5p=gate_5p, sequence=sequence, name=strand_name))
    elif strand_name.startswith('fuel'):
        assert len(strand_name_split) == 2
        gate_3p = strand_name_split[1]
        assert len(line_split) == 5
        fuel_strands.append(FuelStrand(gate_3p=gate_3p, sequence=sequence, name=strand_name))
    elif strand_name.startswith('gate_base'):
        assert len(strand_name_split) == 3
        gate = strand_name_split[2]
        assert len(line_split) == 5
        gate_base_strands.append(GateBaseStrand(gate=gate, sequence=sequence, name=strand_name))
    elif strand_name.startswith('threshold_bottom'):
        assert len(strand_name_split) == 4
        input = strand_name_split[2]
        gate = strand_name_split[3]
        assert len(line_split) == 5
        threshold_bottom_strands.append(ThresholdBottomStrand(
            input=input, gate=gate, sequence=sequence, name=strand_name))
    elif strand_name.startswith('threshold_top'):
        assert len(strand_name_split) == 3
        gate = strand_name_split[2]
        assert len(line_split) == 3
        threshold_top_strands.append(ThresholdTopStrand(gate=gate, sequence=sequence, name=strand_name))
    elif strand_name.startswith('reporter_bottom'):
        assert len(strand_name_split) == 3
        gate = strand_name_split[2]
        assert len(line_split) == 4
        reporter_bottom_strands.append(ReporterBottomStrand(gate=gate, sequence=sequence, name=strand_name))
    elif strand_name.startswith('reporter_top'):
        assert len(strand_name_split) == 3
        gate = strand_name_split[2]
        assert len(line_split) == 3
        reporter_top_strands.append(ReporterTopStrand(gate=gate, sequence=sequence, name=strand_name))
    else:
        raise ValueError("Unexpected strand name")


def strands() -> Iterable[Strand]:
    return chain(*strand_iterators)


def format_strand_name(name) -> str:
    return f'{name: <{longest_strand_name_length}}'


binding_width = 5
binding_percision = 5


def calculate_and_print_binding(strand1: Strand, strand2: Strand, is_expected_to_bind: bool = False) -> None:
    is_expected_to_bind_str = ''
    if is_expected_to_bind:
        is_expected_to_bind_str = '(expected to bind)'
    binding_val = binding(strand1.sequence, strand2.sequence)
    print(f'{format_strand_name(strand1.name)} | {format_strand_name(strand2.name)} | binding: {binding_val: {binding_width}.{binding_percision}} {is_expected_to_bind_str}')


# Define physical model
model = Model(material='dna', celsius=37)


def calculate_and_print_pairs_and_mfe(strands: Iterable[Strand]) -> None:
    nupack_strands = [NupackStrand(s.sequence, name=s.name) for s in strands]

    # Define the complex of interest
    strand_complex = Complex(nupack_strands)

    # Define the complex set to contain only one complex
    complex_set = ComplexSet(strands=nupack_strands, complexes=SetSpec(max_size=0, include=[strand_complex]))

    # Analyze the complex
    # Calculate pfunc, pairs, mfe
    result = complex_analysis(complex_set, compute=['pfunc', 'pairs', 'mfe'], model=model)

    complex_result = result[strand_complex]
    print(f'\nMFE proxy structure for complex {strand_complex.name}:')
    for i, s in enumerate(complex_result.mfe):
        print('    %2d: %s (%.2f kcal/mol)' % (i, s.structure, s.energy))

    print(f'\nPair probabilities for complex {strand_complex.name}:')
    arr = complex_result.pairs.to_array()
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            p = arr[i][j]
            if p > 0.01:
                print(f'    P({i}, {j}) = {p}')


def main():
    parser = argparse.ArgumentParser(description='Runs analysis of square root circuit DNA sequences.')
    parser.add_argument('filename', help='The sequence.txt file.')

    args = parser.parse_args()
    filename = args.filename

    with open(filename) as f:
        for line in f:
            process_line(line)
    global longest_strand_name_length
    longest_strand_name_length = reduce(lambda acc, cur: max(acc, len(cur.name)), strands(), 0)

    gate_5p_to_signal_strands: Dict[str, List[SignalStrand]] = defaultdict(list)
    gate_3p_to_signal_strands: Dict[str, List[SignalStrand]] = defaultdict(list)
    gate_3p_gate_5p_to_signal_strands: Dict[Tuple[str, str], SignalStrand] = {}
    global signal_strands
    for s in signal_strands:
        gate_5p = s.gate_5p
        gate_3p = s.gate_3p
        gate_5p_to_signal_strands[gate_5p].append(s)
        gate_3p_to_signal_strands[gate_3p].append(s)
        assert (gate_3p, gate_5p) not in gate_3p_gate_5p_to_signal_strands
        gate_3p_gate_5p_to_signal_strands[(gate_3p, gate_5p)] = s

    print(f'Done processing {filename}')

    print('\nCalculating pfunc...')
    pfunc4_width = 5
    pfunc4_percision = 5
    for strand in strands():
        pfunc4_val = pfunc(strand.sequence)
        print(f'{format_strand_name(strand.name)} | pfunc: {pfunc4_val: {pfunc4_width}.{pfunc4_percision}}')

    print('\nCalculating binding between pairs of signal strands (including fuel)...')
    for strand1, strand2 in combinations(chain(signal_strands, fuel_strands), 2):
        calculate_and_print_binding(strand1, strand2)

    print('\nCalculating base-pairing probabilities and MFE for each input:gate complex')
    for gate_base_strand in gate_base_strands:
        assert gate_base_strand.gate in gate_5p_to_signal_strands
        output_strands = gate_5p_to_signal_strands[gate_base_strand.gate]
        for output_strand in output_strands:
            calculate_and_print_pairs_and_mfe([output_strand, gate_base_strand])

    print('\nCalculating base-pairing probabilities and MFE for each gate:output complex')
    for gate_base_strand in gate_base_strands:
        assert gate_base_strand.gate in gate_3p_to_signal_strands
        input_strands = gate_3p_to_signal_strands[gate_base_strand.gate]
        for input_strand in input_strands:
            calculate_and_print_pairs_and_mfe([input_strand, gate_base_strand])

    print('\nCalculating base-pairing probabilities and MFE for each gate:fuel complex')
    gate_to_gate_base_strand: Dict[str, GateBaseStrand] = {s.gate: s for s in gate_base_strands}
    for fuel_strand in fuel_strands:
        assert fuel_strand.gate_3p in gate_to_gate_base_strand
        gate_base_strand = gate_to_gate_base_strand[fuel_strand.gate_3p]
        calculate_and_print_pairs_and_mfe([fuel_strand, gate_base_strand])

    print('\nCalculating base-pairing probabilities and MFE for each threshold complex')
    gate_to_threshold_top_strand = {s.gate: s for s in threshold_top_strands}
    for threshold_bottom_strand in threshold_bottom_strands:
        assert threshold_bottom_strand.gate in gate_to_threshold_top_strand
        threshold_top_strand = gate_to_threshold_top_strand[threshold_bottom_strand.gate]
        calculate_and_print_pairs_and_mfe([threshold_top_strand, threshold_bottom_strand])

    print('\nCalculating base-pairing probabilities and MFE for each threshold waste complex')
    for threshold_bottom_strand in threshold_bottom_strands:
        assert (threshold_bottom_strand.input, threshold_bottom_strand.gate) in gate_3p_gate_5p_to_signal_strands
        signal_strand = gate_3p_gate_5p_to_signal_strands[(
            threshold_bottom_strand.input, threshold_bottom_strand.gate)]
        calculate_and_print_pairs_and_mfe([signal_strand, threshold_bottom_strand])

    print('\nCalculating base-pairing probabilities and MFE for each reporter complex')
    gate_to_reporter_top_strand: Dict[int, ReporterTopStrand] = {s.gate: s for s in reporter_top_strands}
    for reporter_bottom_strand in reporter_bottom_strands:
        assert reporter_bottom_strand.gate in gate_to_reporter_top_strand
        reporter_top_strand = gate_to_reporter_top_strand[reporter_bottom_strand.gate]
        calculate_and_print_pairs_and_mfe([reporter_top_strand, reporter_bottom_strand])

    print('\nCalculating base-pairing probabilities and MFE for each reporter waste complex')
    for reporter_bottom_strand in reporter_bottom_strands:
        signal_strands = gate_5p_to_signal_strands[reporter_bottom_strand.gate]
        assert len(signal_strands) == 1
        signal_strand = signal_strands[0]
        calculate_and_print_pairs_and_mfe([signal_strand, reporter_bottom_strand])

    # print('\nCalculating binding between same strand type...')
    # for itr in strand_iterators:
    #     for (strand1, strand2) in combinations(itr, 2):
    #         calculate_and_print_binding(strand1, strand2)

    # for signal_strand in signal_strands:
    #     log_prefix = f'\nCalculating binding between {signal_strand.name} and '
    #     print(f'{log_prefix}[fuel strands and threshold/reporter top strands]')
    #     for other_strand in chain(fuel_strands, threshold_top_strands, reporter_top_strands):
    #         calculate_and_print_binding(signal_strand, other_strand)

    #     print(f'{log_prefix}gate base strands')
    #     for gate_base_strand in gate_base_strands:
    #         gate = gate_base_strand.gate
    #         expected_to_bind = signal_strand.gate_3p == gate or signal_strand.gate_5p == gate
    #         calculate_and_print_binding(signal_strand, gate_base_strand, is_expected_to_bind=expected_to_bind)

    #     print(f'{log_prefix}threshold bottom strands')
    #     for threshold_bottom_strand in threshold_bottom_strands:
    #         expected_to_bind = signal_strand.gate_3p == threshold_bottom_strand.input and signal_strand.gate_5p == threshold_bottom_strand.gate
    #         calculate_and_print_binding(signal_strand, gate_base_strand, is_expected_to_bind=expected_to_bind)

    #     print(f'{log_prefix}reporter bottom strands')
    #     for reporter_bottom_strand in reporter_bottom_strands:
    #         expected_to_bind = signal_strand.gate_5p == reporter_bottom_strand.gate
    #         calculate_and_print_binding(signal_strand, reporter_bottom_strand, is_expected_to_bind=expected_to_bind)


if __name__ == '__main__':
    main()
