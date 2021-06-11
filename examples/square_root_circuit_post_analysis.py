import argparse
from dataclasses import dataclass
from typing import List, Iterable
from itertools import chain
from functools import reduce
from dsd.vienna_nupack import pfunc4


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
    return chain(
        signal_strands, fuel_strands, gate_base_strands, threshold_bottom_strands, threshold_top_strands,
        reporter_bottom_strands, reporter_top_strands)


def main():
    parser = argparse.ArgumentParser(description='Runs analysis of square root circuit DNA sequences.')
    parser.add_argument('filename', help='The sequence.txt file.')

    args = parser.parse_args()
    filename = args.filename

    with open(filename) as f:
        for line in f:
            process_line(line)

    print(f'Done processing {filename}')

    print('Calculating pfunc...')

    longest_strand_name_length = reduce(lambda acc, cur: max(acc, len(cur.name)), strands(), 0)
    for strand in strands():
        pfunc4_val = pfunc4(strand.sequence)
        print(f'{strand.name: <{longest_strand_name_length}} | pfunc: {pfunc4_val}')


if __name__ == '__main__':
    main()
