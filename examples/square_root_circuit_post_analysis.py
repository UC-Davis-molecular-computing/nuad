import argparse
from dataclasses import dataclass
from typing import List


@dataclass
class SignalStrand:
    gate_3p: str
    gate_5p: str
    sequence: str


@dataclass
class FuelStrand:
    gate_3p: str
    sequence: str


@dataclass
class GateBaseStrand:
    gate: str
    sequence: str


@dataclass
class ThresholdBottomStrand:
    input: str
    gate: str
    sequence: str


@dataclass
class ThresholdTopStrand:
    gate: str
    sequence: str


@dataclass
class ReporterBottomStrand:
    gate: str
    sequence: str


@dataclass
class ReporterTopStrand:
    gate: str
    sequence: str


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
        signal_strands.append(SignalStrand(gate_3p=gate_3p, gate_5p=gate_5p, sequence=sequence))
    elif strand_name.startswith('fuel'):
        assert len(strand_name_split) == 2
        gate_3p = strand_name_split[1]
        assert len(line_split) == 5
        fuel_strands.append(FuelStrand(gate_3p=gate_3p, sequence=sequence))
    elif strand_name.startswith('gate_base'):
        assert len(strand_name_split) == 3
        gate = strand_name_split[2]
        assert len(line_split) == 5
        gate_base_strands.append(GateBaseStrand(gate=gate, sequence=sequence))
    elif strand_name.startswith('threshold_bottom'):
        assert len(strand_name_split) == 4
        input = strand_name_split[2]
        gate = strand_name_split[3]
        assert len(line_split) == 5
        threshold_bottom_strands.append(ThresholdBottomStrand(input=input, gate=gate, sequence=sequence))
    elif strand_name.startswith('threshold_top'):
        assert len(strand_name_split) == 3
        gate = strand_name_split[2]
        assert len(line_split) == 3
        threshold_top_strands.append(ThresholdTopStrand(gate=gate, sequence=sequence))
    elif strand_name.startswith('reporter_bottom'):
        assert len(strand_name_split) == 3
        gate = strand_name_split[2]
        assert len(line_split) == 4
        reporter_bottom_strands.append(ReporterBottomStrand(gate=gate, sequence=sequence))
    elif strand_name.startswith('reporter_top'):
        assert len(strand_name_split) == 3
        gate = strand_name_split[2]
        assert len(line_split) == 3
        reporter_top_strands.append(ReporterTopStrand(gate=gate, sequence=sequence))
    else:
        raise ValueError("Unexpected strand name")


def main():
    parser = argparse.ArgumentParser(description='Runs analysis of square root circuit DNA sequences.')
    parser.add_argument('filename', help='The sequence.txt file.')

    args = parser.parse_args()
    filename = args.filename

    with open(filename) as f:
        for line in f:
            process_line(line)

    print('Done processing')


if __name__ == '__main__':
    main()
