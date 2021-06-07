from dataclasses import dataclass, field
from math import ceil, floor
from typing import Dict, List, Set, Tuple, Union, cast

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
TOEHOLD_LENGTH = 5

# Constants -- Fuel domain
FUEL_DOMAIN = 'f'

# NumpyConstraints
ILLEGAL_SUBSTRINGS_FOUR = ['G'*4, 'C'*4]
ILLEGAL_SUBSTRINGS_FIVE = ['A'*5, 'T'*5]
ILLEGAL_SUBSTRINGS = ILLEGAL_SUBSTRINGS_FOUR + ILLEGAL_SUBSTRINGS_FIVE


three_letter_code_constraint = dc.RestrictBasesConstraint(('A', 'C', 'T'))
no_gggg_constraint = dc.ForbiddenSubstringConstraint(ILLEGAL_SUBSTRINGS_FOUR)
no_aaaaa_constraint = dc.ForbiddenSubstringConstraint(ILLEGAL_SUBSTRINGS_FIVE)


def c_content_constraint(length: int) -> dc.BaseCountConstraint:
    """Returns a BaseCountConstraint that enforces 30% to 70% C-content

    :param length: Length of DNA sequence
    :type length: int
    :return: BaseCountConstraint
    :rtype: dc.BaseCountConstraint
    """
    high_count = floor(0.7 * length)
    low_count = ceil(0.3 * length)
    return dc.BaseCountConstraint('C', high_count, low_count)


# TODO: 30% to 70% C-content on non-fuel strands
# TODO:  For any two sequences in the pool, we require at least 30% of bases are
# different - Use a DomainPairConstraint
# and the longest run of matches is at most 35% of the domain length - Use a
# DomainPairConstraint
# * Check that the domain pairs are same length and both long domains

# Domain pools
sup_reg_domain_constraints = [
    no_gggg_constraint,
    three_letter_code_constraint,
]
SUP_REG_DOMAIN_POOL: dc.DomainPool = dc.DomainPool(
    'SUP_REG_DOMAIN_POOL', SUP_REG_DOMAIN_LENGTH,
    numpy_constraints=sup_reg_domain_constraints)

sub_reg_domain_constraints: List[dc.NumpyConstraint] = [
    three_letter_code_constraint,
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

SIGNAL_DOMAIN_POOL: dc.DomainPool = dc.DomainPool(
    'SIGNAL_DOMAIN_POOL', REG_DOMAIN_LENGTH,
    [three_letter_code_constraint, c_content_constraint(REG_DOMAIN_LENGTH)])

TOEHOLD_DOMAIN: dc.Domain = dc.Domain('T', pool=TOEHOLD_DOMAIN_POOL)

# Alias
dc_complex_constraint = dc.nupack_4_complex_secondary_structure_constraint

# Stores all domains used in design
all_domains: Dict[str, dc.Domain] = {'T': TOEHOLD_DOMAIN}


def get_signal_domain(gate: Union[int, str]) -> dc.Domain:
    """Returns a signal domain with S{gate} and stores it to all_domains for
    future use.

    :param gate: Gate.
    :type gate: str
    :return: Domain
    :rtype: Domain
    """
    if f'S{gate}' not in all_domains:
        d3p_sup: dc.Domain = dc.Domain(f'ss{gate}', pool=SUP_REG_DOMAIN_POOL, dependent=True)
        d3p_sub: dc.Domain = dc.Domain(f's{gate}', pool=SUB_REG_DOMAIN_POOL, dependent=True)
        d3p: dc.Domain = dc.Domain(f'S{gate}', pool=SIGNAL_DOMAIN_POOL, subdomains=[d3p_sub, d3p_sup])

        all_domains[f'ss{gate}'] = d3p_sup
        all_domains[f's{gate}'] = d3p_sub
        all_domains[f'S{gate}'] = d3p

    return all_domains[f'S{gate}']


def set_domain_pool(domain: dc.Domain, domain_pool: dc.DomainPool) -> None:
    """Assigns domain_pool to domain. If domain already has a domain pool, this
    function asserts that the pool matches the domain_pool.

    :param domain: Domain to be assigned a pool
    :type domain: dc.Domain
    :param domain_pool: Pool to assign to Domain
    :type domain_pool: dc.DomainPool
    """
    if domain._pool:
        if domain.pool is not domain_pool:
            raise AssertionError(f'Assigning pool {domain_pool} to domain '
                                 f'{domain} but {domain} already has domain '
                                 f'pool {domain_pool}')
    else:
        domain.pool = domain_pool


def signal_strand(
        gate3p: Union[int, str],
        gate5p: Union[int, str],
        name: str = '') -> dc.Strand:
    """Returns a signal strand with recognition domains
    gate3p and gate5p on the 3' and 5' respectively

    .. code-block:: none

                S{g3p}                       S{g5p}
          ss{g3p}      s{g3p}  T       ss{g5p}    s{g5p}
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
    d3p = get_signal_domain(gate3p)
    d5p = get_signal_domain(gate5p)

    if name == '':
        name = f'signal {gate3p} {gate5p}'
    return dc.Strand(domains=[d5p, TOEHOLD_DOMAIN, d3p], starred_domain_indices=[], name=name)


def fuel_strand(gate: int) -> dc.Strand:
    """Returns a fuel strand with recognition domain `gate`.

    .. code-block:: none

         ss{g3p}      s{g3p}  T        ssf         sf
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
                        S{gate}*
           T*      ss{gate}* s{gate}* T*
           |          |        |      |
        [=====--=============--==--=====>

    :param gate: Gate to be identified by the recognition domain
    :type gate: int
    :return: Gate base strand
    :rtype: dc.Strand
    """
    d = get_signal_domain(gate)
    s: dc.Strand = dc.Strand(
        domains=[TOEHOLD_DOMAIN, d, TOEHOLD_DOMAIN],
        starred_domain_indices=[0, 1, 2],
        name=f'gate base {gate}')
    return s


def threshold_base_strand(input: int, gate: int) -> dc.Strand:
    """Returns a threshold base strand for seesaw gate labeled `gate` that
    thresholds `input`

    .. code-block:: none

     s{input}* T*      ss{gate}*   s{gate}*
         |     |          |        |
        [==--=====--=============--==>

    :param input: Name of input that is being thresholded
    :type input: int
    :param gate: Name of gate
    :type gate: int
    :return: Threshold base strand
    :rtype: dc.Strand
    """
    # Note, this assumes that this input signal domain has already been built
    d_input_sub = all_domains[f's{input}']
    d_gate = get_signal_domain(gate)

    s: dc.Strand = dc.Strand(
        domains=[d_input_sub, TOEHOLD_DOMAIN, d_gate],
        starred_domain_indices=[0, 1, 2],
        name=f'threshold base {input} {gate}')
    return s


def waste_strand(gate: int) -> dc.Strand:
    """Returns a waste strand for a thresholding/reporting reaction involving
    the seesaw gate labeled `gate`

    .. code-block:: none

            ss{gate}   s{gate}
               |        |
        <=============--==]

    :param gate: Name of gate
    :type gate: int
    :return: Waste strand
    :rtype: dc.Strand
    """
    s: dc.Strand = dc.Strand(domains=[get_signal_domain(gate)], starred_domain_indices=[], name=f'waste {gate}')
    return s


def reporter_base_strand(gate) -> dc.Strand:
    """Returns a reporter base strand for seesaw gate labeled `gate`

    .. code-block:: none

           T*     ss{gate}*   s{gate}*
           |          |        |
        [=====--=============--==>

    :param gate: Name of gate
    :type gate: [type]
    :return: Reporter base strand
    :rtype: dc.Strand
    """
    s: dc.Strand = dc.Strand(
        domains=[TOEHOLD_DOMAIN, get_signal_domain(gate)], starred_domain_indices=[0, 1], name=f'reporter {gate}')
    return s


def input_gate_complex_constraint(input_gate_complexes: List[Tuple[dc.Strand, dc.Strand]]) -> dc.ComplexConstraint:
    """Returns a input:gate complex constraint

    .. code-block:: none

          S{input}  s{input}  T       S{gate}    s{gate}
            |           |     |          |         |
        <=============--==--=====--=============--==]
                            |||||  |||||||||||||  ||
                           [=====--=============--==--=====>
                              |          |        |     |
                              T*      S{gate}* s{gate}* T*

              S2        s2    T          S5       s5
                       21
         34          22|20 19  15 14          2  10
         |           | ||  |   |  |           |  ||
        <=============-==--=====--=============--==]
                           |||||  |||||||||||||  ||
                          [=====--=============--==--=====>
                           |   |  |           |  ||  |   |
                           35  39 40          52 |54 55  59
                                                 53
                             T*         S5*      s5*   T*

    :param input_gate_complexes: List of input:gate complexes
    :type input_gate_complexes: List[Tuple[dc.Strand, ...]]
    :return: A complex constraint on the base-pairing probabilities
    :rtype: dc.ComplexConstraint
    """
    assert input_gate_complexes
    template_complex = input_gate_complexes[0]
    assert len(template_complex) == 2
    template_top_strand = template_complex[0]
    template_bot_strand = template_complex[1]
    addr_T = template_top_strand.address_of_first_domain_occurence('T')
    addr_T_star = template_bot_strand.address_of_first_domain_occurence('T*')
    return dc_complex_constraint(
        strand_complexes=cast(
            List[Tuple[dc.Strand, ...]],
            input_gate_complexes),
        nonimplicit_base_pairs=[(addr_T, addr_T_star)],
        description="input:gate Complex",
        short_description="input:gate")


def gate_output_complex_constraint(gate_output_complexes: List[Tuple[dc.Strand, ...]]) -> dc.ComplexConstraint:
    """Returns a gate:output complex constraint

    .. code-block:: none

                   S{gate}  s{gate}  T      S{output}    s{output}
                      |        |     |          |         |
               <=============--==--=====--=============--==]
                |||||||||||||  ||  |||||
        [=====--=============--==--=====>
           |          |        |     |
           T*      S{gate}* s{gate}* T*

                    S5         s5    T        S6 / S7  s6 / s7
                               21
                34          22 |20 19  15 14          2  10
                |           |  ||  |   |  |           |  ||
               <=============--==--=====--=============--==]
                |||||||||||||  ||  |||||
        [=====--=============--==--=====>
         |   |  |           |  ||  |   |
         35  39 40          52 |54 55  59
                               53
           T*         S5*      s5*   T*

    :param output_gate_complexes: List of gate:output complexes
    :type output_gate_complexes: List[Tuple[dc.Strand, ...]]
    :return: A complex constraint on the base-pairing probabilities
    :rtype: dc.ComplexConstraint
    """
    assert gate_output_complexes
    template_complex = gate_output_complexes[0]
    assert len(template_complex) == 2
    template_top_strand = template_complex[0]
    template_bot_strand = template_complex[1]
    addr_T = template_top_strand.address_of_first_domain_occurence('T')
    addr_T_star = template_bot_strand.address_of_last_domain_occurence('T*')
    return dc_complex_constraint(
        strand_complexes=gate_output_complexes,
        nonimplicit_base_pairs=[(addr_T, addr_T_star)],
        description="gate:output Complex",
        short_description="gate:output"
    )


def strand_substring_constraint(
        strands: List[dc.Strand],
        substrings: List[str]) -> dc.StrandConstraint:
    """Returns a strand constraint that restricts the substrings in the strand
    sequence

    :param strands: Strands to apply constraint on
    :type strands: List[dc.Strand]
    :param substrings: Substrings to disallow
    :type substrings: List[str]
    :return: [description]
    :rtype: dc.StrandConstraint
    """
    def violated(strand: dc.Strand):
        for substring in substrings:
            if substring in strand.sequence():
                return True
        return False

    def evaluate(strand: dc.Strand):
        if violated(strand):
            return 100
        else:
            return 0

    def summary(strand: dc.Strand):
        violation_str: str
        if violated(strand):
            violation_str = ''
        else:
            violation_str = "** violation**"
        return f"{strand.name}: {strand.sequence()}{violation_str}"

    return dc.StrandConstraint(description="Strand Substring Constraint",
                               short_description="Strand Substring Constraint",
                               evaluate=evaluate,
                               strands=tuple(strands),
                               summary=summary)


@dataclass
class SeesawCircuit:
    """Class for keeping track of a seesaw circuit and its DNA representation.
    """
    seesaw_gates: List['SeesawGate']
    strands: List[dc.Strand] = field(init=False, default_factory=list)
    constraints: List[dc.ComplexConstraint] = field(
        init=False, default_factory=list)

    signal_strands: Dict[Tuple[int, int], dc.Strand] = field(
        init=False, default_factory=dict)
    fuel_strands: Dict[int, dc.Strand] = field(
        init=False, default_factory=dict)
    gate_base_strands: Dict[int, dc.Strand] = field(
        init=False, default_factory=dict)
    threshold_base_strands: Dict[Tuple[int, int], dc.Strand] = field(
        init=False, default_factory=dict)
    waste_strands: Dict[int, dc.Strand] = field(
        init=False, default_factory=dict)
    reporter_base_strands: Dict[Tuple[int, int], dc.Strand] = field(
        init=False, default_factory=dict)

    def _set_gate_base_strands(self) -> None:
        """Sets self.gate_base_strands

        :raises ValueError: If duplicate gate name found
        """
        # Set of all gates
        gates: Set[int] = set()
        for seesaw_gate in self.seesaw_gates:
            gate_name = seesaw_gate.gate_name
            if gate_name in gates:
                raise ValueError(f'Invalid seesaw circuit: '
                                 'Multiple gates labeled {gate_name} found')
            gates.add(gate_name)

        self.gate_base_strands = {gate: gate_base_strand(gate)
                                  for gate in gates}

    def _set_signal_strands(self) -> None:
        """Sets self.signal_strands

        :raises ValueError: If duplicate gate name found
        """
        # Set of all input, gate pairs
        input_gate_pairs: Set[Tuple[int, int]] = set()
        for seesaw_gate in self.seesaw_gates:
            gate_name = seesaw_gate.gate_name
            if gate_name in input_gate_pairs:
                raise ValueError(f'Invalid seesaw circuit: '
                                 'Multiple gates labeled {gate_name} found')
            for input in seesaw_gate.inputs:
                assert (input, gate_name) not in input_gate_pairs
                input_gate_pairs.add((input, gate_name))

        self.signal_strands = {(input, gate): signal_strand(input, gate)
                               for input, gate in input_gate_pairs}

    def _set_fuel_strands(self) -> None:
        """Sets self.fuel_strands

        :raises ValueError: If duplicate gate name found
        """
        # Set of all gates with fuel
        gates_with_fuel: Set[int] = set()
        for seesaw_gate in self.seesaw_gates:
            if seesaw_gate.has_fuel:
                gate_name = seesaw_gate.gate_name
                if gate_name in gates_with_fuel:
                    raise ValueError(
                        f'Invalid seesaw circuit: '
                        'Multiple gates labeled {gate_name} found')
                gates_with_fuel.add(gate_name)

        self.fuel_strands = {gate: fuel_strand(
            gate) for gate in gates_with_fuel}

    def _set_threshold_base_strands(self) -> None:
        """Sets self.threshold_base_strands

        :raises ValueError: If duplicate gate name found
        """
        # Set of all input, gate pairs with threshold
        input_gate_pairs_with_threshold: Set[Tuple[int, int]] = set()
        for seesaw_gate in self.seesaw_gates:
            if seesaw_gate.has_threshold:
                gate_name = seesaw_gate.gate_name
                if gate_name in input_gate_pairs_with_threshold:
                    raise ValueError(
                        f'Invalid seesaw circuit: '
                        'Multiple gates labeled {gate_name} found')
                for input in seesaw_gate.inputs:
                    assert (input, gate_name) not in input_gate_pairs_with_threshold
                    input_gate_pairs_with_threshold.add((input, gate_name))

        self.threshold_base_strands = {(input, gate): threshold_base_strand(
            input, gate) for input, gate in input_gate_pairs_with_threshold}

    def _set_waste_strands(self) -> None:
        """Sets self.waste_strands

        :raises ValueError: If duplicate gate name found
        """
        # Set of all gates with threshold
        gates_with_threshold: Set[int] = set()

        for seesaw_gate in self.seesaw_gates:
            if seesaw_gate.has_threshold:
                gate_name = seesaw_gate.gate_name
                if gate_name in gates_with_threshold:
                    raise ValueError(
                        f'Invalid seesaw circuit: '
                        'Multiple gates labeled {gate_name} found')
                gates_with_threshold.add(gate_name)

        self.waste_strands = {gate: waste_strand(gate)
                              for gate in gates_with_threshold}

    def _set_reporter_gates(self) -> None:
        """Sets self.reporter_gates

        :raises ValueError: If duplicate gate name found
        """
        # Set of all reporter gates
        reporter_gates: Set[Tuple[int, int]] = set()
        for seesaw_gate in self.seesaw_gates:
            if seesaw_gate.is_reporter:
                gate_name = seesaw_gate.gate_name
                if gate_name in reporter_gates:
                    raise ValueError(
                        f'Invalid seesaw circuit: '
                        'Multiple gates labeled {gate_name} found')
                inputs = seesaw_gate.inputs
                assert len(inputs) == 1
                reporter_gates.add((inputs[0], gate_name))

        self.reporter_base_strands = {(input, gate): reporter_base_strand(gate)
                                      for input, gate in reporter_gates}

    def _set_strands(self) -> None:
        """Sets self.strands
        """
        self._set_gate_base_strands()
        self._set_signal_strands()
        self._set_fuel_strands()
        self._set_threshold_base_strands()
        self._set_waste_strands()
        self._set_reporter_gates()
        self.strands = (list(self.signal_strands.values())
                        + list(self.fuel_strands.values())
                        + list(self.gate_base_strands.values())
                        + list(self.threshold_base_strands.values())
                        + list(self.waste_strands.values())
                        + list(self.reporter_base_strands.values()))

    def _add_input_gate_complex_constraint(self) -> None:
        """Adds input:gate complexes to self.constraint
        """
        input_gate_complexes = []
        for (input, gate), s in self.signal_strands.items():
            g = self.gate_base_strands[gate]
            input_gate_complexes.append((s, g))

        self.constraints.append(
            input_gate_complex_constraint(
                input_gate_complexes))

    def _add_gate_output_complex_constriant(self) -> None:
        """Adds gate:output complexes to self.constraint
        """
        gate_output_complexes: List[Tuple[dc.Strand, ...]] = []

        for (gate, _), s in self.signal_strands.items():
            if gate in self.gate_base_strands:
                g = self.gate_base_strands[gate]
                gate_output_complexes.append((s, g))

        for gate in self.fuel_strands:
            if gate in self.fuel_strands:
                f = self.fuel_strands[gate]
                g = self.gate_base_strands[gate]
                gate_output_complexes.append((f, g))

        self.constraints.append(
            gate_output_complex_constraint(
                gate_output_complexes
            )
        )

    def _add_threshold_complex_constraint(self) -> None:
        """Adds threshold complexes to self.constraint

        .. code-block:: none

                              S5       s5
                        14          2  10
                        |           |  ||
                       <=============--==]
                        |||||||||||||  ||
            [==--=====--=============--==>
             ||  |   |  |           |  ||
            15|  17  21 22          34 |36
              16                      35
             s2*   T*        S5*       s5*
        """
        threshold_complexes: List[Tuple[dc.Strand, ...]] = []
        for (_, gate), threshold_base_strand in self.threshold_base_strands.items():
            waste_strand = self.waste_strands[gate]
            threshold_complexes.append((waste_strand, threshold_base_strand))

        self.constraints.append(
            dc_complex_constraint(
                threshold_complexes,
                description="Threshold Complex",
                short_description="threshold"))

    def _add_threshold_waste_complex_constraint(self) -> None:
        """Adds threshold waste complexes to self.constraint

        .. code-block:: none


                 S2        s2    T          S5       s5
                           21
             34          22|20 19  15 14          2  10
             |           | ||  |   |  |           |  ||
            <=============-==--=====--=============--==]
                           ||  |||||  |||||||||||||  ||
                          [==--=====--=============--==>
                           ||  |   |  |           |  ||
                          35|  37  41 42          54 |56
                            36                       55
                           s2*   T*        S5*       s5*
        """
        threshold_waste_complexes: List[Tuple[dc.Strand, ...]] = []
        for (input, gate), threshold_base_strand in self.threshold_base_strands.items():
            signal_strand = self.signal_strands[(input, gate)]
            threshold_waste_complexes.append(
                (signal_strand, threshold_base_strand))

        self.constraints.append(
            dc_complex_constraint(
                threshold_waste_complexes,
                description="Threshold Waste Complex",
                short_description="threshold waste"))

    def _add_reporter_complex_constraint(self) -> None:
        """Adds reporter complexes to self.constraint

        .. code-block:: none

                          S6       s6
                    14          2  10
                    |           |  ||
                   <=============--==]
                    |||||||||||||  ||
            [=====--=============--==>
             |   |  |           |  ||
             15  19 20          32 |34
                                   33
               T*        S6*       s6*
        """
        reporter_complexes: List[Tuple[dc.Strand, ...]] = []
        for (_, gate), reporter_base_strand in self.reporter_base_strands.items():
            waste_strand = self.waste_strands[gate]
            reporter_complexes.append((waste_strand, reporter_base_strand))

        self.constraints.append(
            dc_complex_constraint(
                reporter_complexes,
                description="Reporter Complex",
                short_description="reporter"))

    def _add_reporter_waste_complex_constraint(self) -> None:
        """Adds reporter waste complexes to self.constraint

        .. code-block:: none

                  S5        s5    T          S6       s6
                            21
             34          22 |20 19  15 14          2  10
             |           |  ||  |   |  |           |  ||
            <=============--==--=====--=============--==]
                                |||||  |||||||||||||  ||
                               [=====--=============--==>
                                |   |  |           |  ||
                                35  39 40          52 |54
                                                      53
                                  T*        S6*       s6*
        """
        reporter_waste_complexes: List[Tuple[dc.Strand, ...]] = []
        for (input, gate), reporter_base_strand in self.reporter_base_strands.items():
            signal_strand = self.signal_strands[(input, gate)]
            reporter_waste_complexes.append(
                (signal_strand, reporter_base_strand))

        self.constraints.append(
            dc_complex_constraint(
                reporter_waste_complexes,
                description="Reporter Waste Complex",
                short_description="reporter waste"))

    def _set_constraints(self) -> None:
        """Sets self.constraints (self.strands must be set)
        """
        self._add_input_gate_complex_constraint()
        self._add_gate_output_complex_constriant()
        self._add_threshold_complex_constraint()
        self._add_threshold_waste_complex_constraint()
        self._add_reporter_complex_constraint()
        self._add_reporter_waste_complex_constraint()

    def __post_init__(self) -> None:
        self._set_strands()
        self._set_constraints()


@dataclass(frozen=True)
class SeesawGate:
    """Class for keeping track of seesaw gate and its input."""
    gate_name: int
    inputs: List[int]
    has_threshold: bool
    has_fuel: bool
    is_reporter: bool = False


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

    # Uncomment below for debugging:
    # for s in sorted(strands, key=lambda s: s.name):
    #     print(s)

    # for c in seesaw_circuit.constraints:
    #     print(c)
    # exit(0)

    design = dc.Design(
        strands=strands, complex_constraints=seesaw_circuit.constraints,
        strand_constraints=[
            strand_substring_constraint(
                strands, ILLEGAL_SUBSTRINGS)],)

    ds.search_for_dna_sequences(design=design,
                                # weigh_violations_equally=True,
                                report_delay=0.0,
                                # restart=True,
                                out_directory='output/square_root_circuit',
                                )


if __name__ == '__main__':
    main()
