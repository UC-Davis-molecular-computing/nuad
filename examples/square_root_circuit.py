from dataclasses import dataclass, field
from math import ceil, floor
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import itertools

import nuad.search as ns  # type: ignore
import nuad.constraints as nc

# TODO: Go over each constraint, using NUPACK
#   - check pfunc for each strand
#   - check every pair of signal strands not in same complex (use binding4)
#   - check every complex is well-formed (print base-pair probabilties and MFE)


# Constants

# Constants -- Toehold domain
SIGNAL_DOMAIN_LENGTH = 15
EXTENDED_TOEHOLD_LENGTH = 2
TOEHOLD_LENGTH = 5

# Constants -- Illegal DNA Base sequences
ILLEGAL_SUBSTRINGS_FOUR = ['G' * 4, 'C' * 4]
ILLEGAL_SUBSTRINGS_FIVE = ['A' * 5, 'T' * 5]
ILLEGAL_SUBSTRINGS = ILLEGAL_SUBSTRINGS_FOUR + ILLEGAL_SUBSTRINGS_FIVE

# NumpyConstraints
three_letter_code_constraint = nc.RestrictBasesConstraint(('A', 'C', 'T'))
no_gggg_constraint = nc.ForbiddenSubstringConstraint(ILLEGAL_SUBSTRINGS_FOUR)
no_aaaaa_constraint = nc.ForbiddenSubstringConstraint(ILLEGAL_SUBSTRINGS_FIVE)
c_content_constraint = nc.BaseCountConstraint('C', floor(0.7 * SIGNAL_DOMAIN_LENGTH),
                                              ceil(0.3 * SIGNAL_DOMAIN_LENGTH))

# Domain pools
SUBDOMAIN_SS_POOL: nc.DomainPool = nc.DomainPool(f'SUBDOMAIN_SS_POOL',
                                                 SIGNAL_DOMAIN_LENGTH - EXTENDED_TOEHOLD_LENGTH)
SUBDOMAIN_S_POOL: nc.DomainPool = nc.DomainPool(f'SUBDOMAIN_S_POOL', EXTENDED_TOEHOLD_LENGTH)
TOEHOLD_DOMAIN_POOL: nc.DomainPool = nc.DomainPool(
    name='TOEHOLD_DOMAIN_POOL', length=TOEHOLD_LENGTH, numpy_constraints=[three_letter_code_constraint])

SIGNAL_DOMAIN_POOL: nc.DomainPool = nc.DomainPool(
    name='SIGNAL_DOMAIN_POOL', length=SIGNAL_DOMAIN_LENGTH,
    numpy_constraints=[three_letter_code_constraint, c_content_constraint, no_aaaaa_constraint,
                       no_gggg_constraint])

# Alias
dc_complex_constraint = nc.nupack_complex_base_pair_probability_constraint

# Stores all domains used in design
TOEHOLD_DOMAIN: nc.Domain = nc.Domain('T', pool=TOEHOLD_DOMAIN_POOL)
FUEL_DOMAIN: nc.Domain = nc.Domain('fuel', sequence='CATTTTTTTTTTTCA', fixed=True)
recognition_domains_and_subdomains: Dict[str, nc.Domain] = {}
recognition_domains: Set[nc.Domain] = set()


def get_signal_domain(gate: Union[int, str]) -> nc.Domain:
    """Returns a signal domain with S{gate} and stores it to all_domains for
    future use.

    :param gate: Gate.
    :type gate: str
    :return: Domain
    :rtype: Domain
    """
    if f'S{gate}' not in recognition_domains_and_subdomains:
        d_13: nc.Domain = nc.Domain(f'ss{gate}', pool=SUBDOMAIN_SS_POOL, dependent=True)
        d_2: nc.Domain = nc.Domain(f's{gate}', pool=SUBDOMAIN_S_POOL, dependent=True)
        d: nc.Domain = nc.Domain(f'S{gate}', pool=SIGNAL_DOMAIN_POOL, dependent=False, subdomains=[d_2, d_13])

        recognition_domains_and_subdomains[f'ss{gate}'] = d_13
        recognition_domains_and_subdomains[f's{gate}'] = d_2
        recognition_domains_and_subdomains[f'S{gate}'] = d
        assert d not in recognition_domains
        recognition_domains.add(d)

    return recognition_domains_and_subdomains[f'S{gate}']


def set_domain_pool(domain: nc.Domain, domain_pool: nc.DomainPool) -> None:
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
        gate5p: Union[int, str]) -> nc.Strand:
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
    :return: Strand
    :rtype: dc.Strand
    """
    d3p = get_signal_domain(gate3p)
    d5p = get_signal_domain(gate5p)

    name = f'signal_{gate3p}_{gate5p}'
    return nc.Strand(domains=[d5p, TOEHOLD_DOMAIN, d3p], starred_domain_indices=[], name=name)


def fuel_strand(gate: int) -> nc.Strand:
    """Returns a fuel strand with recognition domain `gate`.

    .. code-block:: none

         ss{gate}    s{gate}  T         ssf        sf
            |           |     |          |         |
        <=============--==--=====--=============--==]

    :param gate: The name of the gate that this fuel strand will fuel.
    :type gate: int
    :return: Fuel strand
    :rtype: dc.Strand
    """
    d3p = get_signal_domain(gate)
    fuel = FUEL_DOMAIN

    name = f'fuel_{gate}'
    return nc.Strand(domains=[fuel, TOEHOLD_DOMAIN, d3p], starred_domain_indices=[], name=name)


def gate_base_strand(gate: int) -> nc.Strand:
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
    s: nc.Strand = nc.Strand(
        domains=[TOEHOLD_DOMAIN, d, TOEHOLD_DOMAIN],
        starred_domain_indices=[0, 1, 2],
        name=f'gate_base_{gate}')
    return s


def threshold_bottom_strand(input_: int, gate: int) -> nc.Strand:
    """Returns a threshold bottom strand for seesaw gate labeled `gate` that
    thresholds `input`

    .. code-block:: none

     s{input}* T*      ss{gate}*   s{gate}*
         |     |          |        |
        [==--=====--=============--==>

    :param input_: Name of input that is being thresholded
    :type input_: int
    :param gate: Name of gate
    :type gate: int
    :return: Threshold bottom strand
    :rtype: dc.Strand
    """
    # Note, this assumes that this input signal domain has already been built
    d_input_sub = recognition_domains_and_subdomains[f's{input_}']
    d_gate = get_signal_domain(gate)

    s: nc.Strand = nc.Strand(
        domains=[d_input_sub, TOEHOLD_DOMAIN, d_gate],
        starred_domain_indices=[0, 1, 2],
        name=f'threshold_bottom_{input_}_{gate}')
    return s


def threshold_top_strand(gate: int) -> nc.Strand:
    """Returns a waste strand for a thresholding reaction involving
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
    s: nc.Strand = nc.Strand(
        domains=[get_signal_domain(gate)],
        starred_domain_indices=[],
        name=f'threshold_top_{gate}')
    return s


def reporter_top_strand(gate: int) -> nc.Strand:
    """Returns a waste strand for a reporting reaction involving
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
    s: nc.Strand = nc.Strand(domains=[get_signal_domain(gate)], starred_domain_indices=[],
                             name=f'reporter_top_{gate}')
    return s


def reporter_bottom_strand(gate) -> nc.Strand:
    """Returns a reporter bottom strand for seesaw gate labeled `gate`

    .. code-block:: none

           T*     ss{gate}*   s{gate}*
           |          |        |
        [=====--=============--==>

    :param gate: Name of gate
    :type gate: [type]
    :return: Reporter bottom strand
    :rtype: dc.Strand
    """
    s: nc.Strand = nc.Strand(
        domains=[TOEHOLD_DOMAIN, get_signal_domain(gate)],
        starred_domain_indices=[0, 1],
        name=f'reporter_bottom_{gate}')
    return s


def input_gate_complex_constraint(
        input_gate_complexes: List[nc.Complex]) -> nc.ComplexConstraint:
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
    addr_t = template_top_strand.address_of_first_domain_occurence('T')
    addr_t_star = template_bot_strand.address_of_first_domain_occurence('T*')
    return dc_complex_constraint(
        strand_complexes=input_gate_complexes,
        nonimplicit_base_pairs=[(addr_t, addr_t_star)],
        description="input:gate Complex",
        short_description="input:gate")


def gate_output_complex_constraint(
        gate_output_complexes: List[nc.Complex],
        base_pair_prob_by_type: Optional[Dict[nc.BasePairType, float]] = None,
        description: str = 'gate:output') -> nc.ComplexConstraint:
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

    :param gate_output_complexes: List of gate:output complexes
    :type gate_output_complexes: List[Tuple[dc.Strand, ...]]
    :param base_pair_prob_by_type: probabilities to assign to each type of base pair
    :type base_pair_prob_by_type: Optional[Dict[dc.BasePairType, float]]
    :param description: description of complex
    :type description: str
    :return: A complex constraint on the base-pairing probabilities
    :rtype: dc.ComplexConstraint
    """
    assert gate_output_complexes
    template_complex = gate_output_complexes[0]
    assert len(template_complex) == 2
    template_top_strand = template_complex[0]
    template_bot_strand = template_complex[1]
    addr_t = template_top_strand.address_of_first_domain_occurence('T')
    addr_t_star = template_bot_strand.address_of_last_domain_occurence('T*')
    return dc_complex_constraint(
        strand_complexes=gate_output_complexes,
        nonimplicit_base_pairs=[(addr_t, addr_t_star)],
        base_pair_prob_by_type=base_pair_prob_by_type, description=f"{description} Complex",
        short_description=f"{description}"
    )


def base_difference_constraint(domains: Iterable[nc.Domain]) -> nc.DomainPairConstraint:
    """
    For any two sequences in the pool, we require at least 30% of bases are
    different and the longest run of matches is at most 35% of the domain length
    :param domains: Domains to compare
    :type domains: Iterable[dc.Domain]
    :return: DomainPairConstraint
    :rtype: dc.DomainPairConstraint
    """

    def evaluate(seqs: Tuple[str, ...], domain_pair: Optional[nc.DomainPair]) \
            -> Tuple[float, str]:
        seq1, seq2 = seqs
        if domain_pair is not None:
            domain1, domain2 = domain_pair.domain1, domain_pair.domain2
        else:
            domain1, domain2 = None, None

        # evaluate
        num_of_matches = 0
        run_of_matches = 0
        assert len(seq1) == len(seq2)
        length = len(seq1)
        run_of_matches_limit = 0.35 * length
        num_of_matches_limit = 0.7 * length
        result = 0
        for i in range(length):
            if seq1[i] == seq2[i]:
                num_of_matches += 1
                run_of_matches += 1
            else:
                run_of_matches = 0
            if num_of_matches > num_of_matches_limit or run_of_matches > run_of_matches_limit:
                result = 100
                break

        # summary

        if result > 0:
            summary = (f'Too many matches between domains {domain1} and {domain2}\n'
                       f'\t{domain1}: {domain1.sequence}\n'
                       f'\t{domain2}: {domain2.sequence}\n')
        else:
            summary = (f'Sufficient difference between domains {domain1} and {domain2}\n'
                       f'\t{domain1}: {domain1.sequence}\n'
                       f'\t{domain2}: {domain2.sequence}\n')

        return result, summary

    pairs = itertools.combinations(domains, 2)

    return nc.DomainPairConstraint(pairs=tuple(pairs), evaluate=evaluate,
                                   description='base difference constraint',
                                   short_description='base difference constraint')


def strand_substring_constraint(
        strands: List[nc.Strand],
        substrings: List[str]) -> nc.StrandConstraint:
    """Returns a strand constraint that restricts the substrings in the strand
    sequence

    :param strands: Strands to apply constraint on
    :type strands: List[dc.Strand]
    :param substrings: Substrings to disallow
    :type substrings: List[str]
    :return: [description]
    :rtype: dc.StrandConstraint
    """

    def violated(seq: str):
        for substring in substrings:
            if substring in seq:
                return True
        return False

    def evaluate(seqs: Tuple[str, ...], strand: Optional[nc.Strand]) -> Tuple[float, str]:
        seq = seqs[0]
        if violated(seq):
            violation_str = '** violation**'
            score = 100
        else:
            violation_str = ''
            score = 0
        return score, f"{strand.name}: {strand.sequence()}{violation_str}"

    return nc.StrandConstraint(description="Strand Substring Constraint",
                               short_description="Strand Substring Constraint",
                               evaluate=evaluate,
                               strands=tuple(strands))


@dataclass
class SeesawCircuit:
    """Class for keeping track of a seesaw circuit and its DNA representation.
    """
    seesaw_gates: List['SeesawGate']
    strands: List[nc.Strand] = field(init=False, default_factory=list)
    constraints: List[nc.ComplexConstraint] = field(
        init=False, default_factory=list)

    signal_strands: Dict[Tuple[int, int], nc.Strand] = field(
        init=False, default_factory=dict)
    fuel_strands: Dict[int, nc.Strand] = field(
        init=False, default_factory=dict)
    gate_base_strands: Dict[int, nc.Strand] = field(
        init=False, default_factory=dict)
    threshold_top_strands: Dict[int, nc.Strand] = field(
        init=False, default_factory=dict)
    threshold_bottom_strands: Dict[Tuple[int, int], nc.Strand] = field(
        init=False, default_factory=dict)
    reporter_top_strands: Dict[int, nc.Strand] = field(
        init=False, default_factory=dict)
    reporter_bottom_strands: Dict[Tuple[int, int], nc.Strand] = field(
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
            if not seesaw_gate.is_reporter:
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
            for input_ in seesaw_gate.inputs:
                assert (input_, gate_name) not in input_gate_pairs
                input_gate_pairs.add((input_, gate_name))

        self.signal_strands = {(input_, gate): signal_strand(input_, gate)
                               for input_, gate in input_gate_pairs}

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

    def _set_threshold_bottom_strands(self) -> None:
        """Sets self.threshold_bottom_strands

        :raises ValueError: If duplicate gate name found
        """
        # Set of all input, gate pairs with threshold
        input_gate_pairs_with_threshold: Set[Tuple[int, int]] = set()
        for seesaw_gate in self.seesaw_gates:
            if seesaw_gate.has_threshold and not seesaw_gate.is_reporter:
                gate_name = seesaw_gate.gate_name
                if gate_name in input_gate_pairs_with_threshold:
                    raise ValueError(
                        f'Invalid seesaw circuit: '
                        'Multiple gates labeled {gate_name} found')
                for input_ in seesaw_gate.inputs:
                    assert (input_, gate_name) not in input_gate_pairs_with_threshold
                    input_gate_pairs_with_threshold.add((input_, gate_name))

        self.threshold_bottom_strands = {(input_, gate): threshold_bottom_strand(
            input_, gate) for input_, gate in input_gate_pairs_with_threshold}

    def _set_threshold_top_strands(self) -> None:
        """Sets self.threshold_top_strands

        :raises ValueError: If duplicate gate name found
        """
        # Set of all gates with threshold
        gates_with_threshold_but_not_reporter: Set[int] = set()

        for seesaw_gate in self.seesaw_gates:
            if seesaw_gate.has_threshold and not seesaw_gate.is_reporter:
                gate_name = seesaw_gate.gate_name
                if gate_name in gates_with_threshold_but_not_reporter:
                    raise ValueError(
                        f'Invalid seesaw circuit: '
                        'Multiple gates labeled {gate_name} found')
                gates_with_threshold_but_not_reporter.add(gate_name)

        self.threshold_top_strands = {gate: threshold_top_strand(gate)
                                      for gate in gates_with_threshold_but_not_reporter}

    def _set_reporter_top_strands(self) -> None:
        """Sets self.reporter_top_strands

        :raises ValueError: If duplicate gate name found
        """
        # Set of all gates that are reporter
        gates_that_are_reporter: Set[int] = set()

        for seesaw_gate in self.seesaw_gates:
            if seesaw_gate.is_reporter:
                gate_name = seesaw_gate.gate_name
                if gate_name in gates_that_are_reporter:
                    raise ValueError(
                        f'Invalid seesaw circuit: '
                        'Multiple gates labeled {gate_name} found')
                gates_that_are_reporter.add(gate_name)

        self.reporter_top_strands = {gate: reporter_top_strand(gate)
                                     for gate in gates_that_are_reporter}

    def _set_reporter_bottom_strands(self) -> None:
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

        self.reporter_bottom_strands = {(input_, gate): reporter_bottom_strand(gate)
                                        for input_, gate in reporter_gates}

    def _set_strands(self) -> None:
        """Sets self.strands
        """
        self._set_gate_base_strands()
        self._set_signal_strands()
        self._set_fuel_strands()
        self._set_threshold_bottom_strands()
        self._set_threshold_top_strands()
        self._set_reporter_bottom_strands()
        self._set_reporter_top_strands()
        self.strands = (list(self.signal_strands.values())
                        + list(self.fuel_strands.values())
                        + list(self.gate_base_strands.values())
                        + list(self.threshold_bottom_strands.values())
                        + list(self.threshold_top_strands.values())
                        + list(self.reporter_bottom_strands.values())
                        + list(self.reporter_top_strands.values()))

    def _add_input_gate_complex_constraint(self) -> None:
        """Adds input:gate complexes to self.constraint
        """
        input_gate_strands = []
        for (input_, gate), s in self.signal_strands.items():
            if gate in self.gate_base_strands:
                g = self.gate_base_strands[gate]
                input_gate_strands.append((s, g))

        input_gate_complexes = [nc.Complex(*strands) for strands in input_gate_strands]
        self.constraints.append(
            input_gate_complex_constraint(input_gate_complexes))

    def _add_gate_output_complex_constriant(self) -> None:
        """Adds gate:output complexes to self.constraint
        """
        gate_output_strands: List[Tuple[nc.Strand, ...]] = []

        for (gate, _), s in self.signal_strands.items():
            if gate in self.gate_base_strands:
                g = self.gate_base_strands[gate]
                gate_output_strands.append((s, g))

        gate_output_complexes = [nc.Complex(*strands) for strands in gate_output_strands]

        self.constraints.append(
            gate_output_complex_constraint(
                gate_output_complexes
            )
        )

    def _add_gate_fuel_complex_constriant(self) -> None:
        """Adds gate:fuel complexes to self.constraint
        """
        gate_output_strands: List[Tuple[nc.Strand, ...]] = []

        for gate in self.fuel_strands:
            if gate in self.fuel_strands:
                f = self.fuel_strands[gate]
                g = self.gate_base_strands[gate]
                gate_output_strands.append((f, g))

        gate_output_complexes = [nc.Complex(*strands) for strands in gate_output_strands]

        # TODO: Make it so that only specific base pairs have lower threshold (such as base index 1)
        #       which is an A that can bind to any T but it doesn't matter which.
        self.constraints.append(
            gate_output_complex_constraint(
                gate_output_complexes,
                base_pair_prob_by_type={nc.BasePairType.UNPAIRED: 0.8},
                description='gate:fuel'
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
        threshold_strands: List[Tuple[nc.Strand, ...]] = []
        for (_, gate), thres_bottom_strand in self.threshold_bottom_strands.items():
            waste_strand = self.threshold_top_strands[gate]
            threshold_strands.append((waste_strand, thres_bottom_strand))

        threshold_complexes = [nc.Complex(*strands) for strands in threshold_strands]

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
        threshold_waste_strands: List[Tuple[nc.Strand, ...]] = []
        for (input_, gate), thres_bottom_strand in self.threshold_bottom_strands.items():
            sig_strand = self.signal_strands[(input_, gate)]
            threshold_waste_strands.append(
                (sig_strand, thres_bottom_strand))

        threshold_waste_complexes = [nc.Complex(*strands) for strands in threshold_waste_strands]

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
        reporter_strands: List[Tuple[nc.Strand, ...]] = []
        for (_, gate), reporter_bottom_strand_ in self.reporter_bottom_strands.items():
            waste_strand = self.reporter_top_strands[gate]
            reporter_strands.append((waste_strand, reporter_bottom_strand_))

        reporter_complexes = [nc.Complex(*strands) for strands in reporter_strands]

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
        reporter_waste_strands: List[Tuple[nc.Strand, ...]] = []
        for (input_, gate), reporter_bottom_strand_ in self.reporter_bottom_strands.items():
            signal_strand_ = self.signal_strands[(input_, gate)]
            reporter_waste_strands.append(
                (signal_strand_, reporter_bottom_strand_))

        reporter_waste_complexes = [nc.Complex(*strands) for strands in reporter_waste_strands]

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
        self._add_gate_fuel_complex_constriant()
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
    return integrating_gate, amplifying_gate


def reporter_gate(gate_name: int, input_: int) -> SeesawGate:
    """Returns a SeesawGate for a reporter

    :param gate_name: Name of the reporter
    :type gate_name: int
    :param input_: Input
    :type input_: int
    :return: SeesawGate for a reporter
    :rtype: SeesawGate
    """
    return SeesawGate(
        gate_name=gate_name, inputs=[input_],
        has_threshold=True, has_fuel=False, is_reporter=True)


def input_gate(gate_name: int, input_: int) -> SeesawGate:
    """Returns a SeesawGate for an input

    :param gate_name: Name of the gate
    :type gate_name: int
    :param input_: Input
    :type input_: int
    :return: SeesawGate
    :rtype: SeesawGate
    """
    return SeesawGate(
        gate_name=gate_name, inputs=[input_],
        has_threshold=True, has_fuel=True)


def main() -> None:
    seesaw_gates = [
        *and_or_gate(integrating_gate_name=10,
                     amplifying_gate_name=1, inputs=[21, 27]),
        *and_or_gate(integrating_gate_name=53,
                     amplifying_gate_name=5, inputs=[18, 22]),
        reporter_gate(gate_name=6, input_=5),
        *and_or_gate(integrating_gate_name=20,
                     amplifying_gate_name=8, inputs=[35, 38]),
        *and_or_gate(integrating_gate_name=26,
                     amplifying_gate_name=13, inputs=[33, 37]),
        *and_or_gate(integrating_gate_name=34,
                     amplifying_gate_name=18, inputs=[28, 33, 37]),
        *and_or_gate(integrating_gate_name=36,
                     amplifying_gate_name=21, inputs=[29, 35, 38]),
        reporter_gate(gate_name=23, input_=1),
        reporter_gate(gate_name=24, input_=13),
        reporter_gate(gate_name=25, input_=8),
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
        input_gate(gate_name=33, input_=49),
        input_gate(gate_name=35, input_=50),
        input_gate(gate_name=37, input_=51),
        input_gate(gate_name=38, input_=52),
    ]

    seesaw_circuit = SeesawCircuit(seesaw_gates=seesaw_gates)
    strands = seesaw_circuit.strands
    non_fuel_strands = []
    for s in strands:
        if FUEL_DOMAIN not in s.domains:
            non_fuel_strands.append(s)

    # Uncomment below for debugging:
    # for s in sorted(strands, key=lambda s: s.name):
    #     print(s)

    # for c in seesaw_circuit.constraints:
    #     print(c)
    # exit(0)

    constraints: List[nc.Constraint] = [base_difference_constraint(recognition_domains),
                                        strand_substring_constraint(non_fuel_strands, ILLEGAL_SUBSTRINGS)]
    constraints.extend(seesaw_circuit.constraints)  # make mypy happy about the generics with List
    design = nc.Design(strands=strands)
    params = ns.SearchParameters(constraints=constraints,
                                 out_directory='output/square_root_circuit',
                                 # weigh_violations_equally=True,
                                 # restart=True
                                 )

    ns.search_for_sequences(design, params)


if __name__ == '__main__':
    main()
