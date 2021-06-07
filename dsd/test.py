from typing import Dict, List
import unittest
from dsd import constraints
from dsd.constraints import Domain, _get_base_pair_domain_endpoints_to_check, _get_implicitly_bound_domain_addresses, _exterior_base_type_of_domain_3p_end, _BasePairDomainEndpoint, Strand, DomainPool, BasePairType, StrandDomainAddress

_domain_pools: Dict[int, DomainPool] = {}


def clear_domains_interned() -> None:
    """Clear interned domains.
    """
    constraints._domains_interned.clear()


def assign_domain_pool_of_size(length: int) -> DomainPool:
    """Returns a DomainPool of given size

    :param length: Size of DomainPool
    :type length: int
    :return: DomainPool
    :rtype: DomainPool
    """
    if length in _domain_pools:
        return _domain_pools[length]
    else:
        new_domain_pool = DomainPool(f'POOL_{length}', length)
        _domain_pools[length] = new_domain_pool
        return new_domain_pool


def construct_strand(domain_names: List[str], domain_lengths: List[int]) -> Strand:
    """Constructs a strand with given domain names and domain lengths.

    :param domain_names: Names of the domain on the strand
    :type domain_names: List[str]
    :param domain_lengths: Lengths of the domain on the strand
    :type domain_lengths: List[int]
    :raises ValueError: If domain_names is not same size as domain_lengths
    :return: Strand
    :rtype: Strand
    """
    if len(domain_names) != len(domain_lengths):
        raise ValueError(f'domain_names and domain_lengths need to contain same'
                         f'number of elements but instead found that '
                         f'domain_names contained {len(domain_names)} names '
                         f'but domain_lengths contained {len(domain_lengths)} '
                         f'lengths')
    s: Strand = Strand(domain_names)
    for (i, length) in enumerate(domain_lengths):
        s.domains[i].pool = assign_domain_pool_of_size(length)
    return s


class TestExteriorBaseTypeOfDomain3PEnd(unittest.TestCase):
    def setUp(self):
        clear_domains_interned()

    def test_adjacent_to_exterior_base_pair_on_length_2_domain(self):
        """Test that base pair on domain of length two is properly classified as
        ADJACENT_TO_EXTERIOR_BASE_PAIR

        .. code-block:: none

                    this base pair type should be ADJACENT_TO_EXTERIOR_BASE_PAIR
                          |
                          V
                 b        a
          <=============--==]
           |||||||||||||  ||
          [=============--==>
                 b*       a*
        """
        top_strand = construct_strand(['a', 'b'], [2, 13])
        bot_strand = construct_strand(['b*', 'a*'], [13, 2])

        top_a = top_strand.address_of_domain(0)

        all_bound_domain_addresses = _get_implicitly_bound_domain_addresses([top_strand, bot_strand])

        self.assertEqual(
            _exterior_base_type_of_domain_3p_end(
                top_a, all_bound_domain_addresses),
            BasePairType.ADJACENT_TO_EXTERIOR_BASE_PAIR)

    @unittest.skip('MISMATCH detection has not been implemented')
    def test_mismatch(self):
        """Test MISMATCH is properly classified

        .. code-block:: none

               c    b    a
            <=====--=--=====]
             |||||     |||||
            [=====--=--=====>
               c*   d    a*
        """
        top_strand = construct_strand(['a', 'b', 'c'], [5, 1, 5])
        bot_strand = construct_strand(['c*', 'd', 'a*'], [5, 1, 5])

        top_a = top_strand.address_of_domain(0)

        all_bound_domain_addresses = _get_implicitly_bound_domain_addresses([top_strand, bot_strand])

        self.assertEqual(
            _exterior_base_type_of_domain_3p_end(
                top_a, all_bound_domain_addresses),
            BasePairType.MISMATCH)

    @unittest.skip('BULGE_LOOP_3P detection has not been implemented')
    def test_bulge_loop_3p(self):
        """Test BULGE_LOOP_3P is properly classified

        .. code-block:: none

               c    b    a
            <=====--=--=====]
             |||||     |||||
            [=====-----=====>
               c*        a*
        """
        top_strand = construct_strand(['a', 'b', 'c'], [5, 1, 5])
        bot_strand = construct_strand(['c*', 'a*'], [5, 5])

        all_bound_domain_addresses = _get_implicitly_bound_domain_addresses([top_strand, bot_strand])

        top_a = top_strand.address_of_domain(0)

        self.assertEqual(
            _exterior_base_type_of_domain_3p_end(
                top_a, all_bound_domain_addresses),
            BasePairType.BULGE_LOOP_3P)

    @unittest.skip('BULGE_LOOP_5P detection has not been implemented')
    def test_bulge_loop_5p(self):
        """Test BULGE_LOOP_5P is properly classified

        .. code-block:: none

               c    b    a
            <=====--=--=====]
             |||||     |||||
            [=====-----=====>
               c*        a*
        """
        top_strand = construct_strand(['a', 'c'], [5, 5])
        bot_strand = construct_strand(['c*', 'd', 'a*'], [5, 1, 5])

        all_bound_domain_addresses = _get_implicitly_bound_domain_addresses([top_strand, bot_strand])

        top_a = top_strand.address_of_domain(0)

        self.assertEqual(
            _exterior_base_type_of_domain_3p_end(
                top_a, all_bound_domain_addresses),
            BasePairType.BULGE_LOOP_5P)


class TestGetBasePairDomainEndpointsToCheck(unittest.TestCase):
    def test_seesaw_input_gate_complex(self):
        """Test endpoints for seesaw gate input:gate complex

        .. code-block:: none

          S{input}  s{input}  T       S{gate}    s{gate}
            |           |     |          |         |
        <=============--==--=====--=============--==]
                            |||||  |||||||||||||  ||
                           [=====--=============--==--=====>
                              |          |        |     |
                              T*      S{gate}* s{gate}* T*

                       21
         34          22|20 19  15 14          2  10
         |           | ||  |   |  |           |  ||
        <=============-==--=====--=============--==]
                           |||||  |||||||||||||  ||
                          [=====--=============--==--=====>
                           |   |  |           |  ||  |   |
                           35  39 40          52 |54 55  59
                                                 53

                    DANGLE_3P INTERIOR_TO_STRAND ADJACENT_TO_EXTERIOR_BASE_PAIR
                           |      |              |
        <=============-==--=====--=============--==]
                           |||||  |||||||||||||  ||
                          [=====--=============--==--=====>
                               |              |   |
                              INTERIOR_TO_STRAND  DANGLE_3P
        """
        ssg = Domain('ssg', assign_domain_pool_of_size(13), dependent=True)
        sg = Domain('sg', assign_domain_pool_of_size(2), dependent=True)
        Sg = Domain('Sg', assign_domain_pool_of_size(15), subdomains=[sg, ssg])
        T = Domain('T', assign_domain_pool_of_size(5))
        ssi = Domain('ssi', assign_domain_pool_of_size(13), dependent=True)
        si = Domain('si', assign_domain_pool_of_size(2), dependent=True)
        Si = Domain('Si', assign_domain_pool_of_size(15), subdomains=[si, ssi])
        input_strand = Strand(domains=[Sg, T, Si], starred_domain_indices=[])
        gate_base_strand = Strand(domains=[T, Sg, T], starred_domain_indices=[0, 1, 2])
        input_gate_complex = [input_strand, gate_base_strand]

        input_t = input_strand.address_of_domain(1)
        gate_base_t = gate_base_strand.address_of_domain(0)
        nonimplicit_base_pairs = [
            (input_t, gate_base_t)
        ]

        expected = set([
            _BasePairDomainEndpoint(
                domain1_5p_index=15, domain2_3p_index=39, domain_base_length=5,
                domain1_5p_domain2_base_pair_type=BasePairType.INTERIOR_TO_STRAND,
                domain1_3p_domain1_base_pair_type=BasePairType.DANGLE_3P),
            _BasePairDomainEndpoint(
                domain1_5p_index=2, domain2_3p_index=52, domain_base_length=13,
                domain1_5p_domain2_base_pair_type=BasePairType.INTERIOR_TO_STRAND,
                domain1_3p_domain1_base_pair_type=BasePairType.INTERIOR_TO_STRAND),
            _BasePairDomainEndpoint(
                domain1_5p_index=0, domain2_3p_index=54, domain_base_length=2,
                domain1_5p_domain2_base_pair_type=BasePairType.DANGLE_3P,
                domain1_3p_domain1_base_pair_type=BasePairType.ADJACENT_TO_EXTERIOR_BASE_PAIR), ])

        actual = _get_base_pair_domain_endpoints_to_check(input_gate_complex, nonimplicit_base_pairs)
        self.assertEqual(actual, expected)

    def test_seesaw_gate_output_complex(self):
        """Test endpoints for seesaw gate gate:output complex

        .. code-block:: none

                   S{gate}  s{gate}  T      S{output}    s{output}
                      |        |     |          |         |
               <=============--==--=====--=============--==]
                |||||||||||||  ||  |||||
        [=====--=============--==--=====>
           |          |        |     |
           T*      S{gate}* s{gate}* T*

                               21
                34          22 |20 19  15 14          2  10
                |           |  ||  |   |  |           |  ||
               <=============--==--=====--=============--==]
                |||||||||||||  ||  |||||
        [=====--=============--==--=====>
         |   |  |           |  ||  |   |
         35  39 40          52 |54 55  59
                               53

                DANGLE_5P      INTERIOR_TO_STRAND
                |              |   |
               <=============--==--=====--=============--==]
                |||||||||||||  ||  |||||
        [=====--=============--==--=====>
                            |   |      |
               INTERIOR_TO_STRAND      DANGLE_5P
        """
        output_strand = construct_strand(['so', 'So', 'T', 'sg', 'Sg'], [2, 13, 5, 2, 13])
        gate_base_strand = construct_strand(['T*', 'Sg*', 'sg*', 'T*'], [5, 13, 2, 5])
        gate_output_complex = [output_strand, gate_base_strand]

        output_t = output_strand.address_of_domain(2)
        gate_base_t = gate_base_strand.address_of_domain(3)
        nonimplicit_base_pairs = [
            (output_t, gate_base_t)
        ]

        expected = set([
            _BasePairDomainEndpoint(
                domain1_5p_index=22, domain2_3p_index=52, domain_base_length=13,
                domain1_5p_domain2_base_pair_type=BasePairType.INTERIOR_TO_STRAND,
                domain1_3p_domain1_base_pair_type=BasePairType.DANGLE_5P),
            _BasePairDomainEndpoint(
                domain1_5p_index=20, domain2_3p_index=54, domain_base_length=2,
                domain1_5p_domain2_base_pair_type=BasePairType.INTERIOR_TO_STRAND,
                domain1_3p_domain1_base_pair_type=BasePairType.INTERIOR_TO_STRAND),
            _BasePairDomainEndpoint(
                domain1_5p_index=15, domain2_3p_index=59, domain_base_length=5,
                domain1_5p_domain2_base_pair_type=BasePairType.DANGLE_5P,
                domain1_3p_domain1_base_pair_type=BasePairType.INTERIOR_TO_STRAND), ])

        actual = _get_base_pair_domain_endpoints_to_check(gate_output_complex, nonimplicit_base_pairs)
        self.assertEqual(actual, expected)

    def test_seesaw_threshold_complex(self):
        """Test endpoints for seesaw threshold complex

        .. code-block:: none

                                S{gate}  s{gate}
                            14          2  10
                            |           |  ||
                           <=============--==]
                            |||||||||||||  ||
                [==--=====--=============--==>
                 ||  |   |  |           |  ||
                 15|  17  21 22          34 |36
                 16                      35
            s{input}*  T*      S{gate}*    s{gate}*

                            DANGLE_5P      ADJACENT_TO_EXTERIOR_BASE_PAIR
                            |              |
                           <=============--==]
                            |||||||||||||  ||
                [==--=====--=============--==>
                                        |   |
                       INTERIOR_TO_STRAND   BLUNT_END
        """
        waste_strand = construct_strand(['sg', 'Sg'], [2, 13])
        threshold_base_strand = construct_strand(['si*', 'T*', 'Sg*', 'sg*'], [2, 5, 13, 2])
        threshold_complex = [waste_strand, threshold_base_strand]

        expected = set([
            _BasePairDomainEndpoint(
                domain1_5p_index=2, domain2_3p_index=34, domain_base_length=13,
                domain1_5p_domain2_base_pair_type=BasePairType.INTERIOR_TO_STRAND,
                domain1_3p_domain1_base_pair_type=BasePairType.DANGLE_5P),
            _BasePairDomainEndpoint(
                domain1_5p_index=0, domain2_3p_index=36, domain_base_length=2,
                domain1_5p_domain2_base_pair_type=BasePairType.BLUNT_END,
                domain1_3p_domain1_base_pair_type=BasePairType.ADJACENT_TO_EXTERIOR_BASE_PAIR), ])

        actual = _get_base_pair_domain_endpoints_to_check(threshold_complex)
        self.assertEqual(actual, expected)

    def test_seesaw_threshold_waste_complex(self):
        """Test endpoints for seesaw threshold waste complex

        .. code-block:: none

            S{input}  s{input}   T      S{gate}    s{gate}
                          21
             34          22|20 19  15 14          2  10
             |           | ||  |   |  |           |  ||
            <=============-==--=====--=============--==]
                           ||  |||||  |||||||||||||  ||
                          [==--=====--=============--==>
                           ||  |   |  |           |  ||
                          35|  37  41 42          54 |56
                            36                       55
                      s{input}*  T*      S{gate}*    s{gate}*

                   DANGLE_3P   INTERIOR_TO_STRAND    ADJACENT_TO_EXTERIOR_BASE_PAIR
                           |   |      |              |
            <=============-==--=====--=============--==]
                           ||  |||||  |||||||||||||  ||
                          [==--=====--=============--==>
                            |      |              |   |
                            |     INTERIOR_TO_STRAND  BLUNT_END
                            ADJACENT_TO_EXTERIOR_BASE_PAIR
        """
        ssg = Domain('ssg', assign_domain_pool_of_size(13), dependent=True)
        sg = Domain('sg', assign_domain_pool_of_size(2), dependent=True)
        Sg = Domain('Sg', assign_domain_pool_of_size(15), subdomains=[sg, ssg])
        T = Domain('T', assign_domain_pool_of_size(5))
        ssi = Domain('ssi', assign_domain_pool_of_size(13), dependent=True)
        si = Domain('si', assign_domain_pool_of_size(2), dependent=True)
        Si = Domain('Si', assign_domain_pool_of_size(15), subdomains=[si, ssi])

        input_strand = Strand(domains=[Sg, T, Si], starred_domain_indices=[])
        threshold_base_strand = Strand(domains=[si, T, Sg], starred_domain_indices=[0, 1, 2])
        threshold_waste_complex = [input_strand, threshold_base_strand]

        expected = set([
            _BasePairDomainEndpoint(
                domain1_5p_index=20, domain2_3p_index=36, domain_base_length=2,
                domain1_5p_domain2_base_pair_type=BasePairType.ADJACENT_TO_EXTERIOR_BASE_PAIR,
                domain1_3p_domain1_base_pair_type=BasePairType.DANGLE_3P),
            _BasePairDomainEndpoint(
                domain1_5p_index=15, domain2_3p_index=41, domain_base_length=5,
                domain1_5p_domain2_base_pair_type=BasePairType.INTERIOR_TO_STRAND,
                domain1_3p_domain1_base_pair_type=BasePairType.INTERIOR_TO_STRAND),
            _BasePairDomainEndpoint(
                domain1_5p_index=2, domain2_3p_index=54, domain_base_length=13,
                domain1_5p_domain2_base_pair_type=BasePairType.INTERIOR_TO_STRAND,
                domain1_3p_domain1_base_pair_type=BasePairType.INTERIOR_TO_STRAND),
            _BasePairDomainEndpoint(
                domain1_5p_index=0, domain2_3p_index=56, domain_base_length=2,
                domain1_5p_domain2_base_pair_type=BasePairType.BLUNT_END,
                domain1_3p_domain1_base_pair_type=BasePairType.ADJACENT_TO_EXTERIOR_BASE_PAIR), ])

        actual = _get_base_pair_domain_endpoints_to_check(threshold_waste_complex)
        self.assertEqual(actual, expected)

    def test_seesaw_reporter_complex(self):
        """Test endpoints for seesaw reporter complex

        .. code-block:: none

                      S{output}   s{output}
                    14          2  10
                    |           |  ||
                   <=============--==]
                    |||||||||||||  ||
            [=====--=============--==>
             |   |  |           |  ||
             15  19 20          32 |34
                                   33
               T*     S{output}*  s{output}*

                    DANGLE_5P      ADJACENT_TO_EXTERIOR_BASE_PAIR
                    |              |
                   <=============--==]
                    |||||||||||||  ||
            [=====--=============--==>
                                |   |
               INTERIOR_TO_STRAND   BLUNT_END
        """
        waste_strand = construct_strand(['so', 'So'], [2, 13])
        reporter_base_strand = construct_strand(['T*', 'So*', 'so*'], [5, 13, 2])
        reporter_complex = [waste_strand, reporter_base_strand]

        expected = set([
            _BasePairDomainEndpoint(
                domain1_5p_index=2, domain2_3p_index=32, domain_base_length=13,
                domain1_5p_domain2_base_pair_type=BasePairType.INTERIOR_TO_STRAND,
                domain1_3p_domain1_base_pair_type=BasePairType.DANGLE_5P),
            _BasePairDomainEndpoint(
                domain1_5p_index=0, domain2_3p_index=34, domain_base_length=2,
                domain1_5p_domain2_base_pair_type=BasePairType.BLUNT_END,
                domain1_3p_domain1_base_pair_type=BasePairType.ADJACENT_TO_EXTERIOR_BASE_PAIR), ])

        actual = _get_base_pair_domain_endpoints_to_check(reporter_complex)
        self.assertEqual(actual, expected)

    def test_seesaw_reporter_waste_complex(self):
        """Test endpoints for seesaw reporter waste complex

        .. code-block:: none

                S{gate}  s{gate}  T      S{output}    s{output}
                            21
             34          22 |20 19  15 14          2  10
             |           |  ||  |   |  |           |  ||
            <=============--==--=====--=============--==]
                                |||||  |||||||||||||  ||
                               [=====--=============--==>
                                |   |  |           |  ||
                                35  39 40          52 |54
                                                      53
                                  T*     S{output}*  s{output}*

                        DANGLE_3P INTERIOR_TO_STRAND  ADJACENT_TO_EXTERIOR_BASE_PAIR
                                |      |              |
            <=============--==--=====--=============--==]
                                |||||  |||||||||||||  ||
                               [=====--=============--==>
                                    |              |   |
                                   INTERIOR_TO_STRAND  BLUNT_END
        """
        output_strand = construct_strand(['so', 'So', 'T', 'sg', 'Sg'], [2, 13, 5, 2, 13])
        reporter_base_strand = construct_strand(['T*', 'So*', 'so*'], [5, 13, 2])
        reporter_waste_complex = [output_strand, reporter_base_strand]

        expected = set([
            _BasePairDomainEndpoint(
                domain1_5p_index=15, domain2_3p_index=39, domain_base_length=5,
                domain1_5p_domain2_base_pair_type=BasePairType.INTERIOR_TO_STRAND,
                domain1_3p_domain1_base_pair_type=BasePairType.DANGLE_3P),
            _BasePairDomainEndpoint(
                domain1_5p_index=2, domain2_3p_index=52, domain_base_length=13,
                domain1_5p_domain2_base_pair_type=BasePairType.INTERIOR_TO_STRAND,
                domain1_3p_domain1_base_pair_type=BasePairType.INTERIOR_TO_STRAND),
            _BasePairDomainEndpoint(
                domain1_5p_index=0, domain2_3p_index=54, domain_base_length=2,
                domain1_5p_domain2_base_pair_type=BasePairType.BLUNT_END,
                domain1_3p_domain1_base_pair_type=BasePairType.ADJACENT_TO_EXTERIOR_BASE_PAIR), ])

        actual = _get_base_pair_domain_endpoints_to_check(reporter_waste_complex)
        self.assertEqual(actual, expected)


class TestStrandDomainAddress(unittest.TestCase):
    def setUp(self):
        clear_domains_interned()
        self.strand = construct_strand(['a', 'b', 'c'], [10, 20, 30])
        self.addr = StrandDomainAddress(self.strand, 1)

    def test_init(self):
        self.assertEqual(self.addr.strand, self.strand)
        self.assertEqual(self.addr.domain_idx, 1)

    def test_neighbor_5p(self):
        self.assertEqual(self.addr.neighbor_5p(), StrandDomainAddress(self.strand, 0))

    def test_neighbor_5p_none(self):
        addr = StrandDomainAddress(self.strand, 0)
        self.assertEqual(addr.neighbor_5p(), None)

    def test_neighbor_3p(self):
        self.assertEqual(self.addr.neighbor_3p(), StrandDomainAddress(self.strand, 2))

    def test_neighbor_3p_none(self):
        addr = StrandDomainAddress(self.strand, 2)
        self.assertEqual(addr.neighbor_3p(), None)

    def test_domain(self):
        self.assertEqual(self.addr.domain(), self.strand.domains[1])


class TestSubdomains(unittest.TestCase):
    def test_init(self):
        """
        Test constructing a domain with subdomains

        .. code-block:: none

                       a
            <====================]

                 b      c      d      e
            <--=====--=====--=====--=====]
        """
        b = Domain('b', assign_domain_pool_of_size(5), dependent=True)
        c = Domain('c', assign_domain_pool_of_size(5), dependent=True)
        d = Domain('d', assign_domain_pool_of_size(5), dependent=True)
        e = Domain('e', assign_domain_pool_of_size(5), dependent=True)

        a = Domain('a', assign_domain_pool_of_size(20), subdomains=[b, c, d, e])
        self.assertListEqual([b, c, d, e], a.subdomains)
        self.assertEqual(a, b.parent)
        self.assertEqual(a, c.parent)
        self.assertEqual(a, d.parent)
        self.assertEqual(a, e.parent)

    def test_construct_fixed_domain_with_fixed_subdomains(self):
        """
        Test constructing a fixed domain with fixed subdomains

        .. code-block:: none

               [a]
               / \
             [b] [c]
        """
        b = Domain('b', assign_domain_pool_of_size(5), fixed=True)
        c = Domain('c', assign_domain_pool_of_size(4), fixed=True)

        a = Domain('a', assign_domain_pool_of_size(9), fixed=True, subdomains=[b, c])
        self.assertTrue(a.fixed)

    def test_construct_unfixed_domain_with_unfixed_subdomain(self):
        """
        Test constructing an unfixed domain with a unfixed subdomain should
        set domain's fixed to False.

        .. code-block:: none

                a
               / \
              b  [c]
        """
        b = Domain('b', assign_domain_pool_of_size(5), fixed=False)
        c = Domain('c', assign_domain_pool_of_size(4), fixed=True)

        a = Domain('a', assign_domain_pool_of_size(9), subdomains=[b, c], fixed=False)
        self.assertFalse(a.fixed)

    def test_error_construct_fixed_domain_with_unfixed_subdomain(self):
        """
        Test that constructing a fixed domain with a unfixed subdomain should
        raise ValueError.

        .. code-block:: none

               [a]
               / \
              b  [c]
        """
        b = Domain('b', assign_domain_pool_of_size(5), fixed=False)
        c = Domain('c', assign_domain_pool_of_size(4), fixed=True)

        self.assertRaises(ValueError, Domain, 'a', assign_domain_pool_of_size(9), fixed=True, subdomains=[b, c])

    def test_error_constructed_unfixed_domain_with_fixed_subdomains(self):
        """
        Test that constructing a domain by setting fixed to False when all subdomains
        are fixed should raise ValueError

        .. code-block:: none

                a
               / \
             [b] [c]
        """
        b = Domain('b', assign_domain_pool_of_size(5), fixed=True)
        c = Domain('c', assign_domain_pool_of_size(4), fixed=True)

        self.assertRaises(ValueError, Domain, 'a', assign_domain_pool_of_size(9), fixed=False, subdomains=[b, c])

    def test_construst_strand(self):
        """
        Test strand construction with nested subdomains

        .. code-block:: none

                  a
                /   \
               b     C
              / \   / \
             E   F g   h
        """
        E = Domain('e', assign_domain_pool_of_size(5), dependent=False)
        F = Domain('f', assign_domain_pool_of_size(5), dependent=False)
        g = Domain('g', assign_domain_pool_of_size(5), dependent=True)
        h = Domain('h', assign_domain_pool_of_size(5), dependent=True)

        b = Domain('b', assign_domain_pool_of_size(10), dependent=True, subdomains=[E, F])
        C = Domain('C', assign_domain_pool_of_size(10), dependent=False, subdomains=[g, h])

        a = Domain('a', assign_domain_pool_of_size(20), dependent=True, subdomains=[b, C])

        # Test that constructor runs without errors
        strand = Strand(domains=[a], starred_domain_indices=[])
        self.assertEqual(strand.domains[0], a)

    def test_error_strand_with_unassignable_subsequence(self):
        """
        Test that constructing a strand with an unassignable subsequence raises
        a ValueError.

        This happens due to when no independent domain assigns a sequence for a
        portion of a strand

        .. code-block:: none

                  a
                /   \
               b     C
              / \   / \
             e   f g   h
        """
        e = Domain('e', assign_domain_pool_of_size(5), dependent=True)
        f = Domain('f', assign_domain_pool_of_size(5), dependent=True)
        g = Domain('g', assign_domain_pool_of_size(5), dependent=True)
        h = Domain('h', assign_domain_pool_of_size(5), dependent=True)

        b = Domain('b', assign_domain_pool_of_size(10), dependent=True, subdomains=[e, f])
        C = Domain('C', assign_domain_pool_of_size(10), dependent=False, subdomains=[g, h])

        a = Domain('a', assign_domain_pool_of_size(20), dependent=True, subdomains=[b, C])

        self.assertRaises(ValueError, Strand, domains=[a], starred_domain_indices=[])

    def test_error_strand_with_redundant_independence(self):
        """
        Test that constructing a strand with an redundant indepndence in subdomain
        graph raises a ValueError.

        Below, in the path from F to a, two independent subdomains are found: F and B

        .. code-block:: none

                  a
                /   \
               B     C
              / \   / \
             e   F g   h
        """
        e = Domain('e', assign_domain_pool_of_size(5), dependent=True)
        F = Domain('F', assign_domain_pool_of_size(5), dependent=False)
        g = Domain('g', assign_domain_pool_of_size(5), dependent=True)
        h = Domain('h', assign_domain_pool_of_size(5), dependent=True)

        B = Domain('B', assign_domain_pool_of_size(10), dependent=False, subdomains=[e, F])
        C = Domain('C', assign_domain_pool_of_size(10), dependent=False, subdomains=[g, h])

        a = Domain('a', assign_domain_pool_of_size(20), dependent=True, subdomains=[B, C])

        self.assertRaises(ValueError, Strand, domains=[a], starred_domain_indices=[])

    def test_error_cycle(self):
        """
        Test that constructing a domain with a cycle in its subdomain graph
        rasies a ValueError.

        .. code-block:: none

            a
            |
            b
            |
            a
        """
        a = Domain('a', assign_domain_pool_of_size(5), dependent=True)
        b = Domain('b', assign_domain_pool_of_size(5), subdomains=[a], dependent=True)
        a.subdomains = [b]

        self.assertRaises(ValueError, Strand, domains=[a], starred_domain_indices=[])

    def sample_nested_domains(self) -> Dict[str, Domain]:
        """Returns domains with the following subdomain hierarchy:

        .. code-block:: none

                  a
                /   \
               b     C
              / \   / \
             E   F g   h

        :return: Map of domain name to domain object.
        :rtype: Dict[str, Domain]
        """
        E: Domain = Domain('E', assign_domain_pool_of_size(5), dependent=False)
        F: Domain = Domain('F', assign_domain_pool_of_size(6), dependent=False)
        g: Domain = Domain('g', assign_domain_pool_of_size(7), dependent=True)
        h: Domain = Domain('h', assign_domain_pool_of_size(8), dependent=True)

        b: Domain = Domain('b', assign_domain_pool_of_size(11), dependent=True, subdomains=[E, F])
        C: Domain = Domain('C', assign_domain_pool_of_size(15), dependent=False, subdomains=[g, h])

        a: Domain = Domain('a', assign_domain_pool_of_size(26), dependent=True, subdomains=[b, C])
        return {domain.name: domain for domain in [a, b, C, E, F, g, h]}

    def test_assign_dna_sequence_to_parent(self):
        """
        Test assigning dna sequence to parent (a) and propagating it downwards

        .. code-block:: none

                  a
                /   \
               b     C
              / \   / \
             E   F g   h
        """
        domains = self.sample_nested_domains()
        sequence = 'CATAGCTTTCTTGTTCTGATCGGAAC'
        a = domains['a']
        a.sequence = sequence
        self.assertEqual(sequence, a.sequence)
        self.assertEqual(sequence[0: 11], domains['b'].sequence)
        self.assertEqual(sequence[11:], domains['C'].sequence)
        self.assertEqual(sequence[0:5], domains['E'].sequence)
        self.assertEqual(sequence[5:11], domains['F'].sequence)
        self.assertEqual(sequence[11:18], domains['g'].sequence)
        self.assertEqual(sequence[18:], domains['h'].sequence)

    def test_error_assign_dna_sequence_to_parent_with_incorrect_size_subdomain(self):
        """
        Test error is raised if assigning dna sequence to domain when subdomains
        length do not add up to domain length.

        .. code-block:: none

                  a
                /   \
               B     C
        """
        B: Domain = Domain('B', assign_domain_pool_of_size(10), dependent=False)
        C: Domain = Domain('C', assign_domain_pool_of_size(20), dependent=False)

        a: Domain = Domain('a', assign_domain_pool_of_size(15), dependent=True, subdomains=[B, C])
        with self.assertRaises(ValueError):
            a.sequence = 'A' * 15

    def test_construct_strand_using_dependent_subdomain(self) -> None:
        """Test constructing a strand using a dependent subdomain (not parent)

        .. code-block:: none

                  a
                /   \
               b     C
              / \   / \
             E   F g   h

        Test constructing a strand using g.
        """
        g = self.sample_nested_domains()['g']
        Strand(domains=[g], starred_domain_indices=[])


if __name__ == '__main__':
    unittest.main()
