from typing import Dict, List
import unittest
from dsd import constraints
from dsd.constraints import _get_implicitly_bound_domain_addresses, _exterior_base_type_of_domain_3p_end, Strand, DomainPool, BasePairType

POOL_1 = DomainPool('POOL_1', 1)
POOL_2 = DomainPool('POOL_2', 2)
POOL_5 = DomainPool('POOL_5', 5)
POOL_13 = DomainPool('POOL_13', 13)

_domain_pools: Dict[int, DomainPool] = {}


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
        constraints._domains_interned = {}

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


if __name__ == '__main__':
    unittest.main()
