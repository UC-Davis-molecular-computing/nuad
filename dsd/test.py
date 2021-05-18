import unittest
from dsd import constraints
from dsd.constraints import Domain, _exterior_base_type_of_domain_3p_end, Strand, DomainPool, BasePairType, _domains_interned

POOL_1 = DomainPool('POOL_1', 1)
POOL_2 = DomainPool('POOL_2', 2)
POOL_5 = DomainPool('POOL_5', 5)
POOL_13 = DomainPool('POOL_13', 13)


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
        top_strand = Strand(['a', 'b'])
        bot_strand = Strand(['b*', 'a*'])
        top_strand.domains[0].pool = POOL_2
        top_strand.domains[1].pool = POOL_13
        bot_strand.domains[0].pool = POOL_13
        bot_strand.domains[1].pool = POOL_2

        top_a = top_strand.address_of_domain(0)
        top_b = top_strand.address_of_domain(1)
        bot_a = bot_strand.address_of_domain(1)
        bot_b = bot_strand.address_of_domain(0)

        all_bound_domain_addresses = {
            top_a: bot_a,
            bot_a: top_a,
            top_b: bot_b,
            bot_b: top_b,
        }

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
        top_strand = Strand(['a', 'b', 'c'])
        top_strand.domains[0].pool = POOL_5
        top_strand.domains[1].pool = POOL_1
        top_strand.domains[2].pool = POOL_5

        bot_strand = Strand(['c*', 'd', 'a*'])
        bot_strand.domains[0].pool = POOL_5
        bot_strand.domains[1].pool = POOL_1
        bot_strand.domains[2].pool = POOL_5

        top_a = top_strand.address_of_domain(0)
        top_c = top_strand.address_of_domain(2)

        bot_c = bot_strand.address_of_domain(0)
        bot_a = bot_strand.address_of_domain(2)

        all_bound_domain_addresses = {
            top_a: bot_a,
            bot_a: top_a,
            top_c: bot_c,
            bot_c: top_c,
        }

        self.assertEqual(
            _exterior_base_type_of_domain_3p_end(
                top_a, all_bound_domain_addresses),
            BasePairType.MISMATCH)


if __name__ == '__main__':
    unittest.main()
