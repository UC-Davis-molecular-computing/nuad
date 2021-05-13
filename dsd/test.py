import unittest
from dsd.constraints import _exterior_base_type_of_domain_3p_end, Strand, DomainPool, BasePairType

POOL_13 = DomainPool('POOL_13', 13)
POOL_2 = DomainPool('POOL_2', 2)


class TestExteriorBaseTypeOfDomain3PEnd(unittest.TestCase):
    def test_something(self):
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


if __name__ == '__main__':
    unittest.main()
