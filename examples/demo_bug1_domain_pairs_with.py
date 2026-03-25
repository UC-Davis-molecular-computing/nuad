"""
Demonstrates Bug 1: duplicate self-pairs in create_domain_pairs_with_dict.

Creates 3 domains (d, a, b) and domain pairs including the self-pair (d,d).
Shows how the original function produces duplicates in domain_pairs_with[d],
while the fixed version does not.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import nuad.constraints as nc


def create_domain_pairs_with_dict_ORIGINAL(domain_pairs):
    """Original version: self-pairs get added twice."""
    domain_pairs_with_list: dict[nc.Domain, list[nc.DomainPair]] = {}
    for pair in domain_pairs:
        for domain_in_pair in pair:
            domains_in_tree = domain_in_pair.all_domains_in_tree()
            for domain_in_tree in domains_in_tree:
                if domain_in_tree not in domain_pairs_with_list:
                    domain_pairs_with_list[domain_in_tree] = []
                domain_pairs_with_list[domain_in_tree].append(pair)
    return {domain: tuple(pairs) for domain, pairs in domain_pairs_with_list.items()}


def create_domain_pairs_with_dict_FIXED(domain_pairs):
    """Fixed version: tracks which domains already got each pair."""
    domain_pairs_with_list: dict[nc.Domain, list[nc.DomainPair]] = {}
    for pair in domain_pairs:
        added_for: set[nc.Domain] = set()
        for domain_in_pair in pair:
            domains_in_tree = domain_in_pair.all_domains_in_tree()
            for domain_in_tree in domains_in_tree:
                if domain_in_tree in added_for:
                    continue
                added_for.add(domain_in_tree)
                if domain_in_tree not in domain_pairs_with_list:
                    domain_pairs_with_list[domain_in_tree] = []
                domain_pairs_with_list[domain_in_tree].append(pair)
    return {domain: tuple(pairs) for domain, pairs in domain_pairs_with_list.items()}


def main():
    pool = nc.DomainPool(name="p", length=7)
    d = nc.Domain(name="d", pool=pool)
    a = nc.Domain(name="a", pool=pool)
    b = nc.Domain(name="b", pool=pool)

    # These are the pairs that combinations_with_replacement([d, a, b], 2) would produce
    # that involve domain d:
    pair_dd = nc.DomainPair(d, d)  # self-pair
    pair_da = nc.DomainPair(d, a)
    pair_db = nc.DomainPair(d, b)

    all_pairs = [pair_dd, pair_da, pair_db]

    print("Input domain pairs:")
    for p in all_pairs:
        print(f"  {p.name}")

    print("\n--- ORIGINAL (buggy) ---")
    original = create_domain_pairs_with_dict_ORIGINAL(all_pairs)
    pairs_for_d = original[d]
    print(f"domain_pairs_with[d] has {len(pairs_for_d)} entries:")
    for p in pairs_for_d:
        print(f"  {p.name}")

    print("\n--- FIXED ---")
    fixed = create_domain_pairs_with_dict_FIXED(all_pairs)
    pairs_for_d = fixed[d]
    print(f"domain_pairs_with[d] has {len(pairs_for_d)} entries:")
    for p in pairs_for_d:
        print(f"  {p.name}")

    print("\n--- Why this matters ---")
    print("In evaluate_new, the 'parts' list comes from domain_pairs_with[d].")
    print("These parts go to evaluate_bulk, which groups by name and returns")
    print("one Result per unique name. Then call_evaluate_bulk does zip(results, parts).")
    print()

    original_parts = original[d]
    unique_names = list(dict.fromkeys(p.name for p in original_parts))  # preserves order, deduplicates
    print(f"ORIGINAL: parts has {len(original_parts)} entries, but evaluate_bulk produces {len(unique_names)} results.")
    print(f"  parts:   {[p.name for p in original_parts]}")
    print(f"  results: {unique_names}")
    print(f"  zip pairs them up as:")
    for result_name, part in zip(unique_names, original_parts):
        match = "OK" if result_name == part.name else "WRONG"
        print(f"    result for '{result_name}' -> assigned to part '{part.name}'  {match}")
    remaining = list(original_parts)[len(unique_names):]
    for part in remaining:
        print(f"    (no result)              -> part '{part.name}' DROPPED")

    print()
    fixed_parts = fixed[d]
    unique_names_fixed = list(dict.fromkeys(p.name for p in fixed_parts))
    print(f"FIXED: parts has {len(fixed_parts)} entries, evaluate_bulk produces {len(unique_names_fixed)} results.")
    print(f"  parts:   {[p.name for p in fixed_parts]}")
    print(f"  results: {unique_names_fixed}")
    print(f"  zip pairs them up as:")
    for result_name, part in zip(unique_names_fixed, fixed_parts):
        match = "OK" if result_name == part.name else "WRONG"
        print(f"    result for '{result_name}' -> assigned to part '{part.name}'  {match}")


if __name__ == "__main__":
    main()
