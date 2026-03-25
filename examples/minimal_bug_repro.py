"""
Minimal reproduction script for the score drift bug.
Uses search_for_sequences directly (which now has score drift detection built in).

Usage:
    python examples/minimal_bug_repro.py
"""

import nuad.constraints as nc
import nuad.search as ns


def create_design() -> nc.Design:
    """12 strands sharing 16 domains (each length 10)."""
    pool = nc.DomainPool("pool10", 10)
    design = nc.Design()

    strand_defs = [
        (['a', 'b', 'c'], 's0'),
        (['b*', 'd', 'e'], 's1'),
        (['c*', 'e*', 'f'], 's2'),
        (['a*', 'd*', 'f*'], 's3'),
        (['g', 'h', 'a'], 's4'),
        (['h*', 'i', 'b'], 's5'),
        (['i*', 'j', 'c'], 's6'),
        (['j*', 'g*', 'd'], 's7'),
        (['k', 'l', 'e', 'g'], 's8'),
        (['l*', 'm', 'f', 'h'], 's9'),
        (['m*', 'n', 'k*', 'i'], 's10'),
        (['n*', 'j', 'l', 'a*'], 's11'),
    ]

    for domain_names, name in strand_defs:
        design.add_strand(domain_names=domain_names, name=name)

    for strand in design.strands:
        for domain in strand.domains:
            if not domain.has_pool():
                domain.pool = pool

    return design


def create_constraints() -> list[nc.Constraint]:
    """
    StrandConstraints with always-positive, float-valued excesses.
    """

    # C1: excess = 0.1 + count_A / len  (always > 0, float-valued)
    def eval_a_frac(seqs: tuple[str, ...], _strand: nc.Strand | None) -> nc.Result:
        seq = seqs[0]
        frac = seq.count('A') / len(seq)
        excess = 0.1 + frac
        return nc.Result(excess=excess, summary=f"A_frac={frac:.3f}", value=frac)

    c1 = nc.StrandConstraint(
        description="A fraction", short_description="A_frac",
        weight=1.0, evaluate=eval_a_frac,
    )

    # C2: excess = 0.1 + count_T / len
    def eval_t_frac(seqs: tuple[str, ...], _strand: nc.Strand | None) -> nc.Result:
        seq = seqs[0]
        frac = seq.count('T') / len(seq)
        excess = 0.1 + frac
        return nc.Result(excess=excess, summary=f"T_frac={frac:.3f}", value=frac)

    c2 = nc.StrandConstraint(
        description="T fraction", short_description="T_frac",
        weight=1.0, evaluate=eval_t_frac,
    )

    # C3: excess = 0.1 + runs / len (adjacent identical bases)
    def eval_runs_frac(seqs: tuple[str, ...], _strand: nc.Strand | None) -> nc.Result:
        seq = seqs[0]
        runs = sum(1 for i in range(len(seq) - 1) if seq[i] == seq[i + 1])
        frac = runs / len(seq)
        excess = 0.1 + frac
        return nc.Result(excess=excess, summary=f"runs_frac={frac:.3f}", value=frac)

    c3 = nc.StrandConstraint(
        description="runs fraction", short_description="RunsFrac",
        weight=1.0, evaluate=eval_runs_frac,
    )

    return [c1, c2, c3]


def main() -> None:
    design = create_design()
    constraints = create_constraints()

    params = ns.SearchParameters(
        constraints=constraints,
        out_directory=None,  # don't write files
        random_seed=42,
        never_increase_score=False,
        scrolling_output=False,
        log_time=False,
        max_iterations=500000,
    )

    ns.search_for_sequences(design, params)


if __name__ == "__main__":
    main()
