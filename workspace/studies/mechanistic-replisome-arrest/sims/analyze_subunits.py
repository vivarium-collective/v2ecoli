"""Study 1, test 3: WHICH replisome subunit is exhausted at the arrest?

The initiation gate (chromosome_replication.py:453) fires only when
  every trimer pool  >= 6 * n_oriC   AND   every monomer pool >= 2 * n_oriC
so the limiting pool is whichever falls below its own requirement.

Reads the mechanistic arm's parquet history for the arrested generation and
reports, per pool, the count against its requirement over time.
"""
import glob
import polars as pl

ARM = "out/mechanistic-replisome-arrest/mechanistic"
TRIMERS = {"CPLX0-2361[c]": "pol III core", "CPLX0-3761[c]": "beta clamp"}
MONOMERS = {"CPLX0-3621[c]": "DnaB hexamer", "EG10239-MONOMER[c]": "DnaG",
            "EG11500-MONOMER[c]": "HolB (delta')", "EG11412-MONOMER[c]": "HolA (delta)"}
ALL = {**TRIMERS, **MONOMERS}


def load(generation, agent=None):
    pat = f"{ARM}/**/history/**/generation={generation}/**/*.pq"
    files = sorted(glob.glob(pat, recursive=True))
    if agent:
        files = [f for f in files if f"agent_id={agent}/" in f]
    if not files:
        return None
    df = pl.read_parquet(files)
    return df.sort("time") if "time" in df.columns else df


for gen in (1, 2):
    df = load(gen)
    if df is None:
        print(f"gen {gen}: no data\n")
        continue
    # agent_id partitions: the runner-driven lineage cell vs its daughters
    agents = sorted(set(p.split("agent_id=")[1].split("/")[0]
                        for p in glob.glob(f"{ARM}/**/history/**/generation={gen}/**/*.pq",
                                           recursive=True)))
    print(f"=== generation {gen}  ({df.height} ticks, agents {agents}) ===")

    oric = df["listeners__replication_data__number_of_oric"]
    print(f"  n_oriC: first={oric[0]}  last={oric[-1]}  min={oric.min()}  max={oric.max()}")

    ids = df["bulk__id"][0].to_list()
    idx = {m: ids.index(m) for m in ALL if m in ids}
    missing = [m for m in ALL if m not in idx]
    if missing:
        print(f"  NOT FOUND in bulk: {missing}")

    print(f"  {'pool':<22}{'need':>8}{'first':>8}{'min':>8}{'last':>8}   {'shortfall at min':>16}")
    rows = []
    for mol, label in ALL.items():
        if mol not in idx:
            continue
        counts = df["bulk__count"].list.get(idx[mol])
        mult = 6 if mol in TRIMERS else 2
        need = oric * mult
        deficit = (counts - need)
        worst = deficit.min()
        rows.append((label, mol, int(counts.min()), int(worst)))
        print(f"  {label:<22}{f'{mult}x oriC':>8}{counts[0]:>8}{counts.min():>8}"
              f"{counts[-1]:>8}   {worst:>16}")
    short = [r for r in rows if r[3] < 0]
    print(f"\n  pools that ever fell BELOW requirement: "
          f"{[f'{r[0]} (deficit {r[3]})' for r in short] or 'NONE'}")
    if short:
        worst = min(short, key=lambda r: r[3])
        print(f"  most limiting: {worst[0]}  ({worst[1]})")
    print()
