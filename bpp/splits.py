"""
Zero-shot split construction shared by the PrimeKG and DRKG builders.

A split is a dict saved as splits.json:
  protocol            name of the protocol
  targets             {target_name: [family node ids]}   (labels are evaluated on the family)
  target_primary      {target_name: primary node id}
  valid_diseases      [node ids]  zero-shot validation diseases (random, seeded)
  held_out_diseases   [node ids]  every disease whose therapeutic edges must NEVER enter the
                                  message graph (all target families + valid diseases + extra)
  train_diseases      [node ids]  query diseases used for training (>=1 indication, not held out)
  area_diseases       [node ids]  autoimmune area (TxGNN DOID:417 definition) when available
"""
from __future__ import annotations

import numpy as np

# The six rare/sparse autoimmune targets and the three data-rich ones (MONDO ids, no padding)
TARGETS_MONDO = {
    "Primary Biliary Cholangitis": "5388",
    "Systemic Sclerosis": "5100",
    "Sjogren Syndrome": "10030",
    "Autoimmune Hepatitis": "16264",
    "Myasthenia Gravis": "9688",
    "Crohn Disease": "5011",
}
RICH_MONDO = {
    "Rheumatoid Arthritis": "8383",
    "Systemic Lupus Erythematosus": "7915",
    "Multiple Sclerosis": "5301",
}
TARGETS_MESH = {
    "Primary Biliary Cholangitis": "Disease::MESH:D008105",
    "Systemic Sclerosis": "Disease::MESH:D012595",
    "Sjogren Syndrome": "Disease::MESH:D012859",
    "Autoimmune Hepatitis": "Disease::MESH:D019693",
    "Myasthenia Gravis": "Disease::MESH:D009157",
    "Crohn Disease": "Disease::MESH:D003424",
}


def make_split(protocol: str, families: dict, primary: dict, disease_ids, indication_pos: dict,
               valid_frac: float = 0.05, seed: int = 0, extra_held_out=(), area=(),
               exclude_from_train=(), neighbours: dict | None = None):
    """families: {name: set(node ids)}; indication_pos: {disease: set(drugs)}."""
    rng = np.random.default_rng(seed)
    fam_nodes = set().union(*families.values()) if families else set()
    extra = set(int(x) for x in extra_held_out)
    excl = set(int(x) for x in exclude_from_train)
    labelled = sorted(d for d in disease_ids if d in indication_pos and d not in fam_nodes
                      and d not in extra and d not in excl)
    n_valid = max(1, int(round(valid_frac * len(labelled))))
    valid = sorted(rng.choice(labelled, n_valid, replace=False).tolist())
    vset = set(valid)
    # validation diseases are held out with their disease-disease neighbours (strict, same
    # rule as the targets) so that model selection cannot reward copying from a sibling
    vfam = set(vset)
    for v in vset:
        vfam |= set((neighbours or {}).get(v, ()))
    train = [d for d in labelled if d not in vfam]
    held = sorted(fam_nodes | vfam | extra | excl)
    return {
        "protocol": protocol,
        "seed": seed,
        "targets": {k: sorted(int(x) for x in v) for k, v in families.items()},
        "target_primary": {k: int(v) for k, v in primary.items()},
        "valid_diseases": [int(x) for x in valid],
        "train_diseases": [int(x) for x in train],
        "held_out_diseases": [int(x) for x in held],
        "area_diseases": sorted(int(x) for x in area),
        "excluded_from_train": sorted(excl),
    }


def family_labels(split: dict, indication_pos: dict) -> dict:
    """{target name: set(drugs)} = union of indications over the target's family."""
    out = {}
    for name, fam in split["targets"].items():
        s = set()
        for d in fam:
            s |= indication_pos.get(d, set())
        out[name] = s
    return out
