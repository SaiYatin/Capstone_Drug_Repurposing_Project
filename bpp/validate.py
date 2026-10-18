"""
External and internal validation of BioPathNet++ drug candidates for the six rare autoimmune targets.

  python -m bpp.score ... (full scores, see bpp/score.py)   ->  outputs/bpp/validation/scores/*.npz
  python -m bpp.validate                                     ->  outputs/bpp/validation/*.csv, validation_summary.json,
                                                                 expert_review_sheet.xlsx

Layers (each also run on two CONTROL drug sets so the numbers have a reference point:
20 random unlabelled drugs and the popularity baseline's top-20 unlabelled drugs per target)
  V1  known-drug recovery          rank of every PrimeKG-labelled drug, per seed
  V2  clinical-trial evidence      ClinicalTrials.gov API v2; grade A (phase>=3) B (phase 2) C (phase 1 /
                                   observational / NA) D (none); only trials that NAME the drug as an
                                   intervention count; arm role (experimental / comparator / other) recorded
  V3  literature evidence          PubMed E-utilities "<drug>"[tiab] AND "<disease>"[tiab]; hit count + top-3
                                   PMIDs/titles; titles are classified separately (literature_labels.csv)
  V4  temporal validation          drugs approved for a target AFTER the PrimeKG snapshot: rank in every seed
  V5  mechanism support            drug-target / disease-gene overlap and immune-pathway enrichment
                                   (hypergeometric, Benjamini-Hochberg) from PrimeKG edges
  V7  drug classes                 ATC level-2 classes (DrugBank->ATC from DRKG), popularity-adjusted list
  V8  expert review sheet          top-10 per target with all evidence, empty rating column

Every API response is cached under outputs/bpp/validation/cache/ (the raw evidence, re-usable offline).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import hypergeom

from bpp.kg import CompactKG
from bpp.task import RepurposingTask

TARGET_TERMS = {   # ClinicalTrials.gov condition query / PubMed [tiab] synonyms
    "Primary Biliary Cholangitis": ["primary biliary cholangitis", "primary biliary cirrhosis"],
    "Systemic Sclerosis": ["systemic sclerosis", "scleroderma"],
    "Sjogren Syndrome": ["Sjogren", "Sjögren"],
    "Autoimmune Hepatitis": ["autoimmune hepatitis"],
    "Myasthenia Gravis": ["myasthenia gravis"],
    "Crohn Disease": ["Crohn"],
}
# Approved for the target after the PrimeKG snapshot (2021-22); dates are verified separately and
# recorded with their source in temporal_sources.json. Drugs not found in PrimeKG are reported as such.
TEMPORAL = {
    "Primary Biliary Cholangitis": ["elafibranor", "seladelpar", "obeticholic acid"],
    "Myasthenia Gravis": ["efgartigimod", "ravulizumab", "rozanolixizumab", "zilucoplan", "nipocalimab"],
    "Crohn Disease": ["risankizumab", "upadacitinib", "mirikizumab", "guselkumab"],
    "Systemic Sclerosis": ["tocilizumab", "nintedanib"],
}
IMMUNE_RE = re.compile(r"cytokine|interleukin|\btnf|\bjak\b|\bstat\d?\b|nf-?kb|t cell receptor|b cell receptor|"
                       r"complement|interferon|toll like|immun", re.I)
TOP_K = 20
UA = {"User-Agent": "BioPathNetPP-validation/1.0 (academic research)"}


def clean(name: str) -> str:
    return re.sub(r"\s*\[.*\]$", "", name).strip()


def drugbank_id(name: str):
    m = re.search(r"DrugBank:(DB\d+)", name)
    return m.group(1) if m else None


# ----------------------------------------------------------------------------- cached web queries
class Cache:
    def __init__(self, root: Path):
        self.root = root

    def get(self, kind, key, fetch):
        p = self.root / kind / (re.sub(r"[^A-Za-z0-9]+", "_", key)[:150] + ".json")
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        val = fetch()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(val, indent=1), encoding="utf-8")
        return val


def _get(url, params):
    """GET with retries on rate limits / server errors; None if the API rejects the query (HTTP 400)."""
    for attempt in range(5):
        r = requests.get(url, params=params, headers=UA, timeout=60)
        if r.status_code == 400:
            return None
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 * (attempt + 1)); continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def ctgov(cache: Cache, drug: str, cond_terms: list) -> list:
    cond = " OR ".join(f'"{c}"' for c in cond_terms)

    def fetch():
        studies, token = [], None
        for _ in range(5):
            params = {"query.cond": cond, "query.intr": f'"{drug}"', "pageSize": 100,
                      "fields": "NCTId,BriefTitle,Phase,OverallStatus,StartDate,StudyType,InterventionName,"
                                "ArmGroupLabel,ArmGroupType,ArmGroupInterventionName"}
            if token:
                params["pageToken"] = token
            js = _get("https://clinicaltrials.gov/api/v2/studies", params)
            if js is None:                      # e.g. long IUPAC names the API cannot parse: no evidence
                return [{"query_rejected": True}]
            studies += js.get("studies", [])
            token = js.get("nextPageToken")
            time.sleep(0.2)
            if not token:
                break
        return studies
    return cache.get("ctgov", f"{drug}__{cond}", fetch)


def parse_trials(studies: list, drug: str) -> list:
    """Keep trials that name the drug as an intervention; record phase and the drug's arm role."""
    out, d = [], drug.lower()
    for s in studies:
        if s.get("query_rejected"):
            continue
        ps = s.get("protocolSection", {})
        ai = ps.get("armsInterventionsModule", {})
        names = [i.get("name", "") for i in ai.get("interventions", [])]
        if not any(d in n.lower() for n in names):
            continue
        role = "other"
        for arm in ai.get("armGroups", []):
            if any(d in n.lower() for n in arm.get("interventionNames", [])):
                t = arm.get("type", "")
                if t == "EXPERIMENTAL":
                    role = "experimental"; break
                if t == "ACTIVE_COMPARATOR":
                    role = "comparator"
        phases = ps.get("designModule", {}).get("phases", []) or ["NA"]
        out.append({"nct": ps.get("identificationModule", {}).get("nctId"),
                    "title": ps.get("identificationModule", {}).get("briefTitle"),
                    "phase": "/".join(phases), "status": ps.get("statusModule", {}).get("overallStatus"),
                    "start": (ps.get("statusModule", {}).get("startDateStruct", {}) or {}).get("date", ""),
                    "type": ps.get("designModule", {}).get("studyType"), "role": role})
    return out


def grade(trials: list) -> str:
    if not trials:
        return "D"
    ph = " ".join(t["phase"] for t in trials)
    if "PHASE3" in ph or "PHASE4" in ph:
        return "A"
    if "PHASE2" in ph:
        return "B"
    return "C"


def pubmed(cache: Cache, drug: str, terms: list) -> dict:
    term = f'"{drug}"[tiab] AND (' + " OR ".join(f'"{t}"[tiab]' for t in terms) + ")"

    def fetch():
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        js = _get(base + "esearch.fcgi", {"db": "pubmed", "term": term, "retmode": "json", "retmax": 3, "sort": "relevance"})
        if js is None:
            return {"term": term, "count": 0, "top": [], "query_rejected": True}
        es = js["esearchresult"]
        time.sleep(0.4)
        hits = []
        if es["idlist"]:
            res = _get(base + "esummary.fcgi", {"db": "pubmed", "id": ",".join(es["idlist"]), "retmode": "json"})["result"]
            hits = [{"pmid": p, "title": res[p].get("title", ""), "year": res[p].get("pubdate", "")[:4]}
                    for p in es["idlist"] if p in res]
            time.sleep(0.4)
        return {"term": term, "count": int(es["count"]), "top": hits}
    return cache.get("pubmed", term, fetch)


# ----------------------------------------------------------------------------- helpers
def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def ranks(scores: np.ndarray) -> np.ndarray:
    """1 = best; ties broken by order (as in a sorted list)."""
    r = np.empty_like(scores, dtype=int)
    r[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
    return r


class Ctx:
    def __init__(self, data: Path, score_dir: Path):
        self.kg = CompactKG.load(str(data))
        self.task = RepurposingTask(self.kg, json.loads((data / "splits.json").read_text()))
        self.names = self.kg.node_names
        self.drugs = self.task.drugs
        self.seeds = []
        for f in sorted(score_dir.glob("main_s*.npz")):
            z = np.load(f)
            assert np.array_equal(z["drugs"], self.drugs)
            self.seeds.append({str(n): s for n, s in zip(z["names"], z["scores"])})
        self.targets = {n: int(p) for n, p in self.task.split["target_primary"].items()}
        self.labels = {t: set(self.task.eval_labels(d, True).tolist()) for t, d in self.targets.items()}
        self.rank = {t: np.stack([ranks(s[t]) for s in self.seeds]) for t in self.targets}   # (seeds, C)
        self.mean_rank = {t: r.mean(0) for t, r in self.rank.items()}

    def dname(self, col):
        return clean(self.names[self.drugs[col]])


# ----------------------------------------------------------------------------- candidate sets
def candidate_sets(ctx: Ctx, rng_seed=0) -> dict:
    """target -> {"model": [cols], "random": [cols], "popularity": [cols]} (model: top-20 by mean rank)."""
    out = {}
    for t, d in ctx.targets.items():
        lab = ctx.labels[t]
        model = list(np.argsort(ctx.mean_rank[t], kind="stable")[:TOP_K])
        pop = ctx.task.baselines(d)["popularity"]
        pop_top = [c for c in np.argsort(-pop, kind="stable") if c not in lab][:TOP_K]
        rng = np.random.default_rng(rng_seed + len(out))
        rand = list(rng.choice([c for c in range(ctx.task.C) if c not in lab], TOP_K, replace=False))
        union = set()
        for r in ctx.rank[t]:
            union |= set(np.flatnonzero(r <= TOP_K))
        out[t] = {"model": [int(c) for c in model], "random": [int(c) for c in rand],
                  "popularity": [int(c) for c in pop_top], "union_size": len(union)}
    return out


# ----------------------------------------------------------------------------- V1
def v1(ctx: Ctx) -> pd.DataFrame:
    rows = []
    for t in ctx.targets:
        for c in sorted(ctx.labels[t]):
            rows.append({"target": t, "drug": ctx.dname(c), **{f"rank_s{i}": int(r[c]) for i, r in enumerate(ctx.rank[t])},
                         "mean_rank": float(ctx.mean_rank[t][c])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- V2 + V3
def evidence(ctx: Ctx, cands: dict, cache: Cache):
    trial_rows, lit_rows = [], []
    for t, sets in cands.items():
        for group, cols in (("model", sets["model"]), ("random", sets["random"]), ("popularity", sets["popularity"])):
            for c in cols:
                drug = ctx.dname(c)
                labelled = c in ctx.labels[t]
                if group != "model" and labelled:
                    continue
                trials = parse_trials(ctgov(cache, drug, TARGET_TERMS[t]), drug)
                g = grade(trials)
                best = sorted(trials, key=lambda x: ("PHASE4" in x["phase"], "PHASE3" in x["phase"],
                                                     "PHASE2" in x["phase"]), reverse=True)[:3]
                trial_rows.append({"target": t, "group": group, "drug": drug, "labelled": labelled,
                                   "mean_rank": float(ctx.mean_rank[t][c]), "grade": g, "n_trials": len(trials),
                                   "roles": ",".join(sorted({x["role"] for x in trials})),
                                   "top_trials": "; ".join(f"{x['nct']} {x['phase']} {x['status']} {x['start'][:4]} {x['role']}"
                                                           for x in best)})
                pm = pubmed(cache, drug, TARGET_TERMS[t])
                lit_rows.append({"target": t, "group": group, "drug": drug, "labelled": labelled,
                                 "pubmed_hits": pm["count"],
                                 **{f"pmid{i + 1}": h["pmid"] for i, h in enumerate(pm["top"])},
                                 **{f"title{i + 1}": h["title"] for i, h in enumerate(pm["top"])}})
    return pd.DataFrame(trial_rows), pd.DataFrame(lit_rows)


# ----------------------------------------------------------------------------- V4
def v4(ctx: Ctx) -> pd.DataFrame:
    rows = []
    lower = {i: clean(ctx.names[d]).lower() for i, d in enumerate(ctx.drugs)}
    for t, drugs in TEMPORAL.items():
        d = ctx.targets[t]
        bl = ctx.task.baselines(d)
        knn_r, pop_r = ranks(bl["disease_knn"]), ranks(bl["popularity"])
        for name in drugs:
            cols = [c for c, n in lower.items() if n.startswith(name)]
            if not cols:
                rows.append({"target": t, "drug": name, "in_primekg": False}); continue
            c = cols[0]
            r = ctx.rank[t][:, c]
            rows.append({"target": t, "drug": name, "in_primekg": True, "primekg_name": ctx.dname(c),
                         "already_labelled": c in ctx.labels[t],
                         **{f"rank_s{i}": int(x) for i, x in enumerate(r)}, "mean_rank": float(r.mean()),
                         "percentile_top": round(100 * r.mean() / ctx.task.C, 2),
                         "knn_rank": int(knn_r[c]), "popularity_rank": int(pop_r[c])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- V5
def v5(ctx: Ctx, cands: dict) -> pd.DataFrame:
    kg, rel = ctx.kg, ctx.kg.relation_names

    def edges(name):
        m = kg.rel == rel.index(name)
        return kg.head[m], kg.tail[m]
    prot_t = kg.type_id("gene/protein")
    tp = np.asarray(kg.node_type)

    def pairs(name, other_type):
        h, t = edges(name)
        a_is_other = tp[h] == other_type
        return np.where(a_is_other, h, t), np.where(a_is_other, t, h)     # (other node, protein)
    drug_t, dis_t, pw_t = kg.type_id(kg.meta["drug_type"]), kg.type_id(kg.meta["disease_type"]), kg.type_id("pathway")
    dtargets, dgenes, pw_genes = defaultdict(set), defaultdict(set), defaultdict(set)
    for a, p in zip(*pairs("drug_protein", drug_t)):
        dtargets[int(a)].add(int(p))
    for a, p in zip(*pairs("disease_protein", dis_t)):
        dgenes[int(a)].add(int(p))
    for a, p in zip(*pairs("pathway_protein", pw_t)):
        pw_genes[int(a)].add(int(p))
    immune = set().union(*(g for pw, g in pw_genes.items() if IMMUNE_RE.search(ctx.names[pw])))
    pw_universe = set().union(*pw_genes.values())
    n_prot = int((tp == prot_t).sum())
    rows = []
    for t, sets in cands.items():
        dis_genes = dgenes[ctx.targets[t]]
        for group in ("model", "random", "popularity"):
            for c in sets[group]:
                tg = dtargets[int(ctx.drugs[c])]
                k_dis = len(tg & dis_genes)
                p_dis = hypergeom.sf(k_dis - 1, n_prot, len(dis_genes), len(tg)) if tg and k_dis else 1.0
                tg_pw = tg & pw_universe
                k_imm = len(tg_pw & immune)
                p_imm = hypergeom.sf(k_imm - 1, len(pw_universe), len(immune), len(tg_pw)) if tg_pw and k_imm else 1.0
                rows.append({"target": t, "group": group, "drug": ctx.dname(c), "labelled": c in ctx.labels[t],
                             "n_targets": len(tg), "targets_in_disease_genes": k_dis, "p_disease": p_dis,
                             "targets_in_immune_pathways": k_imm, "p_immune": p_imm})
    df = pd.DataFrame(rows)
    df["q_disease"] = bh(df["p_disease"].to_numpy())
    df["q_immune"] = bh(df["p_immune"].to_numpy())
    return df


# ----------------------------------------------------------------------------- V7
def atc_map(drkg_tsv: Path) -> dict:
    out = defaultdict(set)
    if not drkg_tsv.exists():
        return out
    with open(drkg_tsv, encoding="utf-8") as f:
        for line in f:
            if "Atc::" in line and "x-atc" in line:
                h, _, t = line.rstrip("\n").split("\t")
                out[h.split("::")[1]].add(t.split("::")[1])
    return out


ATC2 = {"H02": "corticosteroids (systemic)", "L04": "immunosuppressants", "L01": "antineoplastics",
        "J01": "antibacterials (systemic)", "M01": "anti-inflammatory/antirheumatic", "A07": "intestinal anti-inflammatory/anti-infective",
        "N07": "other nervous system (incl. cholinesterase inhibitors)", "D07": "corticosteroids (dermatological)",
        "S01": "ophthalmologicals", "A05": "bile therapy"}


# drug-class flags from ALL of a drug's ATC codes (DRKG lists e.g. prednisolone only under D07/S01/R01, not H02)
CLASS_PREFIXES = {
    "corticosteroid": ("H02", "D07A", "S01BA", "S01CA", "R01AD", "R03BA", "A07EA", "C05AA", "D10AA", "S02B", "S03B"),
    "antibacterial": ("J01", "J04", "A07AA", "S01AA", "S02AA", "S03AA", "D06A", "D10AF", "R02AB", "G01A"),
    "immunosuppressant": ("L04",),
    "antineoplastic": ("L01",),
}


NAME_CLASS = {  # secondary, transparent name heuristic covering drugs without ATC codes in DRKG
    "corticosteroid": re.compile(r"(cortisone|olone|methasone|^prednisone|cortisol|budesonide|ciclesonide)(?![a-z])", re.I),
    "antibacterial": re.compile(r"(cillin|cycline|floxacin|mycin|micin|^cef|penem|bactam|chloramphenicol|"
                                r"^sulfa(cetamide|methoxazole|diazine|salazine)|fusidic)", re.I),
}


def drug_classes(codes) -> list:
    return [c for c, pre in CLASS_PREFIXES.items() if any(x.startswith(pre) for x in codes)]


def v7(ctx: Ctx, cands: dict, atc: dict) -> tuple:
    rows, adj_rows = [], []
    for t, sets in cands.items():
        for c in sets["model"]:
            db = drugbank_id(ctx.names[ctx.drugs[c]])
            codes = sorted(atc.get(db, set()))
            lvl2 = sorted({x[:3] for x in codes if len(x) >= 3})      # DRKG also stores 1-letter ATC groups
            rows.append({"target": t, "drug": ctx.dname(c), "drugbank": db, "labelled": c in ctx.labels[t],
                         "atc": ",".join(codes), "atc2": ",".join(lvl2), "classes": ",".join(drug_classes(codes)),
                         "class": "; ".join(ATC2.get(x, x) for x in lvl2)})
        # popularity-adjusted (exploratory): residual of the seed-averaged z-scored score on log1p(popularity)
        z = np.mean([(s[t] - s[t].mean()) / s[t].std() for s in ctx.seeds], axis=0)
        pop = np.log1p(ctx.task.baselines(ctx.targets[t])["popularity"])
        beta = np.polyfit(pop, z, 1)
        resid = z - np.polyval(beta, pop)
        order = [c for c in np.argsort(-resid) if c not in ctx.labels[t]][:10]
        for i, c in enumerate(order):
            adj_rows.append({"target": t, "adj_rank": i + 1, "drug": ctx.dname(c), "model_mean_rank": float(ctx.mean_rank[t][c]),
                             "residual": float(resid[c]), "popularity_slope": float(beta[0])})
    return pd.DataFrame(rows), pd.DataFrame(adj_rows)


# ----------------------------------------------------------------------------- transfer test set E (lighter check)
def e_check(ctx: Ctx, cache: Cache, score_file: Path, k: int = 10) -> pd.DataFrame:
    """C_E model's top-k for the 12 held-out autoimmune-area diseases: labelled or not, and trial grade."""
    if not score_file.exists():
        return pd.DataFrame()
    z = np.load(score_file)
    rows = []
    for name, node, sc in zip(z["names"], z["diseases"], z["scores"]):
        labels = set(ctx.task.eval_labels(int(node), False).tolist())
        cond = [clean(str(name))]
        for rank, c in enumerate(np.argsort(-sc, kind="stable")[:k], 1):
            drug = ctx.dname(int(c))
            labelled = int(c) in labels
            trials = [] if labelled else parse_trials(ctgov(cache, drug, cond), drug)
            rows.append({"disease": clean(str(name)), "rank": rank, "drug": drug, "labelled": labelled,
                         "grade": "labelled" if labelled else grade(trials),
                         "top_trials": "; ".join(f"{x['nct']} {x['phase']} {x['role']}" for x in trials[:2])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/bpp_primekg")
    ap.add_argument("--out", default="outputs/bpp/validation")
    ap.add_argument("--drkg", default="datasets/processed/enriched_drkg.tsv")
    ap.add_argument("--offline", action="store_true", help="use cached API responses only")
    a = ap.parse_args()
    out = Path(a.out)
    ctx = Ctx(Path(a.data), out / "scores")
    cache = Cache(out / "cache")
    cands = candidate_sets(ctx)
    summary = {"seeds": len(ctx.seeds), "top_k": TOP_K,
               "candidate_union_sizes": {t: c["union_size"] for t, c in cands.items()}}

    df1 = v1(ctx); df1.to_csv(out / "v1_known_drug_recovery.csv", index=False)
    summary["v1"] = {t: {"n_labelled": int(len(g)), "best_mean_rank": float(g.mean_rank.min()),
                         "any_hit10_per_seed": [bool((g[f"rank_s{i}"] <= 10).any()) for i in range(len(ctx.seeds))]}
                     for t, g in df1.groupby("target")}

    trials, lit = evidence(ctx, cands, cache)
    trials.to_csv(out / "trial_evidence.csv", index=False)
    lit.to_csv(out / "literature_evidence.csv", index=False)
    labels_file = out / "literature_labels.csv"
    lab = pd.read_csv(labels_file, dtype=str) if labels_file.exists() else pd.DataFrame(columns=["pmid", "drug", "target", "label"])
    lab_map = {(r.pmid, r.drug, r.target): r.label for r in lab.itertuples()}   # a paper can mean different things per pair

    def lit_class(row):
        ls = [lab_map.get((str(int(row[f"pmid{i}"])), row.drug, row.target)) for i in (1, 2, 3)
              if f"pmid{i}" in row and pd.notna(row[f"pmid{i}"])]
        if not ls:
            return "no_hits"
        if any(x is None for x in ls):
            return "unclassified"
        if "supports" in ls and "against" in ls:
            return "mixed"
        for c in ("supports", "against", "mixed"):
            if c in ls:
                return c
        return "unrelated"
    lit["lit_class"] = lit.apply(lit_class, axis=1)
    lit.to_csv(out / "literature_evidence.csv", index=False)

    def share(df, col, ok):
        g = df[(~df.labelled) | (df.group != "model")] if "labelled" in df else df
        return {grp: {"n": int((g.group == grp).sum()), "share": round(float(g[g.group == grp][col].isin(ok).mean()), 3)}
                for grp in ("model", "random", "popularity")}
    summary["v2_trial_grade_A_to_C_among_novel"] = share(trials, "grade", ["A", "B", "C"])
    summary["v2_trial_grade_A_or_B_among_novel"] = share(trials, "grade", ["A", "B"])
    summary["v2_grade_counts_model_novel"] = Counter(trials[(trials.group == "model") & (~trials.labelled)].grade)
    summary["v3_any_pubmed_hit_among_novel"] = {
        grp: round(float((lit[(lit.group == grp) & (~lit.labelled)].pubmed_hits > 0).mean()), 3) for grp in ("model", "random", "popularity")}
    summary["v3_supporting_among_novel"] = share(lit, "lit_class", ["supports"])
    summary["v3_against_among_novel"] = share(lit, "lit_class", ["against"])
    summary["v3_class_counts_model_novel"] = Counter(lit[(lit.group == "model") & (~lit.labelled)].lit_class)

    df4 = v4(ctx)
    src_file = out / "temporal_sources.json"
    src = json.loads(src_file.read_text()) if src_file.exists() else {}
    df4["fda_approval"] = df4.drug.map(lambda d: src.get(d, {}).get("fda_approval"))
    df4["post_snapshot"] = df4.drug.map(lambda d: src.get(d, {}).get("post_snapshot"))
    df4["source"] = df4.drug.map(lambda d: src.get(d, {}).get("source"))
    df4.to_csv(out / "temporal_validation.csv", index=False)
    summary["v4"] = df4.to_dict(orient="records")
    from scipy.stats import binomtest
    for tag, mask in (("post_snapshot", df4.post_snapshot == True), ("label_gap", df4.post_snapshot == False)):
        g = df4[mask & df4.in_primekg.astype(bool)]
        if len(g):
            better = int((g.mean_rank < ctx.task.C / 2).sum())
            summary[f"v4_{tag}"] = {
                "n": int(len(g)), "median_model_rank": float(g.mean_rank.median()),
                "median_percentile": float(g.percentile_top.median()), "n_in_top_10pct": int((g.percentile_top <= 10).sum()),
                "n_better_than_median_rank": better,
                "sign_test_p_one_sided": float(binomtest(better, len(g), 0.5, alternative="greater").pvalue),
                "median_knn_rank": float(g.knn_rank.median()), "median_popularity_rank": float(g.popularity_rank.median()),
                "n_model_better_than_knn": int((g.mean_rank < g.knn_rank).sum())}

    df5 = v5(ctx, cands); df5.to_csv(out / "pathway_support.csv", index=False)
    summary["v5_immune_enriched_q05"] = {grp: round(float((df5[df5.group == grp].q_immune < 0.05).mean()), 3)
                                         for grp in ("model", "random", "popularity")}
    summary["v5_mean_immune_pathway_targets"] = {grp: round(float(df5[df5.group == grp].targets_in_immune_pathways.mean()), 2)
                                                 for grp in ("model", "random", "popularity")}
    summary["v5_share_any_disease_gene_target"] = {grp: round(float((df5[df5.group == grp].targets_in_disease_genes > 0).mean()), 3)
                                                   for grp in ("model", "random", "popularity")}
    summary["v5_disease_gene_overlap_q05"] = {grp: round(float((df5[df5.group == grp].q_disease < 0.05).mean()), 3)
                                              for grp in ("model", "random", "popularity")}

    atc = atc_map(Path(a.drkg))
    df7, adj = v7(ctx, cands, atc)
    for c, rx in NAME_CLASS.items():
        df7[f"name_{c}"] = df7.drug.apply(lambda n, rx=rx: bool(rx.search(n)))
    df7.to_csv(out / "drug_classes.csv", index=False)
    adj.to_csv(out / "popularity_adjusted_candidates.csv", index=False)
    have = df7[df7.atc2 != ""]
    summary["v7"] = {"atc_coverage": round(float((df7.atc2 != "").mean()), 3),
                     **{f"share_{c}_of_atc_mapped": round(float(have.classes.str.contains(c).mean()), 3) if len(have) else None
                        for c in CLASS_PREFIXES},
                     **{f"share_{c}_by_name_all_candidates": round(float(df7[f"name_{c}"].mean()), 3) for c in NAME_CLASS},
                     "per_target_class_counts": {t: Counter(x for s in g.classes for x in s.split(",") if x)
                                                 for t, g in df7.groupby("target")},
                     "per_target_top_classes": {t: Counter(x for s in g.atc2 for x in s.split(",") if x).most_common(4)
                                                for t, g in df7.groupby("target")}}

    dfe = e_check(ctx, cache, out / "scores" / "tr_CE_s0_E.npz")
    if len(dfe):
        dfe.to_csv(out / "transfer_E_top10.csv", index=False)
        nov = dfe[~dfe.labelled]
        summary["e_check"] = {"n_slots": int(len(dfe)), "labelled_in_top10": int(dfe.labelled.sum()),
                              "unlabelled": int(len(nov)), "unlabelled_with_trial_A_to_C": int(nov.grade.isin(["A", "B", "C"]).sum()),
                              "diseases_with_labelled_in_top10": int(dfe[dfe.labelled].disease.nunique())}

    # V8 expert sheet: top-10 per target
    rows = []
    for t, sets in cands.items():
        for i, c in enumerate(sets["model"][:10]):
            drug = ctx.dname(c)
            tr = trials[(trials.target == t) & (trials.drug == drug) & (trials.group == "model")].iloc[0]
            li = lit[(lit.target == t) & (lit.drug == drug) & (lit.group == "model")].iloc[0]
            pw = df5[(df5.target == t) & (df5.drug == drug) & (df5.group == "model")].iloc[0]
            cl = df7[(df7.target == t) & (df7.drug == drug)].iloc[0]
            rows.append({"Target": t, "Rank (mean of 3 seeds)": round(float(ctx.mean_rank[t][c]), 1), "Drug": drug,
                         "Already a known drug (PrimeKG)": "yes" if c in ctx.labels[t] else "no",
                         "Trial evidence grade (A-D)": tr.grade, "Example trials": tr.top_trials,
                         "PubMed hits": int(li.pubmed_hits), "Literature (AI-assisted title reading)": li.lit_class,
                         "ATC class": cl["class"], "Targets in immune pathways": int(pw.targets_in_immune_pathways),
                         "Plausibility 1-5 (reviewer)": "", "Reviewer comment": ""})
    pd.DataFrame(rows).to_excel(out / "expert_review_sheet.xlsx", index=False)

    (out / "validation_summary.json").write_text(json.dumps(summary, indent=1, default=lambda o: dict(o) if isinstance(o, Counter) else str(o)))
    print(json.dumps({k: v for k, v in summary.items() if k != "v4"}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
