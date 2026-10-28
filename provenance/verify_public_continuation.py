"""Verify published continuation metadata without models, corpus access or network."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected):
    require(math.isfinite(actual) and math.isfinite(expected)
            and abs(actual - expected) <= 1e-12, "numeric reduction mismatch")


def mean(values):
    require(bool(values), "empty population")
    return sum(values) / len(values)


def rate(row):
    n = row["valid_targets"]
    require(type(n) is int and n > 0, "invalid target denominator")
    value = row["kl_sum"] / n
    require(math.isfinite(value), "nonfinite KL")
    return value


def component_rates(rows, key="component"):
    grouped = defaultdict(lambda: [0.0, 0])
    for row in rows:
        rate(row)
        grouped[row[key]][0] += row["kl_sum"]
        grouped[row[key]][1] += row["valid_targets"]
    require(bool(grouped), "empty component population")
    return {c: total / n for c, (total, n) in grouped.items()}


def verify_scores(value):
    pairs = value["pairs"]
    require(len(pairs) == 20 and pairs == sorted(pairs)
            and len({tuple(p) for p in pairs}) == 20, "incomplete candidate universe")
    components, damage_components = value["score_components"], value["damage_components"]
    require(not set(components) & set(damage_components), "score/damage overlap")
    matrices = value["score_matrices"]
    methods = {"natural_kl", "student_only", "random_donor", "active_1", "active_01"}
    require(set(matrices) == methods, "missing scoring method")
    require(len(value["raw"]) == 20 and len(value["damage_matrix"]) == 20,
            "missing candidate rows")
    for matrix in matrices.values():
        require(len(matrix) == 20 and all(len(row) == len(components) for row in matrix),
                "incomplete score matrix")
    for i, raw in enumerate(value["raw"]):
        require(raw["identity"]["pair"] == pairs[i], "candidate identity mismatch")
        natural = component_rates(raw["natural_score_rows"])
        damage = component_rates(raw["natural_damage_rows"])
        require(set(natural) == set(components) and set(damage) == set(damage_components),
                "incomplete component coverage")
        require(len(value["damage_matrix"][i]) == len(damage_components), "incomplete damage matrix")
        for j, c in enumerate(damage_components):
            close(value["damage_matrix"][i][j], damage[c])
        probes = defaultdict(lambda: {"student": [], "donor": []})
        for row in raw["conditional_probes"]:
            require(row["component_id"] in components and row["depth"] in pairs[i]
                    and row["source"] in {"student", "donor"}, "unexpected probe membership")
            rate(row)
            probes[row["component_id"], row["depth"]][row["source"]].append(row)
        for j, c in enumerate(components):
            students, randoms, actives = [], [], []
            for depth in pairs[i]:
                group = probes[c, depth]
                require(len(group["student"]) == 1 and bool(group["donor"]), "missing probe source")
                donors = sorted(group["donor"], key=lambda p: p["probe_id"])
                require(len({p["probe_id"] for p in donors}) == len(donors), "duplicate donor")
                chosen = random.Random(17 + depth + int(c[:8], 16)).randrange(len(donors))
                students.append(rate(group["student"][0]))
                randoms.append(rate(donors[chosen]))
                actives.append(max(map(rate, donors)))
            n, s, r, a = natural[c], mean(students), mean(randoms), mean(actives)
            expected = {"natural_kl": n, "student_only": n + .5*s,
                        "random_donor": n + .5*(s+r), "active_1": n + .5*(s+a),
                        "active_01": n + .05*(s+a)}
            for method in methods:
                close(matrices[method][i][j], expected[method])
    damage_means = list(map(mean, value["damage_matrix"]))
    oracle = min(range(20), key=lambda i: damage_means[i])
    for method, matrix in matrices.items():
        chosen = min(range(20), key=lambda i: mean(matrix[i]))
        point = value["point_estimates"][method]
        require(point["selected_pair"] == pairs[chosen]
                and point["sampled_oracle_pair"] == pairs[oracle], "selection mismatch")
        close(point["damage"], damage_means[chosen])
        close(point["sampled_regret"], damage_means[chosen] - damage_means[oracle])


def verify_numeric(root):
    def load(name):
        return json.loads((root / name).read_text())
    pilot, summary = load("recovery/result.json"), load("summary.json")
    require(pilot["status"] == "complete_exploratory", "incomplete pilot")
    require(set(pilot["arms"]) == {"A", "B", "C"}, "missing recovery arm")
    require(not set(pilot["population"]["fit_components"]) &
            set(pilot["population"]["validation_components"]), "fit/validation overlap")
    require(pilot["arms"]["A"]["initialized_digest"] ==
            pilot["arms"]["B"]["initialized_digest"], "A/B initialization mismatch")
    for name, arm in pilot["arms"].items():
        require(arm["updates"] == 16 and arm["targets"] == 131072, "incomplete fitting")
        require([o["step"] for o in arm["observations"]] == [0, 1, 4, 8, 16], "missing observation")
        weights = pilot["run_spec"]["resolved_config"]["config"]["arms"][name]["mixture"]
        for row in arm["fit_metrics"]:
            require(sum(row["source_counts"].values()) == 8192, "incomplete update")
            terms = []
            for source, weight in weights.items():
                n = row["source_counts"][source]
                require(n > 0, "missing fitting source")
                terms.append(weight * row["source_kl_sums"][source] / n)
            close(row["weighted_objective"], sum(terms))
        for observation in arm["observations"]:
            for population in ("fit", "validation"):
                metric = observation[population]
                for row in metric["rows"]:
                    rate(row)
                n = sum(r["valid_targets"] for r in metric["rows"])
                require(n == metric["valid_targets"], "target total mismatch")
                close(metric["kl_per_target"], sum(r["kl_sum"] for r in metric["rows"]) / n)
        close(summary["arms"][name]["initial_kl"], arm["observations"][0]["validation"]["kl_per_target"])
        close(summary["arms"][name]["final_kl"], arm["observations"][-1]["validation"]["kl_per_target"])
    reference, labels = None, set()
    for result in (load("frontier/result.json"), load("legacy-frontier-retry1/result.json")):
        require(result["status"] == "complete_exploratory", "incomplete native comparison")
        for row in result["rows"]:
            require(row["label"] not in labels, "duplicate artifact")
            labels.add(row["label"])
            for item in row["rows"]:
                rate(item)
            n = sum(r["valid_targets"] for r in row["rows"])
            require(n == 4088, "incomplete native population")
            coverage = sorted((r["component_id"], r["window_sha256"], r["valid_targets"]) for r in row["rows"])
            if reference is None:
                reference = coverage
            require(coverage == reference, "native populations differ")
            close(row["kl_per_target"], sum(r["kl_sum"] for r in row["rows"]) / n)
            published = next(p for p in summary["native"] if p["label"] == row["label"])
            close(published["kl_per_target"], row["kl_per_target"])
            require(published["payload_bytes"] == row["payload_bytes"], "payload mismatch")
    require(len(labels) == len(summary["native"]) == 8, "missing native artifact")
    scores = load("recovered-scores.json")
    for phase in ("original", "fresh_exploratory"):
        verify_scores(scores[phase])


def verify_package(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "SHA256.json").read_text())["files"]
    actual = {str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and p != root / "SHA256.json"}
    require(set(manifest) == actual, "manifest coverage mismatch")
    for name, digest in manifest.items():
        path = root / name
        require(path.resolve().is_relative_to(root) and not path.is_symlink(), "unsafe manifest path")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, "file hash mismatch: " + name)
    verify_numeric(root)
    return {"hashed_files": len(manifest), "recovery_arms": 3, "native_artifacts": 8,
            "score_candidates_per_phase": 20,
            "scope": "public metadata integrity and arithmetic; no scientific experiment rerun"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "continuation-v2")
    args = parser.parse_args()
    try:
        print(json.dumps(verify_package(args.root), sort_keys=True))
    except (ValueError, KeyError, TypeError, OSError, StopIteration, ZeroDivisionError) as exc:
        parser.exit(1, f"continuation verification failed: {exc}\n")
