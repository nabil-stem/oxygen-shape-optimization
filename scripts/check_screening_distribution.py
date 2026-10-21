"""Exact finite-distribution check of the screened-uniform rejection argument."""

import hashlib
import json
from fractions import Fraction as F
from itertools import product
from pathlib import Path


def main() -> None:
    probabilities = [F(1, 10), F(2, 10), F(3, 10), F(4, 10)]
    perimeters = [1, 3, 2, 4]
    epsilon, cheap, full = F(1, 4), {0, 1, 2}, {0, 1}
    totals = {method: [F(0)] * 4 for method in ("screened_uniform", "perimeter")}
    for pool in product(range(4), repeat=2):
        mass = probabilities[pool[0]] * probabilities[pool[1]]
        eligible = [slot for slot in range(2) if pool[slot] in cheap]
        for method in totals:
            totals[method][pool[0]] += epsilon * mass
            if not eligible:
                selected = [0]
            elif method == "screened_uniform":
                selected = eligible
            else:
                minimum = min(perimeters[pool[i]] for i in eligible)
                selected = [i for i in eligible if perimeters[pool[i]] == minimum]
            for slot in selected:
                totals[method][pool[slot]] += (1 - epsilon) * mass / len(selected)
    a = sum(probabilities[i] for i in cheap)
    b = sum(probabilities[i] for i in full)
    expected_valid = b * (epsilon + (1 - epsilon) * (1 - (1 - a) ** 2) / a)
    output = {
        "model": "Exact four-type, two-draw example; not a PDE experiment",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "formula_full_valid_probability": str(expected_valid),
        "methods": {},
    }
    for method, masses in totals.items():
        assert sum(masses) == 1
        valid = sum(masses[i] for i in full)
        conditional = [masses[i] / valid for i in sorted(full)]
        output["methods"][method] = {
            "full_valid_probability": str(valid),
            "valid_type_probabilities": [str(p) for p in conditional],
        }
    assert expected_valid == sum(totals["screened_uniform"][i] for i in full) == F(39, 100)
    assert [totals["screened_uniform"][i] / expected_valid for i in sorted(full)] == [
        F(1, 3),
        F(2, 3),
    ]
    assert output["methods"]["perimeter"]["valid_type_probabilities"] == ["67/147", "80/147"]
    path = Path(__file__).resolve().parents[1] / "docs/data/screening_distribution_check.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("Exact enumeration matches screened-uniform law; minimum-perimeter changes it.")


if __name__ == "__main__":
    main()
