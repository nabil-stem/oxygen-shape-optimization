"""Curate unchanged new-study files and derive explicitly labeled comparisons."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
root = args.root.resolve()
if json.loads((root / "status.json").read_text())["status"] != "complete":
    raise ValueError("Curate only a completed study")
config = json.loads((root / "identity.json").read_text())
if config["n"] != 100 or config["refinement_factors"] != [2, 4]:
    raise ValueError("This report describes the declared n=100/200/400 protocol")
source_commit = json.loads((root / "provenance.json").read_text())["git_commit"]
invocation = json.loads((root / "invocation.json").read_text())
public = args.output
public.mkdir(exist_ok=False)
manifest = dict(
    numerical_source_commit=source_commit,
    run=root.name,
    source_release="https://github.com/nabil-stem/oxygen-shape-optimization/releases/tag/square-holes-20261022",
    numerical_values_changed=False,
    figure_bytes_changed=False,
    files={},
)


def copy(source, destination):
    target = public / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert hashlib.sha256(target.read_bytes()).hexdigest() == sha
    manifest["files"][destination] = dict(
        release_source=f"experiments/{root.name}/{source.relative_to(root).as_posix()}",
        source_sha256=sha,
        view_sha256=sha,
        transformations=[],
    )


for source in sorted(root.glob("*.csv")):
    copy(source, "tables/" + source.name)
for source in sorted((root / "figures").glob("*")):
    copy(source, "figures/" + source.name)
copy(root / "summary.json", "run_summary.json")
copy(root / "report.md", "raw_report.md")
copy(root / "validation_selection.json", "validation_selection.json")
(public / "public_view_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def read(name):
    return pd.read_csv(root / name, float_precision="round_trip")


v = read("validation.csv")
layouts = read("layout_metrics.csv")
searches = read("search_summary.csv")
rows = []
spread = v[(v["case"] == "layout/spread_6") & (v.Bi == 1)]
for r in searches.itertuples():
    key = f"search/{r.method}_{r.seed}"
    subset = v[(v["case"] == key) & (v.Bi == 1)]
    fv = subset[subset.solver == "FVM"].sort_values("n")
    fe = subset[subset.solver == "FEM"].sort_values("mesh_area", ascending=False)
    lastfv, lastfe = fv.iloc[-1], fe.iloc[-1]
    initialfv = spread[(spread.solver == "FVM") & (spread.n == lastfv.n)].iloc[0]
    initialfe = spread[(spread.solver == "FEM") & (spread.mesh_area == lastfe.mesh_area)].iloc[0]
    metadata = json.loads((root / "cases" / key / "geometry.json").read_text())
    history = json.loads((root / key / "history.json").read_text())
    rejected = json.loads((root / key / "rejections.json").read_text())
    rows.append(
        dict(
            case=key,
            method=r.method,
            seed=r.seed,
            mask_sha256=metadata["mask_sha256"],
            area=r.area,
            robin_perimeter=r.robin_perimeter,
            Q_n100=r.final_Q,
            Q_n400=lastfv.Q_R,
            Q_FEM_fine=lastfe.Q_R,
            Q_FEM_coarse=fe.iloc[0].Q_R,
            FVM_relocation_gain_percent=100 * (lastfv.Q_R / initialfv.Q_R - 1),
            FEM_relocation_gain_percent=100 * (lastfe.Q_R / initialfe.Q_R - 1),
            FEM_gain_vs_unperforated_percent=100 * (lastfe.Q_R / lastfe.Q0 - 1),
            accepted_edits=sum(h["accepted"] for h in history[1:]),
            charged_solves=r.number_of_pde_solves,
            raw_draws=r.raw_draws,
            geometric_rejections=len(rejected),
            status=r.status,
        )
    )
endpoints = pd.DataFrame(rows)
endpoints.to_csv(public / "tables/endpoint_refinement.csv", index=False)
endpoint_geometry = []
for r in endpoints.itertuples():
    metadata = json.loads((root / "cases" / r.case / "geometry.json").read_text())
    endpoint_geometry.append(
        dict(
            case=r.case,
            mask_sha256=r.mask_sha256,
            grid_spacing=metadata["h"],
            holes=metadata["holes"],
        )
    )
(public / "endpoint_geometries.json").write_text(json.dumps(endpoint_geometry, indent=2) + "\n")

ledger = json.loads((root / "solve_ledger.json").read_text())
summary = json.loads((root / "summary.json").read_text())
extrema = (
    read("position_maps.csv").groupby(["family", "width", "Bi"]).gain_percent.agg(["min", "max"])
)
extrema.to_csv(public / "tables/position_gain_ranges.csv")
interactions = []
for (solver, Bi, n, mesh), group in v[v["case"].str.startswith("layout/")].groupby(
    ["solver", "Bi", "n", "mesh_area"], dropna=False
):
    for arrangement in ["near", "far"]:
        selected = [
            group[group["case"] == f"layout/{arrangement}_{suffix}"].iloc[0]
            for suffix in ["single_1", "single_2", "pair"]
        ]
        one, two, both = selected
        ir = both.Q_R - one.Q_R - two.Q_R + both.Q0
        inn = both.Q_N - one.Q_N - two.Q_N + both.Q0
        interactions.append(
            dict(
                solver=solver,
                Bi=Bi,
                n=n,
                mesh_area=mesh,
                arrangement=arrangement,
                reactive_interaction=ir,
                reflecting_interaction=inn,
                activation_interaction=ir - inn,
            )
        )
pd.DataFrame(interactions).to_csv(public / "tables/pair_refinement.csv", index=False)
analysis = dict(
    run_summary=summary,
    solver_seconds_by_kind={
        s: sum(r["seconds"] for r in ledger.values() if r["solver"] == s) for s in ["fvm", "fem"]
    },
    search_endpoint_geometries=endpoints.mask_sha256.nunique(),
    all_search_budgets_completed=bool((endpoints.status == "complete").all()),
    charged_search_solves=int(endpoints.charged_solves.sum()),
    geometric_search_rejections=int(endpoints.geometric_rejections.sum()),
    accepted_search_edits=int(endpoints.accepted_edits.sum()),
    best_coarse_endpoint=endpoints.loc[endpoints.Q_n100.idxmax()].to_dict(),
    best_fine_fem_endpoint=endpoints.loc[endpoints.Q_FEM_fine.idxmax()].to_dict(),
    maximum_fem_mesh_increment=float((endpoints.Q_FEM_fine - endpoints.Q_FEM_coarse).abs().max()),
    endpoint_fem_range=float(endpoints.Q_FEM_fine.max() - endpoints.Q_FEM_fine.min()),
    maximum_endpoint_fvm_fem_difference=float(
        (endpoints.Q_n400 - endpoints.Q_FEM_fine).abs().max()
    ),
    discarded_attempt=dict(
        status="Interrupted; unusable zero-filled solve ledger; not resumed or adopted",
        existing_cache_files=invocation["discarded_existing_cache_files"],
        solver_cost="Unknown; excluded from complete-run totals, additional incurred work",
    ),
    limitations=[
        "Not RL",
        "Selected finite endpoints; no global optimum",
        "Extra membrane vs unperforated baseline",
        "Same staircase exterior",
        "No anatomical inference",
        "Pair examples are not a separation-only sweep",
    ],
)
(public / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
print(json.dumps(analysis, indent=2), flush=True)
print("\nEndpoint refinement\n" + endpoints.to_string(index=False), flush=True)
print(
    "\nSix layouts validation\n"
    + v[
        (
            v["case"].isin(
                ["layout/" + s for s in ["spread_6", "proximal_6", "distal_6", "staggered_6"]]
            )
        )
        & (v.Bi == 1)
    ].to_string(index=False),
    flush=True,
)
