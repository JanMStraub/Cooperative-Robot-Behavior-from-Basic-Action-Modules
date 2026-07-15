#!/usr/bin/env python3
"""Build the per-run benchmark tables for the thesis appendix.

Reads benchmark_results/bN/<model>/*.json and writes three combined,
page-breaking LaTeX tables to --out: system_table.tex (B1-B10),
ablation_table.tex (B11-B16) and b17_combined.tex (B17). The document
must load ltablex.

    python -m tools.GenThesisTables [--root DIR] [--out DIR]
"""

import argparse
import glob
import json
import os

MODELS = [
    "magistral-small-2509",
    "ministral-3-14b-reasoning",
    "qwen3-vl-8b",
    "qwen3-vl-30b",
    "gemma-4-e2b",
    "gemma-4-e4b",
]
GLS = {
    "magistral-small-2509": "\\gls{magi}",
    "ministral-3-14b-reasoning": "\\gls{mini}",
    "qwen3-vl-8b": "\\gls{qwen38}",
    "qwen3-vl-30b": "\\gls{qwen330}",
    "gemma-4-e2b": "\\gls{gema42}",
    "gemma-4-e4b": "\\gls{gema44}",
}
BLABEL = {
    1: "Navigate to Object.",
    2: "Sequential Navigation.",
    3: "Navigate and Lift.",
    4: "Pick and Place.",
    5: "Pose-Aware Grasp.",
    6: "Robot Handoff.",
    7: "Dual-Robot Reorient.",
    8: "Heterogeneous Chain.",
    9: "Impossible Task.",
    10: "Parallel Independent Task.",
}
ABL = {
    11: (
        "\\gls{rag} Ablation",
        [("enabled", "w/ \\gls{rag}."), ("disabled", "w/o \\gls{rag}.")],
    ),
    12: (
        "Reflection Abl.",
        [("enabled", "w/ Reflection."), ("disabled", "w/o Reflection.")],
    ),
    13: (
        "Negotiation Abl.",
        [("enabled", "w/ Negotiation."), ("disabled", "w/o Negotiation.")],
    ),
    14: (
        "\\gls{kg} Ablation",
        [("enabled", "w/ \\gls{kg}."), ("disabled", "w/o \\gls{kg}.")],
    ),
    15: (
        "\\gls{vgn} Ablation",
        [("enabled", "w/ \\gls{vgn}."), ("disabled", "w/o \\gls{vgn}.")],
    ),
    16: ("\\gls{ros} vs Unity", [("ros", "w/ \\gls{ros}."), ("unity", "w/ Unity.")]),
}

RUN_HEADER = [
    "\t\\textbf{Task} & \\textbf{Succ.} & \\textbf{Total} & \\textbf{Ops} & \\textbf{Avg} & \\textbf{Slowest} & \\textbf{Halluc.} & \\textbf{Reflex.} & \\textbf{Negot.} \\\\",
]
COL_LEGEND = (
    " Columns: Succ.\\ = per-run success rate; Total = total duration (s); "
    "Ops = operations succeeded/executed; Avg and Slowest = mean and slowest "
    "step duration (ms); Halluc.\\ = hallucinated operations; Reflex.\\ = "
    "reflection recoveries; Negot.\\ = negotiation rounds."
)
RUN_COLSPEC = "@{}l*{8}{>{\\centering\\arraybackslash}X}@{}"
RUN_NCOL = 9


def load(root, bench, model):
    return [
        json.load(open(f)) for f in sorted(glob.glob(f"{root}/{bench}/{model}/*.json"))
    ]


def run_sr(r):
    return r.get("success_rate", 0.0) or 0.0


def slowest(r):
    ds = [s.get("duration_ms", 0) or 0 for s in (r.get("steps") or [])]
    return max(ds) if ds else 0


def fmt_dur(ms):
    return "0.0s" if not ms else f"{ms / 1000:.2f}s"


def fmt_pct(x):
    return f"{x:.3f}"


def min_idx(values):
    nz = [v for v in values if v > 0]
    return values.index(min(nz)) if nz else -1


def longtabx(colspec, header_rows, body, caption, label, ncol):
    head = "\n".join(header_rows)
    return "\n".join(
        [
            "\\begingroup",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\setlength{\\extrarowheight}{0pt}",
            "\\renewcommand{\\arraystretch}{1.0}",
            "\\convertXColumns",
            f"\\begin{{tabularx}}{{\\linewidth}}{{{colspec}}}",
            f"\t\\caption{{{caption}}}\\label{{{label}}}\\\\",
            "\t\\toprule",
            head,
            "\t\\midrule",
            "\t\\endfirsthead",
            "\t\\toprule",
            head,
            "\t\\midrule",
            "\t\\endhead",
            "\t\\midrule",
            f"\t\\multicolumn{{{ncol}}}{{r}}{{\\footnotesize Continued on next page}} \\\\",
            "\t\\endfoot",
            "\t\\bottomrule",
            "\t\\endlastfoot",
            body,
            "\\end{tabularx}",
            "\\keepXColumns",
            "\\endgroup",
        ]
    )


def run_cells(r, totals, avgs, slows, i, dry=False):
    oe = r.get("ops_executed", 0) or 0
    os_ = r.get("ops_succeeded", 0) or 0
    if dry:
        tcell = acell = scell = "-"
    else:
        tcell, acell, scell = (
            fmt_dur(totals[i]),
            f"{int(avgs[i])}ms",
            f"{int(slows[i])}ms",
        )
        if i == min_idx(totals):
            tcell = f"\\textbf{{{tcell}}}"
        if i == min_idx(avgs):
            acell = f"\\textbf{{{acell}}}"
        if i == min_idx(slows):
            scell = f"\\textbf{{{scell}}}"
    h = r.get("hallucinated_ops", 0) or 0
    rec = r.get("reflection_recoveries", 0) or 0
    ng = r.get("negotiation_rounds", 0) or 0
    return f"{fmt_pct(run_sr(r))} & {tcell} & {os_}/{oe} & {acell} & {scell} & {h} & {rec} & {ng}"


def block_rows(label, rows):
    # \\* keeps the block (and its multirow label) together across a page break.
    n = len(rows)
    out = [f"\t\\multirow{{{n}}}{{*}}{{{label}}}"]
    for i, cells in enumerate(rows):
        out.append(f"\t              & {cells} " + ("\\\\*" if i < n - 1 else "\\\\"))
    return out


def measure(runs):
    totals = [r.get("total_duration_ms", 0) or 0 for r in runs]
    avgs = [r.get("avg_step_duration_ms", 0) or 0 for r in runs]
    slows = [slowest(r) for r in runs]
    return totals, avgs, slows


def system_table(root):
    body = []
    for b in range(1, 11):
        for mi, model in enumerate(MODELS):
            runs = load(root, f"b{b}", model)
            totals, avgs, slows = measure(runs)
            cells = [run_cells(r, totals, avgs, slows, i) for i, r in enumerate(runs)]
            label = f"\\makecell[l]{{\\textbf{{B{b}:}} {GLS[model]} \\\\ {BLABEL[b]}}}"
            body += block_rows(label, cells)
            heavy = mi == len(MODELS) - 1
            body.append("\t\\midrule[\\heavyrulewidth]" if heavy else "\t\\midrule")
    while body and body[-1].strip().startswith("\\midrule"):
        body.pop()
    caption = (
        "Detailed overview of all system benchmark runs (B1--B10) for all six "
        "models, five runs each. The success-rate column is the per-run "
        "\\texttt{success\\_rate} recorded by the harness (operation-level for "
        "these single-task benchmarks); task-level outcomes are analyzed in "
        "Chapter~\\ref{evaluation}." + COL_LEGEND
    )
    return longtabx(
        RUN_COLSPEC,
        RUN_HEADER,
        "\n".join(body),
        caption,
        "tab:system_benchmark_runs",
        RUN_NCOL,
    )


def ablation_table(root):
    body = []
    for b in range(11, 17):
        title, conds = ABL[b]
        body.append(f"\t% ===== B{b}: {title} =====")
        for mi, model in enumerate(MODELS):
            for ci, (cond, clabel) in enumerate(conds):
                runs = [
                    r
                    for r in load(root, f"b{b}", model)
                    if (r.get("ablation") or {}).get("condition") == cond
                ]
                totals, avgs, slows = measure(runs)
                dry = all(t == 0 for t in totals)
                cells = [
                    run_cells(r, totals, avgs, slows, i, dry=dry)
                    for i, r in enumerate(runs)
                ]
                label = f"\\makecell[l]{{\\textbf{{B{b}:}} {GLS[model]} \\\\ {title} \\\\ {clabel}}}"
                body += block_rows(label, cells)
                heavy = mi == len(MODELS) - 1 and ci == len(conds) - 1
                body.append("\t\\midrule[\\heavyrulewidth]" if heavy else "\t\\midrule")
    while body and body[-1].strip().startswith("\\midrule"):
        body.pop()
    caption = (
        "Detailed overview of the isolated impacts for optional system components "
        "(Benchmarks B11 to B16), reported for all six models, five runs per "
        "condition. The success-rate column is the per-run \\texttt{success\\_rate} "
        "recorded by the harness (task-level for the multi-task ablation runs); a "
        "dash marks the dry-run benchmarks (B11, B14) where no operations were dispatched."
        + COL_LEGEND
    )
    return longtabx(
        RUN_COLSPEC,
        RUN_HEADER,
        "\n".join(body),
        caption,
        "tab:combined_ablation_benchmark_runs",
        RUN_NCOL,
    )


def b17_combined_table(root):
    header = [
        "\t\\textbf{Model} & \\textbf{Gate acc.} & \\textbf{False acc.} & \\textbf{False rej.} & \\textbf{Slot succ.} & \\textbf{First-attempt} \\\\",
    ]
    body = []
    for model in MODELS:
        cells = []
        for r in load(root, "b17", model):
            sg = r["per_op_stats"]["safety_gate"]
            gn = r["per_op_stats"]["generation"]
            cells.append(
                f"{sg['accuracy']:.3f} & {sg['false_accept_rate']:.3f} & {sg['false_reject_rate']:.3f} & {gn['slot_success_rate']:.3f} & {gn['first_attempt_rate']:.3f}"
            )
        body += block_rows(GLS[model], cells)
        body.append("\t\\midrule")
    while body and body[-1].strip() == "\\midrule":
        body.pop()
    caption = (
        "Detailed overview of all B17 AutoRT runs across the six models (five runs "
        "each, 19 labelled safety tasks and 48 generation slots per run). Gate "
        "accuracy, false-accept, and false-reject rates are identical across a "
        "model's runs because the semantic layer runs at temperature~0 (confusion "
        "matrix 11/2/6/0 for \\gls{magi}, \\gls{mini}, \\gls{qwen38}, and "
        "\\gls{qwen330}; 11/4/4/0 for the two Gemma~4 models); slot success and "
        "first-attempt validity vary with sampling."
    )
    colspec = "@{}l*{5}{>{\\centering\\arraybackslash}X}@{}"
    return longtabx(
        colspec, header, "\n".join(body), caption, "tab:b17_benchmark_runs", 6
    )


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root",
        default=os.path.normpath(os.path.join(here, "..", "benchmark_results")),
    )
    ap.add_argument("--out", default=os.path.join(here, "..", "thesis_tables"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    open(os.path.join(args.out, "system_table.tex"), "w").write(system_table(args.root))
    open(os.path.join(args.out, "ablation_table.tex"), "w").write(
        ablation_table(args.root)
    )
    open(os.path.join(args.out, "b17_combined.tex"), "w").write(
        b17_combined_table(args.root)
    )
    print(f"Wrote LaTeX fragments to {args.out}")


if __name__ == "__main__":
    main()
