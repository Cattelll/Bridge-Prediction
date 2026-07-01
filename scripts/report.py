"""
Generate a summary report from saved evaluation results.

Reads:  outputs/results/nb04_summary.json  (from notebook 04)
        outputs/results/test_comparison.csv (from run_pipeline.py)

Output: prints to stdout + saves outputs/results/REPORT.md

Run:    py -3.12 scripts/report.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "outputs" / "results"
SUMMARY_JSON = RESULTS_DIR / "nb04_summary.json"
TEST_CSV     = RESULTS_DIR / "test_comparison.csv"
REPORT_PATH  = RESULTS_DIR / "REPORT.md"


def _load_summary() -> dict | None:
    if SUMMARY_JSON.exists():
        return json.loads(SUMMARY_JSON.read_text())
    return None


def _load_test_csv() -> list[dict] | None:
    if not TEST_CSV.exists():
        return None
    import csv
    with open(TEST_CSV, newline="") as f:
        return list(csv.DictReader(f))


def _individual_jsons() -> list[dict]:
    # Prefer nb04_* files over legacy files; deduplicate by model name
    seen: dict[str, dict] = {}
    for path in sorted(RESULTS_DIR.glob("*_test.json")):
        try:
            d = json.loads(path.read_text())
            if "model" not in d or "accuracy" not in d:
                continue
            name = d["model"]
            # nb04_ prefixed files take priority
            if name not in seen or path.stem.startswith("nb04_"):
                seen[name] = d
        except Exception:
            pass
    return list(seen.values())


def _fval(v: str | float, fmt: str = ".4f") -> str:
    try:
        return format(float(v), fmt)
    except (ValueError, TypeError):
        return str(v)


def build_report() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []

    lines += [
        f"# Bridge Contract Prediction – Laporan Hasil",
        f"",
        f"*Dibuat otomatis pada {now}*",
        f"",
        f"---",
        f"",
    ]

    # ── Dataset info ──────────────────────────────────────────────────
    summary = _load_summary()
    if summary:
        d = summary["dataset"]
        lines += [
            "## Dataset",
            "",
            f"| Item | Nilai |",
            f"|------|-------|",
            f"| Total board (setelah dedup) | {d['n_total']:,} |",
            f"| Train / Val / Test | {d['n_train']:,} / {d['n_val']:,} / {d['n_test']:,} |",
            f"| Kelas kontrak | {d['n_classes']} |",
            f"| Fitur | {d['n_features']} |",
            f"",
        ]
    else:
        lines += ["## Dataset\n\n*Data tidak tersedia — jalankan notebook 04 terlebih dahulu.*\n"]

    # ── Test results ──────────────────────────────────────────────────
    results: list[dict] = []
    if summary and summary.get("test_results"):
        results = summary["test_results"]
    else:
        results = _individual_jsons()

    if not results:
        lines += ["## Hasil Test Set\n\n*Belum ada — jalankan `run_pipeline.py` atau notebook 04.*\n"]
    else:
        lines += [
            "## Hasil Test Set",
            "",
            "| Model | Accuracy | F1 Macro | F1 Weighted | Top-3 | Top-5 |",
            "|-------|----------|----------|-------------|-------|-------|",
        ]
        for r in results:
            lines.append(
                f"| {r['model']} "
                f"| {_fval(r.get('accuracy', 'n/a'))} "
                f"| {_fval(r.get('f1_macro', 'n/a'))} "
                f"| {_fval(r.get('f1_weighted', 'n/a'))} "
                f"| {_fval(r.get('top_3_accuracy', 'n/a'))} "
                f"| {_fval(r.get('top_5_accuracy', 'n/a'))} |"
            )
        lines.append("")

        best_acc = max(results, key=lambda r: float(r.get("accuracy", 0)))
        best_f1m = max(results, key=lambda r: float(r.get("f1_macro", 0)))
        best_f1w = max(results, key=lambda r: float(r.get("f1_weighted", 0)))
        best_t3  = max(results, key=lambda r: float(r.get("top_3_accuracy", 0)))

        lines += [
            "### Pemenang per Metrik",
            "",
            f"| Metrik | Model | Nilai |",
            f"|--------|-------|-------|",
            f"| Accuracy    | **{best_acc['model']}** | {_fval(best_acc['accuracy'])} |",
            f"| F1 Macro    | **{best_f1m['model']}** | {_fval(best_f1m['f1_macro'])} |",
            f"| F1 Weighted | **{best_f1w['model']}** | {_fval(best_f1w['f1_weighted'])} |",
            f"| Top-3 Acc   | **{best_t3['model']}** | {_fval(best_t3.get('top_3_accuracy', 0))} |",
            "",
        ]

    # ── Output files ──────────────────────────────────────────────────
    pngs = sorted(RESULTS_DIR.glob("*.png"))
    jsons = sorted(RESULTS_DIR.glob("*.json"))
    csvs  = sorted(RESULTS_DIR.glob("*.csv"))

    if pngs or jsons or csvs:
        lines += ["## Output Files", ""]
        if pngs:
            lines += [f"**Visualisasi ({len(pngs)} file):**"]
            for p in pngs:
                lines.append(f"- `outputs/results/{p.name}`")
            lines.append("")
        if jsons or csvs:
            lines += ["**Data:**"]
            for p in jsons + csvs:
                lines.append(f"- `outputs/results/{p.name}`")
            lines.append("")

    lines += [
        "---",
        "",
        "*Untuk generate ulang laporan ini:*",
        "```",
        "py -3.12 scripts/report.py",
        "```",
    ]

    return "\n".join(lines)


def main() -> None:
    if not RESULTS_DIR.exists():
        print(f"[ERROR] Direktori outputs/results/ belum ada.", file=sys.stderr)
        print(f"        Jalankan pipeline terlebih dahulu: py -3.12 scripts/run_pipeline.py", file=sys.stderr)
        sys.exit(1)

    report = build_report()

    print(report)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n── Laporan disimpan ke: {REPORT_PATH} ──", file=sys.stderr)


if __name__ == "__main__":
    main()
