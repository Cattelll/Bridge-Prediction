"""
Generate a summary report from saved evaluation results.

Reads:  outputs/results/nb04_summary.json        (notebook 04 — 3 baseline resmi)
        outputs/results/nb06_final_comparison.csv (notebook 06 — 8 kandidat)

Target biner `matches_par_contract` (sejak 2026-09-18): metrik yang
dilaporkan accuracy, F1 macro/weighted/positif, ROC-AUC, PR-AUC.

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
NB06_CSV     = RESULTS_DIR / "nb06_final_comparison.csv"

# (kunci, judul kolom) metrik biner yang dilaporkan
METRICS = [
    ("accuracy",          "Accuracy"),
    ("f1_macro",          "F1 Macro"),
    ("f1_weighted",       "F1 Weighted"),
    ("f1_positive",       "F1 Positif"),
    ("roc_auc",           "ROC-AUC"),
    ("average_precision", "PR-AUC"),
]
REPORT_PATH  = RESULTS_DIR / "REPORT.md"


def _load_summary() -> dict | None:
    if SUMMARY_JSON.exists():
        return json.loads(SUMMARY_JSON.read_text())
    return None


def _load_csv(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    import csv
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _metric_table(results: list[dict]) -> list[str]:
    lines = [
        "| Model | " + " | ".join(t for _, t in METRICS) + " |",
        "|-------|" + "|".join("-" * (len(t) + 2) for _, t in METRICS) + "|",
    ]
    for r in results:
        cells = " | ".join(_fval(r.get(k, "n/a")) for k, _ in METRICS)
        lines.append(f"| {r['model']} | {cells} |")
    lines.append("")
    return lines


def _winners(results: list[dict]) -> list[str]:
    lines = [
        "| Metrik | Model | Nilai |",
        "|--------|-------|-------|",
    ]
    for key, title in METRICS:
        scored = [r for r in results if _is_num(r.get(key))]
        if not scored:
            continue
        best = max(scored, key=lambda r: float(r[key]))
        lines.append(f"| {title} | **{best['model']}** | {_fval(best[key])} |")
    lines.append("")
    return lines


def _is_num(v) -> bool:
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False


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
        f"# Klasifikasi Kecocokan Kontrak Par pada Contract Bridge – Laporan Hasil",
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
            f"| Target | `matches_par_contract` (biner, {d['n_classes']} kelas) |",
            f"| Fitur | {d['n_features']} |",
        ]
        rate = d.get("matches_par_contract_rate_test")
        if rate is not None:
            lines += [
                f"| Kelas positif di test set | {rate:.1%} "
                f"(baseline mayoritas = {1 - rate:.1%} accuracy) |",
            ]
        lines.append("")
    else:
        lines += ["## Dataset\n\n*Data tidak tersedia — jalankan notebook 04 terlebih dahulu.*\n"]

    # ── Test results ──────────────────────────────────────────────────
    results: list[dict] = []
    if summary and summary.get("test_results"):
        results = summary["test_results"]
    else:
        results = _individual_jsons()

    if not results:
        lines += ["## Hasil Test Set\n\n*Belum ada — jalankan notebook 04.*\n"]
    else:
        lines += ["## Hasil Test Set — Baseline Resmi", ""]
        lines += _metric_table(results)
        lines += ["### Pemenang per Metrik (baseline)", ""]
        lines += _winners(results)

    # ── Kandidat notebook 06 ──────────────────────────────────────────
    candidates = _load_csv(NB06_CSV)
    if candidates:
        lines += [
            f"## Semua Kandidat — notebook 06 ({len(candidates)} model)",
            "",
            "*Diurutkan F1 macro. Kandidat retuning (Exp A/B/C/E/F) belum "
            "dipromosikan ke baseline resmi.*",
            "",
        ]
        lines += _metric_table(candidates)
        lines += ["### Pemenang per Metrik (semua kandidat)", ""]
        lines += _winners(candidates)

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
        print(f"        Jalankan notebooks/01→06 terlebih dahulu.", file=sys.stderr)
        sys.exit(1)

    report = build_report()

    print(report)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n── Laporan disimpan ke: {REPORT_PATH} ──", file=sys.stderr)


if __name__ == "__main__":
    main()
