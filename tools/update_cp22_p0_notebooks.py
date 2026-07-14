import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] / "liquidez_2026_checkpoint_22_4"
METRICS = '''\n# P0 acceptance diagnostics.\np0_auxiliary_metrics = manager.evaluate_p0_auxiliary_heads(test_unseen_dataset, batch_size=FORECAST_BATCH_SIZE, device=DEVICE)\np0_interval_calibration = manager.backtest_results["interval_calibration_by_horizon"]\np0_patience = manager.p0_patience_diagnostic()\ndisplay(p0_auxiliary_metrics)\ndisplay(p0_interval_calibration)\nprint(json.dumps(p0_patience, indent=2))\n'''
SAVE = '''\n# P0 evidence before checkpoint 22.4.\np0_auxiliary_metrics.to_parquet(reports_dir / "p0_auxiliary_head_metrics.parquet", index=False)\np0_interval_calibration.to_parquet(reports_dir / "p0_interval_calibration_by_horizon.parquet", index=False)\nwrite_json(reports_dir / "p0_patience_diagnostic.json", p0_patience)\nwrite_json(reports_dir / "p0_residual_ablation_protocol.json", {"same_seed": True, "same_split": True, "vary_only": "USE_LOCAL_RESIDUAL_DECODER", "selection_metric": SELECTION_METRIC})\n'''
GATE = '''\nassert not p0_auxiliary_metrics.empty\nassert not p0_interval_calibration.empty\nassert (reports_dir / "p0_auxiliary_head_metrics.parquet").is_file()\nassert (reports_dir / "p0_interval_calibration_by_horizon.parquet").is_file()\nassert (reports_dir / "p0_patience_diagnostic.json").is_file()\nassert (reports_dir / "p0_residual_ablation_protocol.json").is_file()\nprint("✅ P0 diagnostics and residual-ablation protocol persisted")\n'''
def add(cell, marker, text):
    source = ''.join(cell['source'])
    if marker not in source: cell['source'] = (source + text).splitlines(keepends=True)
for path in ROOT.glob('code_03_GLOBAL_*.ipynb'):
    notebook = json.loads(path.read_text())
    add(notebook['cells'][16], 'p0_auxiliary_metrics =', METRICS)
    add(notebook['cells'][18], 'p0_auxiliary_head_metrics.parquet', SAVE)
    add(notebook['cells'][20], 'P0 diagnostics and residual', GATE)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
