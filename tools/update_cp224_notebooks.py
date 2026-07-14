"""Apply the CP22.4 model/configuration contract to the four global notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1] / "liquidez_2026_checkpoint_22_4"
NOTEBOOKS = tuple(ROOT.glob("code_03_GLOBAL_*.ipynb"))

REPLACEMENTS = {
    "22.3.2b": "22.4",
    '"enabled": True,\n    "target_dim_choices"': '"enabled": True,\n    "couple_temporal_encoders": True,\n    "target_dim_choices"',
    '"target_dim_choices": (16, 32, 64, 128)': '"target_dim_choices": (32, 64)',
    '"historical_dim_choices": (16, 32, 64, 128)': '"historical_dim_choices": (32, 64)',
    '"future_dim_choices": (16, 32, 64, 128)': '"future_dim_choices": (32, 64)',
    '"static_dim_choices": (8, 16, 32, 64)': '"static_dim_choices": (16, 32)',
    '"fusion_hidden_size_choices": (32, 64, 128, 256)': '"fusion_hidden_size_choices": (64, 128)',
    '"target_layers": {"minimum": 1, "maximum": 3}': '"target_layers": {"minimum": 1, "maximum": 2}',
    '"historical_layers": {"minimum": 1, "maximum": 3}': '"historical_layers": {"minimum": 1, "maximum": 2}',
    '"future_layers": {"minimum": 1, "maximum": 3}': '"future_layers": {"minimum": 1, "maximum": 2}',
    '"static_layers": {"minimum": 1, "maximum": 2}': '"static_layers": {"minimum": 1, "maximum": 1}',
    '"fusion_layers": {"minimum": 1, "maximum": 3}': '"fusion_layers": {"minimum": 1, "maximum": 2}',
    '"dropout": {"minimum": 0.0, "maximum": 0.35}': '"dropout": {"minimum": 0.0, "maximum": 0.20}',
    '"activations": ("relu", "gelu", "silu", "tanh")': '"activations": ("gelu", "silu")',
    "HPO_AUXILIARY_LOSS_WEIGHTS = True": "HPO_AUXILIARY_LOSS_WEIGHTS = False",
    "HPO_TRIALS = 80": "HPO_TRIALS = 36",
    "HPO_EPOCHS = 5": "HPO_EPOCHS = 4",
    "HPO_WINDOWS_PER_SERIES = 8": "HPO_WINDOWS_PER_SERIES = 6",
    "HPO_VALIDATION_WINDOWS_PER_SERIES = 5": "HPO_VALIDATION_WINDOWS_PER_SERIES = 4",
    "HPO_FINALISTS = 8": "HPO_FINALISTS = 4",
    "HPO_FIDELITY_EPOCHS = 12": "HPO_FIDELITY_EPOCHS = 8",
    "HPO_FIDELITY_WINDOWS_PER_SERIES = 16": "HPO_FIDELITY_WINDOWS_PER_SERIES = 12",
    "# HPO de dos fidelidades, ampliado para el espacio multi-encoder de Stage 2.3.": "# HPO compacto: capacidad suficiente sin combinaciones redundantes.",
    "# Decoder residual local: y_pred = y_global + delta_local.": "# Política interna CP22.4: refinamiento residual autoregresivo causal.",
    "    enabled=USE_LOCAL_RESIDUAL_DECODER,": "    enabled=USE_LOCAL_RESIDUAL_DECODER,\n    autoregressive=True,",
}

# These are model-policy constants, not user-facing Colab controls.
INTERNAL_PREFIXES = (
    "USE_STATIC_CONTEXT =",
    "USE_MODALITY_SPECIFIC_ENCODERS =",
    "USE_AUXILIARY_AUTOENCODER =",
    "USE_LOCAL_RESIDUAL_DECODER =",
    "LOCAL_RESIDUAL_",
    "GLOBAL_AUX_ALPHA =",
    "USE_EVENT_HEAD =",
    "USE_MAGNITUDE_HEAD =",
    "USE_DIRECTION_HEAD =",
    "USE_AUXILIARY_LOSS_BLOCK =",
    "AUXILIARY_LOSS_WEIGHT =",
    "EVENT_LOSS_SHARE =",
    "MAGNITUDE_LOSS_SHARE =",
    "DIRECTION_LOSS_SHARE =",
    "HPO_AUXILIARY_LOSS_WEIGHTS =",
    "EVENT_LOSS_WEIGHT =",
    "MAGNITUDE_LOSS_WEIGHT =",
    "DIRECTION_LOSS_WEIGHT =",
    "AUXILIARY_HEAD_",
    "EVENT_THRESHOLD =",
    "MAGNITUDE_TRANSFORM =",
    "HPO_WINDOWS_PER_SERIES =",
    "HPO_VALIDATION_WINDOWS_PER_SERIES =",
    "HPO_BATCH =",
    "HPO_REDUCTION_FACTOR =",
    "HPO_FINALISTS =",
    "HPO_FIDELITY_EPOCHS =",
    "HPO_FIDELITY_WINDOWS_PER_SERIES =",
    "NONFINITE_",
    "LOSS_FUNCTION =",
    "SELECTION_METRIC =",
)


def update_source(source: str) -> str:
    for old, new in REPLACEMENTS.items():
        source = source.replace(old, new)
    lines = []
    for line in source.splitlines(keepends=True):
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in INTERNAL_PREFIXES):
            line = line.replace("  # @param", "  # internal")
        lines.append(line)
    return "".join(lines)


def main() -> None:
    if len(NOTEBOOKS) != 4:
        raise RuntimeError(f"Expected four global notebooks, found {len(NOTEBOOKS)}")
    for path in NOTEBOOKS:
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            cell.source = update_source(cell.source)
            if cell.cell_type == "code":
                cell.execution_count = None
                cell.outputs = []
        nbformat.write(notebook, path)


if __name__ == "__main__":
    main()
