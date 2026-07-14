import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(ROOT.glob("code_03_GLOBAL_*.ipynb"))


def _text(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])


def test_hotfix_removes_stage_from_user_surface_and_circular_reference() -> None:
    assert len(NOTEBOOKS) == 4
    for path in NOTEBOOKS:
        text = _text(path)
        assert "GTRM_STAGE =" not in text
        assert "active_config.budget.hpo_timeout_seconds" not in text
        assert "hpo_timeout_seconds=HPO_TIMEOUT_SECONDS" in text


def test_hotfix_builds_pydantic_blocks_before_active_configuration() -> None:
    for path in NOTEBOOKS:
        text = _text(path)
        active_at = text.index("active_config = GlobalActiveConfiguration(")
        for marker in (
            "feature_config = ModelFeatureConfig(",
            "modality_defaults_config = ModalityEncoderDefaults(",
            "modality_hpo_config = ModalityEncoderHPOSpace(",
            "residual_config = ResidualDecoderConfig(",
            "auxiliary_config = AuxiliaryHeadsConfig(",
            "budget_config = TrainingBudgetConfig(",
            "inference_config = InferenceConfig(",
        ):
            assert text.index(marker) < active_at


def test_hotfix_notebooks_are_clean_and_compile() -> None:
    for path in NOTEBOOKS:
        nb = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") != "code":
                continue
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []
            source = "".join(cell.get("source", []))
            if source.strip():
                compile(source, f"{path.name}:cell{index}", "exec")
