from __future__ import annotations

from ._global_smoke import config_path_for, run_global_smoke

CONFIG_PATH = config_path_for("mlp_vae")


def run_smoke(
    *,
    config_path: str = str(CONFIG_PATH),
    output_root: str | None = None,
) -> dict:
    return run_global_smoke("mlp_vae", config_path=config_path, output_root=output_root)


if __name__ == "__main__":
    result = run_smoke()
    raise SystemExit(0 if result["ok"] else 1)
