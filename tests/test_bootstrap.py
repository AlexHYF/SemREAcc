from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BootstrapTests(unittest.TestCase):
    def test_cuda_124_keeps_pypi_available_for_nvidia_dependencies(self) -> None:
        script = (ROOT / "scripts" / "bootstrap_vllm.sh").read_text(encoding="utf-8")
        cu124_block = script.split("  cu124)", 1)[1].split("    ;;", 1)[0]

        self.assertIn("--torch-backend=cu124", cu124_block)
        self.assertNotIn("--index-url https://download.pytorch.org/whl/cu124", cu124_block)

    def test_small_ensemble_uses_dedicated_transformers_environment(self) -> None:
        bootstrap = (ROOT / "scripts" / "bootstrap_transformers.sh").read_text(
            encoding="utf-8"
        )
        runner = (ROOT / "scripts" / "run_small_model_ensemble.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("transformers[serving]", bootstrap)
        self.assertIn("TRANSFORMERS_REVISION", bootstrap)
        self.assertIn("--torch-backend=cu124", bootstrap)
        self.assertNotIn("\n  vllm", bootstrap)
        self.assertIn("INFERENCE_BACKEND=transformers", runner)


if __name__ == "__main__":
    unittest.main()
