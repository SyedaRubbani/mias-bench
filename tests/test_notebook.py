"""The Colab notebook must stay valid and in sync with its generator.

A notebook that has drifted from the code it calls is worse than no notebook:
it fails on someone else's GPU after they have waited ten minutes for vLLM to
install. These tests check the structure, the syntax of every code cell, and
that the functions the notebook calls actually exist in the package.

Run:  python -m unittest tests.test_notebook -v
"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

NB_PATH = Path("notebooks/colab_t4_measurement.ipynb")


def code_cells(nb) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


class TestNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = json.loads(NB_PATH.read_text())

    def test_is_valid_notebook_json(self):
        self.assertEqual(self.nb["nbformat"], 4)
        self.assertGreater(len(self.nb["cells"]), 5)
        for cell in self.nb["cells"]:
            self.assertIn(cell["cell_type"], ("code", "markdown"))
            self.assertIsInstance(cell["source"], list)

    def test_requests_a_gpu_runtime(self):
        self.assertEqual(self.nb["metadata"]["accelerator"], "GPU")
        self.assertEqual(self.nb["metadata"]["colab"]["gpuType"], "T4")

    def test_every_code_cell_parses(self):
        for i, src in enumerate(code_cells(self.nb)):
            # Shell magics are not Python; strip them. Top-level await is
            # legal in a notebook, so parse inside an async wrapper.
            lines = [l for l in src.splitlines(keepends=True)
                     if not l.lstrip().startswith(("!", "%"))]
            body = "".join("    " + l for l in lines) or "    pass\n"
            try:
                ast.parse("async def _cell():\n" + body + "\n    pass\n")
            except SyntaxError as exc:
                self.fail(f"code cell {i} does not parse: {exc}")

    def test_t4_specific_flags_are_present(self):
        joined = "\n".join(code_cells(self.nb))
        # Turing has no bfloat16; omitting this makes the server fail to start.
        self.assertIn("--dtype", joined)
        self.assertIn("half", joined)
        self.assertIn("--enable-prefix-caching", joined)
        # The KV-pressure lever that reproduces the simulator's conditions.
        self.assertIn("--num-gpu-blocks-override", joined)

    def test_calls_only_functions_that_exist(self):
        import mias.measure as measure
        joined = "\n".join(code_cells(self.nb))
        for name in ("build_rounds", "run_measurement", "calibrate"):
            self.assertIn(name, joined, f"notebook should use {name}")
            self.assertTrue(hasattr(measure, name),
                            f"mias.measure is missing {name}")

    def test_records_hardware_provenance(self):
        joined = "\n".join(code_cells(self.nb))
        self.assertIn("nvidia-smi", joined)
        self.assertIn("provenance_t4.json", joined)

    def test_generator_reproduces_the_committed_notebook(self):
        """Regenerating must be a no-op, so the two cannot drift apart."""
        import subprocess
        import sys

        before = NB_PATH.read_text()
        subprocess.run([sys.executable, "tools/build_notebook.py"],
                       check=True, capture_output=True)
        self.assertEqual(before, NB_PATH.read_text(),
                         "notebook is out of sync with tools/build_notebook.py")


if __name__ == "__main__":
    unittest.main()
