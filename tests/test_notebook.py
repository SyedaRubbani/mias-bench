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

    def test_setup_does_not_swallow_failures(self):
        """A failed clone must stop the notebook, not be discovered later."""
        joined = "\n".join(code_cells(self.nb))
        self.assertNotIn("|| true", joined,
                         "setup must not swallow a failed clone")
        self.assertNotIn("2>/dev/null", joined,
                         "setup must not hide error output")
        # The repo cells must fail loudly if the working directory is wrong.
        self.assertIn("raise", joined)

    def test_verifies_vllm_imports_before_proceeding(self):
        joined = "\n".join(code_cells(self.nb))
        self.assertIn("import torch, vllm", joined,
                      "notebook should smoke-test the install in a subprocess")
        self.assertIn("returncode", joined)

    def test_offers_an_upload_path_when_the_repo_is_not_pushed(self):
        joined = "\n".join(code_cells(self.nb))
        self.assertIn("files.upload", joined)
        self.assertIn("tarfile", joined)

    def test_launch_probes_flags_rather_than_assuming_them(self):
        """vLLM's CLI changes between releases; the notebook must adapt."""
        joined = "\n".join(code_cells(self.nb))
        self.assertIn("supported_flags", joined,
                      "launch() should read --help and drop unknown flags")
        self.assertIn("--help", joined)
        self.assertIn("DROPPED_FLAGS", joined,
                      "dropped flags must be recorded, not silently discarded")

    def test_makes_the_repo_importable_not_just_current(self):
        """chdir alone does not work in Colab: sys.path holds /content."""
        joined = "\n".join(code_cells(self.nb))
        self.assertIn("sys.path.insert", joined,
                      "cells must put the repo on sys.path, not only chdir")
        self.assertIn("def use_repo", joined)
        self.assertIn("repo_is_valid", joined,
                      "an empty leftover directory must not pass for a checkout")
        self.assertIn("find_repo", joined,
                      "a nested extract must still be locatable")

    def test_later_cells_can_be_rerun_standalone(self):
        """After a Colab disconnect, any cell should recover on its own."""
        cells = code_cells(self.nb)
        needs_repo = [c for c in cells if "from mias." in c or "import mias" in c]
        self.assertGreaterEqual(len(needs_repo), 3)
        for c in needs_repo:
            if "def use_repo" in c:
                continue        # the bootstrap cell defines it
            self.assertIn("use_repo(", c,
                          "a cell importing mias must call use_repo() first")

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
