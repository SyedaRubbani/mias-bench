"""Tests for the live-measurement client, against a mock vLLM server.

The point of these tests is that the code which will run on a GPU is exercised
here, on CPU, with no GPU present. The mock speaks the same streaming
completions protocol and the same Prometheus endpoint, and it deliberately
misorders responses so the tests can prove the client records what actually
happened rather than what was asked for.

Run:  python -m unittest tests.test_measure -v
"""

from __future__ import annotations

import asyncio
import unittest

from aiohttp import web

from mias.measure import (
    build_rounds,
    calibrate,
    measure_round,
    parse_prometheus,
    run_measurement,
)
from mias.metrics import initiative_displacement, order_fidelity_tau


class MockServer:
    """Minimal streaming completions server with controllable delays.

    `delay_for(prompt)` decides how long a request waits before its first
    token, which is how the tests stand in for a real scheduler's decisions.
    """

    def __init__(self, delay_for=None, tokens: int = 3):
        self.delay_for = delay_for or (lambda prompt: 0.01)
        self.tokens = tokens
        self.requests = []
        self.runner = None
        self.port = None

    async def _completions(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.requests.append(body)
        resp = web.StreamResponse(
            status=200, headers={"Content-Type": "text/event-stream"}
        )
        await resp.prepare(request)
        await asyncio.sleep(self.delay_for(body["prompt"]))
        for _ in range(self.tokens):
            await resp.write(b'data: {"choices":[{"text":"x"}]}\n\n')
            await asyncio.sleep(0.002)
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    async def _metrics(self, _request: web.Request) -> web.Response:
        n = len(self.requests)
        return web.Response(text=(
            "# HELP whatever\n"
            f'vllm:prefix_cache_queries_total{{model="m"}} {10 * n}\n'
            f'vllm:prefix_cache_hits_total{{model="m"}} {7 * n}\n'
            f'vllm:num_preemptions_total{{model="m"}} {n // 4}\n'
        ))

    async def start(self) -> str:
        app = web.Application()
        app.router.add_post("/v1/completions", self._completions)
        app.router.add_get("/metrics", self._metrics)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{self.port}"

    async def stop(self) -> None:
        if self.runner:
            await self.runner.cleanup()


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestPrometheus(unittest.TestCase):
    def test_labels_are_stripped_and_summed(self):
        text = (
            "# comment\n"
            'vllm:num_preemptions_total{model="a"} 2.0\n'
            'vllm:num_preemptions_total{model="b"} 3.0\n'
            "garbage line\n"
        )
        self.assertEqual(parse_prometheus(text)["vllm:num_preemptions_total"], 5.0)


class TestRoundBuilder(unittest.TestCase):
    def test_prompts_share_the_intended_prefixes(self):
        rounds = build_rounds(n_sessions=2, rounds_per_session=1)
        p = [rounds[0].prompts[t.turn_id] for t in rounds[0].turns]
        # Every prompt in a deployment shares the system prompt.
        head = p[0][:200]
        self.assertTrue(all(x.startswith(head) for x in p))
        # Prompts in different sessions diverge before the tail.
        q = rounds[1].prompts[rounds[1].turns[0].turn_id]
        self.assertNotEqual(p[0], q)

    def test_designed_ranks_are_contiguous(self):
        spec = build_rounds(n_sessions=1, rounds_per_session=1)[0]
        ranks = sorted(t.designed_rank for t in spec.turns)
        self.assertEqual(ranks, list(range(len(ranks))))


class TestMeasurement(unittest.TestCase):
    def test_concurrent_mode_records_server_order_not_designed_order(self):
        async def scenario():
            # Delay is keyed off the turn tail, so the server answers in an
            # order unrelated to the designed ranks.
            def delay_for(prompt: str) -> float:
                for i, d in enumerate([0.08, 0.02, 0.06, 0.01]):
                    if f"TURN {i}." in prompt:
                        return d
                return 0.01

            server = MockServer(delay_for=delay_for)
            url = await server.start()
            try:
                rounds = build_rounds(n_sessions=1, rounds_per_session=1,
                                      max_tokens=3)
                turns, delta = await run_measurement(
                    url, "mock", rounds, mode="concurrent", warmup_rounds=0
                )
            finally:
                await server.stop()
            return turns, delta

        turns, delta = run(scenario())
        self.assertTrue(all(t.first_token_time is not None for t in turns))
        by_rank = {t.designed_rank: t.first_token_time for t in turns}
        # Rank 3 was served fastest, so the designed order was not realised.
        self.assertLess(by_rank[3], by_rank[0])
        self.assertLess(order_fidelity_tau(turns), 0.5)
        self.assertGreater(delta["vllm:prefix_cache_queries_total"], 0)

    def test_gated_mode_restores_order_and_reopens_the_window(self):
        async def scenario():
            server = MockServer(delay_for=lambda p: 0.01)
            url = await server.start()
            try:
                rounds = build_rounds(n_sessions=1, rounds_per_session=1,
                                      max_tokens=3)
                turns, _ = await run_measurement(
                    url, "mock", rounds, mode="gated",
                    gate_seconds=0.20, warmup_rounds=0
                )
            finally:
                await server.stop()
            return turns

        turns = run(scenario())
        self.assertAlmostEqual(order_fidelity_tau(turns), 1.0, places=6)
        self.assertEqual(initiative_displacement(turns, threshold=0.15), 0.0)

    def test_priority_mode_sends_the_priority_field(self):
        async def scenario():
            server = MockServer()
            url = await server.start()
            try:
                rounds = build_rounds(n_sessions=1, rounds_per_session=1,
                                      max_tokens=3)
                await run_measurement(url, "mock", rounds, mode="priority",
                                      warmup_rounds=0)
            finally:
                await server.stop()
            return server.requests

        reqs = run(scenario())
        self.assertTrue(all("priority" in r for r in reqs))
        self.assertEqual(sorted(r["priority"] for r in reqs), [0, 1, 2, 3])

    def test_warmup_rounds_are_excluded_from_results(self):
        async def scenario():
            server = MockServer()
            url = await server.start()
            try:
                rounds = build_rounds(n_sessions=2, rounds_per_session=1,
                                      max_tokens=3)
                turns, _ = await run_measurement(url, "mock", rounds,
                                                 warmup_rounds=1)
            finally:
                await server.stop()
            return turns, rounds

        turns, rounds = run(scenario())
        self.assertEqual(len(turns), len(rounds[1].turns))

    def test_unknown_mode_is_rejected(self):
        async def scenario():
            server = MockServer()
            url = await server.start()
            try:
                import aiohttp
                spec = build_rounds(n_sessions=1, rounds_per_session=1)[0]
                async with aiohttp.ClientSession() as s:
                    with self.assertRaises(ValueError):
                        await measure_round(s, url, "mock", spec, mode="nope")
            finally:
                await server.stop()

        run(scenario())


class TestCalibration(unittest.TestCase):
    def test_recovers_a_known_linear_cost(self):
        async def scenario():
            # TTFT = 0.0001 * words + 0.05, exactly the model being fitted.
            def delay_for(prompt: str) -> float:
                return 1e-4 * len(prompt.split()) + 0.05

            server = MockServer(delay_for=delay_for, tokens=1)
            url = await server.start()
            try:
                return await calibrate(url, "mock",
                                       prompt_words=(200, 800, 1600),
                                       repeats=3)
            finally:
                await server.stop()

        fit = run(scenario())
        self.assertGreater(fit["r2"], 0.95)
        self.assertAlmostEqual(fit["alpha_s_per_word"], 1e-4, places=4)
        self.assertGreater(fit["beta_s"], 0.0)


if __name__ == "__main__":
    unittest.main()


class TestRunScript(unittest.TestCase):
    """The single-entry-point script must be importable and self-describing.

    It cannot be executed here (no GPU), but everything that is not a vLLM
    call should still be exercised so a typo does not surface only on Colab.
    """

    def test_imports_and_exposes_a_cli(self):
        import importlib
        import sys
        sys.path.insert(0, ".")
        mod = importlib.import_module("scripts.run_t4")
        for name in ("launch", "shutdown", "filter_flags", "supported_flags",
                     "measure_all", "write_outputs", "main"):
            self.assertTrue(hasattr(mod, name), f"run_t4 is missing {name}")

    def test_filter_flags_drops_flag_and_value_together(self):
        import importlib
        import sys
        sys.path.insert(0, ".")
        mod = importlib.import_module("scripts.run_t4")
        mod.DROPPED_FLAGS.clear()
        real = mod.supported_flags
        mod.supported_flags = lambda: frozenset(["--model", "--port"])
        try:
            kept = mod.filter_flags(["py", "-m", "s", "--model", "M",
                                     "--num-gpu-blocks-override", "512",
                                     "--port", "8000"])
        finally:
            mod.supported_flags = real
        self.assertEqual(kept, ["py", "-m", "s", "--model", "M",
                                "--port", "8000"])
        self.assertNotIn("512", kept, "a dropped flag must take its value with it")
        self.assertEqual(mod.DROPPED_FLAGS, ["--num-gpu-blocks-override"])

    def test_help_runs_without_a_gpu(self):
        import subprocess
        import sys
        out = subprocess.run([sys.executable, "scripts/run_t4.py", "--help"],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("--smoke", out.stdout)


class TestVisualization(unittest.TestCase):
    """The explorer must stay buildable and consistent with the data it reads."""

    def test_template_and_build_script_exist(self):
        import pathlib
        self.assertTrue(pathlib.Path("viz/explorer.template.html").exists())
        self.assertTrue(pathlib.Path("tools/build_viz.py").exists())

    def test_build_is_reproducible_and_self_contained(self):
        import pathlib
        import subprocess
        import sys
        out = subprocess.run([sys.executable, "tools/build_viz.py"],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        html = pathlib.Path("viz/index.html").read_text()
        self.assertNotIn("__DATA__", html, "data placeholder was not filled")
        # Nothing may be fetched at view time: the page is opened from an
        # artifact link and from GitHub, and neither serves a sibling JSON.
        self.assertNotIn("fetch(", html)
        self.assertNotIn("XMLHttpRequest", html)

    def test_embedded_data_has_every_field_the_page_reads(self):
        import json
        import pathlib
        import re
        html = pathlib.Path("viz/index.html").read_text()
        m = re.search(r'<script id="viz-data" type="application/json">(.*?)</script>',
                      html, re.S)
        self.assertIsNotNone(m, "embedded data block missing")
        data = json.loads(m.group(1).replace("<\\/", "</"))
        self.assertIn("gate_threshold", data)
        self.assertEqual(len(data["agents"]), 4)
        needed = {"agent", "rank", "gate", "arrive", "admit", "first",
                  "finish", "preempt", "cached", "computed"}
        for cond in data["conditions"].values():
            for pol in cond["policies"].values():
                self.assertTrue(pol["rounds"], "a policy has no rounds")
                for rnd in pol["rounds"]:
                    for turn in rnd["turns"]:
                        self.assertLessEqual(needed, set(turn))

    def test_theme_tokens_are_defined_on_bare_root(self):
        """A colour defined only inside a media block breaks one theme."""
        import pathlib
        import re
        html = pathlib.Path("viz/index.html").read_text()
        root = re.search(r":root \{(.*?)\}", html, re.S).group(1)
        for token in ("--bg", "--ink", "--panel", "--rule", "--accent",
                      "--a0", "--a1", "--a2", "--a3"):
            self.assertIn(token + ":", root, f"{token} missing from bare :root")
