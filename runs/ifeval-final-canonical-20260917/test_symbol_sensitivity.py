"""CPU-only checks for the supplemental literal-symbol analysis."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("symbol_sensitivity", HERE / "score_symbol_sensitivity.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class SymbolSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = [json.loads(line) for line in (HERE / "input_data.jsonl").read_text().splitlines()]

    def fixtures(self):
        data = copy.deepcopy(self.data)
        generated = [{"key": r["key"], "prompt": r["prompt"], "responses": ["No symbols."]} for r in data]
        details = [{"key": r["key"], "sample": 0, "strict_list": [False] * len(r["instruction_id_list"]),
                    "loose_list": [False] * len(r["instruction_id_list"]),
                    "strict_all": False, "loose_all": False} for r in data]
        result = {"recipe": {"n": 1}, "details": details, **{k: 0.0 for k in audit.METRICS}}
        return data, generated, result

    def test_literal_counts_at_threshold_and_below(self):
        for symbol, count in [("#", 4), ("!", 6)]:
            self.assertEqual(audit.literal_flags(symbol * count, symbol, count), (True, True))
            self.assertEqual(audit.literal_flags(symbol * (count - 1), symbol, count), (False, False))
            self.assertEqual(audit.literal_flags("O" * 100 + "h" * 100, symbol, count), (False, False))
            self.assertEqual(audit.literal_flags(" \n ", symbol, count), (False, False))

    def test_exact_eight_loose_candidates(self):
        self.assertEqual(audit.loose_candidates("*first*\n**middle**\n*last*"), [
            "*first*\n**middle**\n*last*", "first\nmiddle\nlast",
            "**middle**\n*last*", "*first*\n**middle**", "**middle**",
            "middle\nlast", "first\nmiddle", "middle"])
        self.assertEqual(len(audit.loose_candidates("single line")), 8)

    def test_strict_implies_loose_with_original_candidate(self):
        self.assertEqual(audit.literal_flags("####\nbody\ntrailer", "#", 4), (True, True))
        self.assertEqual(audit.literal_flags("*!!*\n**!!**\n!!", "!", 6), (True, True))

    def test_candidates_match_frozen_official_loose_function(self):
        official = HERE / "source/third_party/ifeval/instruction_following_eval/evaluation_lib.py"
        node = next(n for n in ast.parse(official.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "test_instruction_following_loose")
        seen = []

        class Recorder:
            def __init__(self, _): pass
            def build_description(self, **_): pass
            def get_instruction_args(self): return {}
            def check_following(self, value):
                seen.append(value)
                return False

        namespace = {"instructions_registry": SimpleNamespace(INSTRUCTION_DICT={"record": Recorder}),
                     "OutputExample": lambda **kwargs: kwargs}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(official), "exec"), namespace)
        response = "*first*\n**middle**\n*last*"
        inp = SimpleNamespace(prompt="p", instruction_id_list=["record"], kwargs=[{}])
        namespace["test_instruction_following_loose"](inp, {"p": response})
        self.assertEqual(seen, audit.loose_candidates(response))

    def test_only_target_constraint_changes_without_mutating_inputs(self):
        data, generated, result = self.fixtures()
        # Reverse target instruction order to prove lookup does not assume an index.
        row = next(r for r in data if r["key"] == 1122)
        row["instruction_id_list"].reverse()
        row["kwargs"].reverse()
        next(r for r in generated if r["key"] == 1122)["responses"] = ["####"]
        before = copy.deepcopy((data, generated, result))
        out = audit.audit_model(data, generated, result)
        self.assertEqual((data, generated, result), before)
        self.assertEqual(out["delta"]["inst_strict"], 1 / 834)
        self.assertEqual(out["delta"]["inst_loose"], 1 / 834)
        self.assertEqual(out["delta"]["prompt_strict"], 0)
        target = next(x for x in out["targets"] if x["key"] == 1122)
        self.assertEqual(target["instruction_index"], row["instruction_id_list"].index("keywords:letter_frequency"))
        self.assertEqual(sum(target["literal_strict_list"]), 1)

    def test_prompt_metrics_require_other_constraints_to_pass(self):
        data, generated, result = self.fixtures()
        row = next(r for r in data if r["key"] == 1129)
        index = row["instruction_id_list"].index("keywords:letter_frequency")
        detail = next(r for r in result["details"] if r["key"] == 1129)
        for mode in ["strict", "loose"]:
            detail[mode + "_list"] = [i != index for i in range(len(row["instruction_id_list"]))]
        baseline = (len(row["instruction_id_list"]) - 1) / 834
        result.update(inst_strict=baseline, inst_loose=baseline)
        next(r for r in generated if r["key"] == 1129)["responses"] = ["!!!!!!"]
        out = audit.audit_model(data, generated, result)
        self.assertAlmostEqual(out["delta"]["prompt_strict"], 1 / 541)
        self.assertAlmostEqual(out["delta"]["prompt_loose"], 1 / 541)

    def test_reject_incomplete_duplicate_and_wrong_prompt(self):
        for damage in ["missing", "duplicate", "prompt", "multisample"]:
            data, generated, result = self.fixtures()
            if damage == "missing": generated.pop()
            elif damage == "duplicate": generated[-1] = generated[0]
            elif damage == "prompt": generated[0]["prompt"] = "wrong prompt"
            else: generated[0]["responses"].append("extra")
            with self.subTest(damage=damage), self.assertRaises(ValueError):
                audit.audit_model(data, generated, result)

    def test_reject_inconsistent_aggregate_flags_or_target_kwargs(self):
        for damage in ["aggregate", "all_flag", "list_type", "kwargs"]:
            data, generated, result = self.fixtures()
            if damage == "aggregate": result["inst_strict"] = 1.0
            elif damage == "all_flag": result["details"][0]["strict_all"] = True
            elif damage == "list_type": result["details"][0]["strict_list"][0] = 1
            else:
                row = next(r for r in data if r["key"] == 1122)
                index = row["instruction_id_list"].index("keywords:letter_frequency")
                row["kwargs"][index]["letter"] = "O"
            with self.subTest(damage=damage), self.assertRaises(ValueError):
                audit.audit_model(data, generated, result)

    def test_cli_all_six_models_preserves_original_files(self):
        data, generated, result = self.fixtures()
        with tempfile.TemporaryDirectory(prefix="ifeval-symbol-test-") as temporary:
            suite = Path(temporary)
            (suite / "input_data.jsonl").write_bytes((HERE / "input_data.jsonl").read_bytes())
            originals = {}
            for arm in ("base", "sft-init", "grpo", "lam2", "lam4", "lam8"):
                directory = suite / "results" / arm
                directory.mkdir(parents=True)
                result["tag"] = arm
                for name, contents in [
                    ("results_t1.0_n1.json", json.dumps(result)),
                    ("generations_t1.0_n1.jsonl", "\n".join(json.dumps(x) for x in generated)),
                ]:
                    path = directory / name
                    path.write_text(contents)
                    originals[path] = path.read_bytes()
            completed = subprocess.run([sys.executable, "-B", str(HERE / "score_symbol_sensitivity.py"),
                                        "--suite", str(suite)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((suite / "symbol_sensitivity.json").read_text())
            self.assertEqual(set(report["models"]), set(audit.ARMS))
            self.assertFalse(report["official_standard_score"])
            self.assertEqual(report["num_prompts"], 541)
            self.assertEqual(report["num_instructions"], 834)
            for path, contents in originals.items():
                self.assertEqual(path.read_bytes(), contents)


if __name__ == "__main__":
    unittest.main()
