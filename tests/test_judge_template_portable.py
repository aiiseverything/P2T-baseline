"""Judge-template selection is portable and never requires an API call."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import judge_alpaca as judge


def test_explicit_template_overrides_legacy_path(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy.txt"
    legacy.write_text("legacy")
    explicit = tmp_path / "explicit.txt"
    explicit.write_text("explicit {instruction} {output_1} {output_2}\n", encoding="utf-8")
    monkeypatch.setattr(judge, "TEMPLATE_PATH", legacy)
    assert judge.load_template(explicit) == explicit.read_text()


def test_explicit_missing_template_fails_without_falling_back(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy.txt"
    legacy.write_text("legacy")
    monkeypatch.setattr(judge, "TEMPLATE_PATH", legacy)
    with pytest.raises(FileNotFoundError, match="missing.txt"):
        judge.load_template(tmp_path / "missing.txt")


def test_legacy_path_is_preferred_without_importing_optional_package(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy.txt"
    legacy.write_text("old template\n")
    monkeypatch.setattr(judge, "TEMPLATE_PATH", legacy)
    monkeypatch.setattr(importlib.util, "find_spec", lambda *a: pytest.fail("legacy must take priority"))
    assert judge.load_template() == "old template\n"


def test_installed_package_template_is_found_without_importing_it(tmp_path, monkeypatch):
    package = tmp_path / "alpaca_eval"
    package.mkdir()
    # Locating data should not import optional dependencies or package side effects.
    (package / "__init__.py").write_text("raise RuntimeError('package must not be imported')\n")
    template = package / "evaluators_configs/alpaca_eval_clf_gpt4_turbo/alpaca_eval_clf.txt"
    template.parent.mkdir(parents=True)
    template.write_text("installed {instruction}\n")
    monkeypatch.delitem(sys.modules, "alpaca_eval", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(judge, "TEMPLATE_PATH", tmp_path / "missing-legacy.txt")
    assert judge.load_template() == "installed {instruction}\n"


@pytest.mark.parametrize("spec", [None, SimpleNamespace(submodule_search_locations=[]),
                                  SimpleNamespace(submodule_search_locations=["/missing/alpaca_eval"])])
def test_missing_template_has_actionable_error(tmp_path, monkeypatch, spec):
    monkeypatch.setattr(judge, "TEMPLATE_PATH", tmp_path / "missing-legacy.txt")
    monkeypatch.setattr(importlib.util, "find_spec", lambda *a: spec)
    with pytest.raises(FileNotFoundError, match="--template"):
        judge.load_template()


def test_cli_passes_explicit_template_as_path_before_any_network_call(tmp_path, monkeypatch):
    explicit = tmp_path / "template.txt"

    class Captured(Exception):
        pass

    def capture(path):
        assert isinstance(path, Path)
        assert path == explicit
        raise Captured()

    monkeypatch.setattr(judge, "load_template", capture)
    monkeypatch.setattr(judge, "Relay", lambda *a: pytest.fail("must not contact judge API"))
    monkeypatch.setattr(sys, "argv", ["judge", "--template", str(explicit)])
    with pytest.raises(Captured):
        judge.main()


def test_real_help_renders_portable_template_option(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["judge", "--help"])
    with pytest.raises(SystemExit) as error:
        judge.main()
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--template TEMPLATE" in help_text
    assert "50%" in help_text
