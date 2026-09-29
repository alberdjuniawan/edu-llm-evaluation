import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_sibi.py"
    spec = importlib.util.spec_from_file_location("check_sibi", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


class FakeGatedError(Exception):
    pass


class FakeNotFoundError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.response = SimpleNamespace(status_code=404)


class FakeApi:
    def __init__(self, behavior):
        self.behavior = behavior

    def dataset_info(self, dataset_id):
        outcome = self.behavior

        if isinstance(outcome, Exception):
            raise outcome

        return SimpleNamespace(siblings=[1, 2, 3])


def test_reachable_dataset_reports_file_count():
    module = _load()

    assert module.check_one(FakeApi(None), "org/ds") == ("ok", "3 files")


def test_gated_dataset_classified():
    module = _load()

    status, _ = module.check_one(FakeApi(FakeGatedError("gated repo")), "org/ds")

    assert status == "gated"


def test_missing_dataset_classified():
    module = _load()

    status, _ = module.check_one(FakeApi(FakeNotFoundError("not found")), "org/ds")

    assert status == "missing"


def test_unknown_error_classified():
    module = _load()

    status, detail = module.check_one(FakeApi(ValueError("boom")), "org/ds")

    assert status == "error"
    assert "boom" in detail
