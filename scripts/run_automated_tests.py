from __future__ import annotations

import csv
import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records: list[dict[str, str]] = []

    @staticmethod
    def _name(test: unittest.case.TestCase) -> str:
        return test.id()

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append({"test": self._name(test), "status": "PASS", "detail": ""})

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.records.append({"test": self._name(test), "status": "SKIP", "detail": reason})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.records.append({"test": self._name(test), "status": "FAIL", "detail": self._exc_info_to_string(err, test)})

    def addError(self, test, err):
        super().addError(test, err)
        self.records.append({"test": self._name(test), "status": "ERROR", "detail": self._exc_info_to_string(err, test)})


class RecordingRunner(unittest.TextTestRunner):
    resultclass = RecordingResult


def main() -> int:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py", top_level_dir=str(ROOT))
    result: RecordingResult = RecordingRunner(verbosity=2).run(suite)
    out_dir = ROOT / "test_reports" / "latest"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tests_run": result.testsRun,
        "passed": sum(r["status"] == "PASS" for r in result.records),
        "failed": sum(r["status"] in {"FAIL", "ERROR"} for r in result.records),
        "skipped_v6_acceptance": sum(r["status"] == "SKIP" for r in result.records),
        "successful": result.wasSuccessful(),
        "records": result.records,
    }
    (out_dir / "test_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "test_report.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["test", "status", "detail"])
        writer.writeheader()
        writer.writerows(result.records)
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, ensure_ascii=False, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

