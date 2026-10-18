# test_export.py
# Checks that exported results can be read and validated.

import json
import subprocess
import sys
from pathlib import Path


def test_export():
    folder = Path(__file__).parent
    output_file = folder / "ranked_results.json"

    # Generate a fresh export
    subprocess.run(
        [sys.executable, str(folder / "export_results.py")],
        check=True,
    )

    # Read the exported JSON
    with output_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # Validate the results
    assert data["data_source"] == "SIMULATED"
    assert data["settlement_count"] == 6
    assert len(data["results"]) == 6

    scores = [item["score"] for item in data["results"]]
    assert scores == sorted(scores, reverse=True)

    for item in data["results"]:
        assert "priority" in item
        assert "reasons" in item
        assert "recommended_actions" in item
        assert item["data_source"] == "SIMULATED"

    print("\nAll export checks passed!")


if __name__ == "__main__":
    test_export()