# export_results.py
# Exports simulated decision-engine results to JSON.

import json
from datetime import datetime, timezone
from pathlib import Path

from mock_data import settlements
from decision_logic import prioritize_settlements


def export_results():
    results = prioritize_settlements(settlements)

    output = {
        "project": "Flood Response Decision Engine",
        "data_source": "SIMULATED",
        "warning": (
            "Fictional demo data only. "
            "Not for real-world emergency decisions."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "settlement_count": len(results),
        "results": results,
    }

    output_path = Path(__file__).parent / "ranked_results.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print(f"Exported {len(results)} settlements.")
    print(f"Output file: {output_path}")
    print("Data source: SIMULATED")


if __name__ == "__main__":
    export_results()