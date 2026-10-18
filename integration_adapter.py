# integration_adapter.py
# Interface between graph-derived settlement data
# and the flood response decision engine.

from decision_logic import prioritize_settlements


REQUIRED_FIELDS = {
    "settlement_id",
    "name",
    "connectivity",
    "estimated_people_needing_help",
    "flood_exposure",
    "data_confidence",
}


def rank_settlements(settlement_records):
    """
    Accept settlement records from a graph workflow
    and return ranked decision-engine results.

    Input records must contain REQUIRED_FIELDS.
    """

    if not isinstance(settlement_records, list):
        raise TypeError("Settlement records must be provided as a list.")

    if not settlement_records:
        raise ValueError("No settlement records were provided.")

    for index, record in enumerate(settlement_records):
        if not isinstance(record, dict):
            raise TypeError(
                f"Settlement at position {index} must be a dictionary."
            )

        missing_fields = REQUIRED_FIELDS - record.keys()

        if missing_fields:
            raise ValueError(
                f"Settlement at position {index} is missing fields: "
                f"{', '.join(sorted(missing_fields))}"
            )

    return prioritize_settlements(settlement_records)


if __name__ == "__main__":
    print("Decision engine integration adapter is ready.")
    print("Expected input: a list of settlement records.")
    print("Required fields:", ", ".join(sorted(REQUIRED_FIELDS)))
    print("Data source: determined by the calling workflow.")
    print("Current prototype data remains simulated.")