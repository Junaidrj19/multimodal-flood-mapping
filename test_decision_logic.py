import unittest

from decision_logic import evaluate_settlement


class TestDecisionLogic(unittest.TestCase):

    def test_cut_off_high_exposure(self):
        settlement = {
            "settlement_id": "T001",
            "name": "Test Village",
            "connectivity": "POTENTIALLY_CUT_OFF",
            "estimated_people_needing_help": 150,
            "flood_exposure": "HIGH",
            "data_confidence": "MEDIUM",
        }

        result = evaluate_settlement(settlement)

        self.assertEqual(result["priority"], "HIGH")
        self.assertEqual(result["score"], 95)

    def test_connected_medium_exposure(self):
        settlement = {
            "settlement_id": "T002",
            "name": "Connected Village",
            "connectivity": "CONNECTED",
            "estimated_people_needing_help": 80,
            "flood_exposure": "MEDIUM",
            "data_confidence": "HIGH",
        }

        result = evaluate_settlement(settlement)

        self.assertEqual(result["priority"], "LOW")
        self.assertEqual(result["score"], 30)

    def test_unknown_information_requires_verification(self):
        settlement = {
            "settlement_id": "T003",
            "name": "Unknown Village",
            "connectivity": "UNKNOWN",
            "estimated_people_needing_help": 40,
            "flood_exposure": "UNKNOWN",
            "data_confidence": "LOW",
        }

        result = evaluate_settlement(settlement)

        self.assertIn(
            "Verify settlement accessibility.",
            result["recommended_actions"],
        )
        self.assertIn(
            "Verify flood conditions.",
            result["recommended_actions"],
        )

    def test_low_confidence_triggers_verification(self):
        settlement = {
            "settlement_id": "T004",
            "name": "Uncertain Village",
            "connectivity": "CONNECTED",
            "estimated_people_needing_help": 20,
            "flood_exposure": "MEDIUM",
            "data_confidence": "LOW",
        }

        result = evaluate_settlement(settlement)

        self.assertIn(
            "Verify data before making a final response decision.",
            result["recommended_actions"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)