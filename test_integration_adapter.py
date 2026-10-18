import unittest

from integration_adapter import rank_settlements


class TestIntegrationAdapter(unittest.TestCase):

    def setUp(self):
        self.valid_settlement = {
            "settlement_id": "T001",
            "name": "Test Village",
            "connectivity": "POTENTIALLY_CUT_OFF",
            "estimated_people_needing_help": 120,
            "flood_exposure": "HIGH",
            "data_confidence": "MEDIUM",
        }

    def test_valid_record_is_ranked(self):
        results = rank_settlements([self.valid_settlement])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["settlement_id"], "T001")
        self.assertEqual(results[0]["priority"], "HIGH")

    def test_missing_field_is_rejected(self):
        incomplete = self.valid_settlement.copy()
        del incomplete["connectivity"]

        with self.assertRaisesRegex(ValueError, "connectivity"):
            rank_settlements([incomplete])

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            rank_settlements([])

    def test_non_list_input_is_rejected(self):
        with self.assertRaises(TypeError):
            rank_settlements(self.valid_settlement)


if __name__ == "__main__":
    unittest.main(verbosity=2)