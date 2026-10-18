from mock_data import settlements
from decision_logic import prioritize_settlements


def main():
    results = prioritize_settlements(settlements)

    print("=" * 55)
    print("FLOOD RESPONSE DECISION ENGINE")
    print("DEMO MODE — USING SIMULATED DATA ONLY")
    print("=" * 55)

    for rank, item in enumerate(results, start=1):
        print(f"\n{rank}. {item['name']} ({item['settlement_id']})")
        print(f"   Priority: {item['priority']}")
        print(f"   Score: {item['score']}")
        print(f"   Data confidence: {item['data_confidence']}")

        print("   Reasons:")
        for reason in item["reasons"]:
            print(f"   - {reason}")

        print("   Recommended actions:")
        for action in item["recommended_actions"]:
            print(f"   - {action}")

        print(f"   Data source: {item['data_source']}")
        print("-" * 55)


if __name__ == "__main__":
    main()