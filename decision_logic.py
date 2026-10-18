def evaluate_settlement(settlement):
    connectivity = settlement["connectivity"]
    exposure = settlement["flood_exposure"]
    people_needing_help = settlement["estimated_people_needing_help"]

    score = 0
    reasons = []
    recommended_actions = []

    # Connectivity assessment
    if connectivity == "POTENTIALLY_CUT_OFF":
        score += 40
        reasons.append("Settlement may be cut off from road access.")
        recommended_actions.append(
            "Verify road access and assess urgent needs."
        )
    elif connectivity == "UNKNOWN":
        score += 15
        reasons.append("Road connectivity information is unknown.")
        recommended_actions.append("Verify settlement accessibility.")
    elif connectivity == "CONNECTED":
        reasons.append("Settlement is currently classified as connected.")

    # Flood exposure assessment
    if exposure == "HIGH":
        score += 30
        reasons.append("High flood exposure is reported.")
    elif exposure == "MEDIUM":
        score += 15
        reasons.append("Medium flood exposure is reported.")
    elif exposure == "UNKNOWN":
        score += 10
        reasons.append("Flood exposure information is unknown.")
        recommended_actions.append("Verify flood conditions.")

    # Estimated assistance needs
    if people_needing_help >= 100:
        score += 25
        reasons.append("Estimated assistance needs are high.")
    elif people_needing_help >= 50:
        score += 15
        reasons.append("Estimated assistance needs are moderate.")
    else:
        score += 5
        reasons.append("Estimated assistance needs are lower.")

    # Priority category
    if score >= 70:
        priority = "HIGH"
    elif score >= 40:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    # Flag uncertain data for verification
    if settlement["data_confidence"] == "LOW":
        recommended_actions.append(
            "Verify data before making a final response decision."
        )
    elif settlement["data_confidence"] == "MEDIUM":
        recommended_actions.append(
            "Cross-check key information with another data source."
        )

    if not recommended_actions:
        recommended_actions.append(
            "Continue monitoring and reassess if conditions change."
        )

    return {
        "settlement_id": settlement["settlement_id"],
        "name": settlement["name"],
        "score": score,
        "priority": priority,
        "recommended_actions": recommended_actions,
        "reasons": reasons,
        "data_confidence": settlement["data_confidence"],
        "data_source": "SIMULATED",
    }


def prioritize_settlements(settlements):
    results = [evaluate_settlement(item) for item in settlements]
    results.sort(key=lambda item: item["score"], reverse=True)
    return results