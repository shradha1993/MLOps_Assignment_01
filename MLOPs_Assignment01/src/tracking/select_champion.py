def select_champion(results):
    champion_name = max(
        results,
        key=lambda name: results[name]["f1"],
    )

    champion_metrics = results[champion_name]

    return {
        "model_name": champion_name,
        "metrics": champion_metrics,
    }
