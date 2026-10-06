TIME_LABELS = {
    0: "+15 minutes",
    1: "+30 minutes",
    2: "+45 minutes",
    3: "+60 minutes"
}


def find_best_option(predictions):

    best_platform = None
    best_index = None
    best_fare = float("inf")

    for platform, values in predictions.items():

        for index, fare in enumerate(values):

            if fare < best_fare:

                best_fare = fare
                best_platform = platform
                best_index = index

    return (
        best_platform,
        best_index,
        best_fare
    )


def find_best_current_platform(
    predictions
):

    best_platform = min(
        predictions,
        key=lambda platform:
        predictions[platform][0]
    )

    return (
        best_platform,
        predictions[best_platform][0]
    )


def generate_recommendation(
    predictions
):

    (
        best_platform,
        best_index,
        best_fare
    ) = find_best_option(
        predictions
    )

    (
        current_platform,
        current_fare
    ) = find_best_current_platform(
        predictions
    )

    best_time = TIME_LABELS[
        best_index
    ]

    if best_index == 0:

        reason = (
            f"{best_platform} has the lowest "
            f"predicted fare right now."
        )

    else:

        reason = (
            f"Waiting {best_time.replace('+', '')} "
            f"is predicted to give a lower fare "
            f"on {best_platform}."
        )

    return {

        "recommended_platform":
            best_platform,

        "recommended_time":
            best_time,

        "expected_fare":
            best_fare,

        "current_best_platform":
            current_platform,

        "current_best_fare":
            current_fare,

        "reason":
            reason
    }