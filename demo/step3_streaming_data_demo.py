"""Step 3 demonstration: streaming Titanic CSV through NumCompute utilities.

Run from repository root:
    python demo/step3_streaming_data_demo.py
"""

from pathlib import Path

import numpy as np

from numcompute_stream.io import iter_supervised_csv
from numcompute_stream.preprocessing import Imputer, StandardScaler
from numcompute_stream.stats import StreamingStats


CSV_PATH = Path(__file__).parent / "data" / "titanic.csv"
FEATURES = ["pclass", "age", "sibsp", "parch", "fare"]


def main() -> None:
    imputer = Imputer(strategy="mean")
    scaler = StandardScaler()
    stats = StreamingStats(window_size=250)

    for chunk_no, (X, y) in enumerate(
        iter_supervised_csv(CSV_PATH, target="survived", features=FEATURES, chunksize=150),
        start=1,
    ):
        missing_before = int(np.isnan(X).sum())
        stats.update_stats(X)
        imputer.partial_fit(X)
        X_filled = imputer.transform(X)
        scaler.partial_fit(X_filled)
        X_scaled = scaler.transform(X_filled)

        print(
            f"chunk={chunk_no:02} rows={len(y):3} "
            f"missing_before={missing_before:3} "
            f"scaled_shape={X_scaled.shape} "
            f"cum_age_mean={stats.mean()[1]:.2f} "
            f"rolling_age_median={stats.quantile(0.5)[1]:.2f}"
        )
        assert np.isfinite(X_scaled).all()

    print("All chunks processed; step 3 streaming preprocessing is operational.")


if __name__ == "__main__":
    main()
