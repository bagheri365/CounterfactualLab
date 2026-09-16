"""Generate the M1 randomized synthetic dataset."""

from pathlib import Path

import yaml

from counterfactuallab.data.synthetic import SyntheticConfig, generate_randomized_data


def main() -> None:
    config_path = Path("configs/randomized.yaml")
    with config_path.open() as stream:
        raw = yaml.safe_load(stream)

    config = SyntheticConfig(
        n_samples=raw["n_samples"],
        n_features=raw["n_features"],
        treatment_probability=raw["treatment_probability"],
        seed=raw["seed"],
    )
    data = generate_randomized_data(config)

    output = Path("data/processed/randomized.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)

    print(f"Wrote {len(data):,} rows to {output}")
    print(f"True ATE: {data['tau_true'].mean():.4f}")
    observed = (
        data.loc[data.treatment == 1, "outcome"].mean()
        - data.loc[data.treatment == 0, "outcome"].mean()
    )
    print(f"Observed difference in means: {observed:.4f}")


if __name__ == "__main__":
    main()
