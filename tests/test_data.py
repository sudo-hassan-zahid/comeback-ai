from comeback_ai.ml.data import FEATURES, generate_demo_data


def test_demo_data_is_reproducible_and_valid():
    first = generate_demo_data(rows=50, seed=7)
    second = generate_demo_data(rows=50, seed=7)
    assert first.equals(second)
    assert set(FEATURES).issubset(first.columns)
    assert set(first["at_risk"].unique()).issubset({0, 1})
