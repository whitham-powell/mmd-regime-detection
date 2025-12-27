# tests/test_api.py
"""
Tests for the refactored regime detection API.

Verifies that:
1. load_ohlcv works with different tickers
2. make_features accepts DataFrame as first argument
3. Backwards compatibility still works (with deprecation warnings)
"""

import warnings

import numpy as np
import pandas as pd
import pytest


class TestDataLoading:
    """Tests for data.py"""

    def test_load_ohlcv_default(self):
        """Load SPY with default parameters."""
        from regime_detection import load_ohlcv

        df = load_ohlcv()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert all(
            col in df.columns for col in ["Open", "High", "Low", "Close", "Volume"]
        )

    def test_load_ohlcv_custom_ticker(self):
        """Load a different ticker."""
        from regime_detection import load_ohlcv

        df = load_ohlcv("AAPL", period="1mo")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_ohlcv_date_range(self):
        """Load with specific date range."""
        from regime_detection import load_ohlcv

        df = load_ohlcv("MSFT", start="2023-01-01", end="2023-06-01")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert df.index[0].year == 2023

    def test_load_ohlcv_invalid_ticker(self):
        """Invalid ticker should raise ValueError."""
        from regime_detection import load_ohlcv

        with pytest.raises(ValueError, match="No data returned"):
            load_ohlcv("INVALIDTICKER123XYZ", period="1mo")


class TestFeatureGeneration:
    """Tests for features.py"""

    @pytest.fixture
    def sample_df(self):
        """Create a small sample DataFrame for testing."""
        from regime_detection import load_ohlcv

        return load_ohlcv("SPY", period="3mo")

    def test_make_features_new_api(self, sample_df):
        """Test new API: make_features(df, groups)"""
        from regime_detection import make_features

        features = make_features(sample_df, "base")
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        assert "log_close" in features.columns

    def test_make_features_multiple_groups(self, sample_df):
        """Test combining multiple feature groups."""
        from regime_detection import make_features

        features = make_features(sample_df, ["base", "intraday_shape"])
        assert "log_close" in features.columns
        assert "body" in features.columns
        assert "CLV" in features.columns

    def test_make_features_all(self, sample_df):
        """Test 'all' feature group."""
        from regime_detection import make_features

        features = make_features(sample_df, "all")
        assert len(features.columns) >= 30  # Should have many features

    def test_make_features_invalid_group(self, sample_df):
        """Invalid group name should raise ValueError."""
        from regime_detection import make_features

        with pytest.raises(ValueError, match="Unknown feature group"):
            make_features(sample_df, "nonexistent_group")

    def test_feature_groups_list(self):
        """FEATURE_GROUPS should be a list of available groups."""
        from regime_detection import FEATURE_GROUPS

        assert isinstance(FEATURE_GROUPS, list)
        assert "base" in FEATURE_GROUPS
        assert "all" in FEATURE_GROUPS


class TestBackwardsCompatibility:
    """Tests for backwards compatibility with deprecation warnings."""

    def test_legacy_df_import_warns(self):
        """Importing df should emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from regime_detection.features import df

            # Check that a deprecation warning was issued
            assert any(
                issubclass(warning.category, DeprecationWarning) for warning in w
            )
            # But it should still work
            assert isinstance(df, pd.DataFrame)


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline(self):
        """Test complete analysis pipeline with new API."""
        from kta import rbf
        from sklearn.preprocessing import StandardScaler

        from regime_detection import (
            find_regime_boundaries,
            load_ohlcv,
            make_features,
            results_to_dataframe,
            sliding_window_mmd,
        )

        # 1. Load data
        df = load_ohlcv("SPY", period="6mo")

        # 2. Generate features
        features = make_features(df, "base")

        # 3. Standardize
        scaler = StandardScaler()
        signal = scaler.fit_transform(features.values)

        # 4. Compute kernel bandwidth
        sigma = np.median(np.abs(signal - np.median(signal)))
        gamma = 1.0 / (2 * sigma**2)

        # 5. Run MMD (with reduced params for speed)
        results = sliding_window_mmd(
            data=signal,
            kernel_fn=rbf,
            kernel_params={"gamma": gamma},
            window=20,
            step=10,
            n_permutations=50,
        )

        # 6. Convert to DataFrame
        results_df = results_to_dataframe(results, features.index)
        assert isinstance(results_df, pd.DataFrame)
        assert "mmd" in results_df.columns
        assert "std_from_null" in results_df.columns

        # 7. Find boundaries
        boundaries = find_regime_boundaries(results_df, threshold=5.0)
        assert isinstance(boundaries, pd.DatetimeIndex)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
