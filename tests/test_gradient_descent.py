import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path to import gradient_descent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradient_descent import prediction, compute_cost


class TestPrediction:
    """Test cases for the prediction function."""
    
    def test_prediction_simple_case(self):
        """Test prediction with simple 1D case."""
        X = np.array([[1], [2], [3]])
        w = np.array([2])
        b = 1
        
        expected = np.array([3, 5, 7])  # [1*2+1, 2*2+1, 3*2+1]
        result = prediction(X, w, b)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_prediction_2d_features(self):
        """Test prediction with 2D features."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        w = np.array([0.5, 1.5])
        b = 2
        
        # Manual calculation: 
        # [1*0.5 + 2*1.5 + 2, 3*0.5 + 4*1.5 + 2, 5*0.5 + 6*1.5 + 2]
        # [0.5 + 3 + 2, 1.5 + 6 + 2, 2.5 + 9 + 2]
        # [5.5, 9.5, 13.5]
        expected = np.array([5.5, 9.5, 13.5])
        result = prediction(X, w, b)
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_prediction_zero_weights(self):
        """Test prediction with zero weights."""
        X = np.array([[1, 2], [3, 4]])
        w = np.array([0, 0])
        b = 5
        
        expected = np.array([5, 5])  # Only bias term
        result = prediction(X, w, b)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_prediction_zero_bias(self):
        """Test prediction with zero bias."""
        X = np.array([[2], [4]])
        w = np.array([3])
        b = 0
        
        expected = np.array([6, 12])  # [2*3, 4*3]
        result = prediction(X, w, b)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_prediction_negative_values(self):
        """Test prediction with negative weights and bias."""
        X = np.array([[1, -1], [2, -2]])
        w = np.array([-1, 2])
        b = -3
        
        # Manual calculation:
        # [1*(-1) + (-1)*2 + (-3), 2*(-1) + (-2)*2 + (-3)]
        # [-1 - 2 - 3, -2 - 4 - 3]
        # [-6, -9]
        expected = np.array([-6, -9])
        result = prediction(X, w, b)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_prediction_single_sample(self):
        """Test prediction with a single sample."""
        X = np.array([[1.5, 2.5, 3.5]])
        w = np.array([1, 2, 3])
        b = 0.5
        
        # 1.5*1 + 2.5*2 + 3.5*3 + 0.5 = 1.5 + 5 + 10.5 + 0.5 = 17.5
        expected = np.array([17.5])
        result = prediction(X, w, b)
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_prediction_large_dataset(self):
        """Test prediction with larger dataset."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        w = np.random.randn(5)
        b = np.random.randn()
        
        result = prediction(X, w, b)
        
        # Check that result has correct shape
        assert result.shape == (100,)
        
        # Verify calculation manually for first sample
        expected_first = np.dot(X[0], w) + b
        np.testing.assert_almost_equal(result[0], expected_first)
    
    def test_prediction_type_consistency(self):
        """Test that prediction returns numpy array."""
        X = np.array([[1, 2]])
        w = np.array([1, 1])
        b = 1
        
        result = prediction(X, w, b)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype in [np.float64, np.int64, np.float32, np.int32]
    
    def test_prediction_dimension_mismatch_error(self):
        """Test that prediction raises error for dimension mismatch."""
        X = np.array([[1, 2, 3]])  # 3 features
        w = np.array([1, 2])       # 2 weights
        b = 1
        
        with pytest.raises(ValueError):
            prediction(X, w, b)
    
    def test_prediction_empty_input(self):
        """Test prediction with empty input."""
        X = np.array([]).reshape(0, 2)
        w = np.array([1, 2])
        b = 1
        
        result = prediction(X, w, b)
        
        assert result.shape == (0,)
        assert isinstance(result, np.ndarray)


class TestComputeCost:
    """Test cases for the compute_cost function."""

    def test_compute_cost_perfect_predictions(self):
        """Test cost when predictions exactly match targets."""
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])
        w = np.array([2])
        b = 0

        # Perfect predictions: [1*2+0, 2*2+0, 3*2+0] = [2, 4, 6]
        cost = compute_cost(X, y, w, b)

        np.testing.assert_almost_equal(cost, 0.0)

    def test_compute_cost_simple_case(self):
        """Test cost calculation with known values."""
        X = np.array([[1], [2]])
        y = np.array([1, 2])
        w = np.array([1])
        b = 1

        # Predictions: [1*1+1, 2*1+1] = [2, 3]
        # Errors: [2-1, 3-2] = [1, 1]
        # Cost: (1/(2*2)) * (1^2 + 1^2) = 0.25 * 2 = 0.5
        expected_cost = 0.5
        cost = compute_cost(X, y, w, b)

        np.testing.assert_almost_equal(cost, expected_cost)

    def test_compute_cost_2d_features(self):
        """Test cost calculation with 2D features."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([5, 10])
        w = np.array([1, 1])
        b = 1

        # Predictions: [1*1+2*1+1, 3*1+4*1+1] = [4, 8]
        # Errors: [4-5, 8-10] = [-1, -2]
        # Cost: (1/(2*2)) * ((-1)^2 + (-2)^2) = 0.25 * (1 + 4) = 1.25
        expected_cost = 1.25
        cost = compute_cost(X, y, w, b)

        np.testing.assert_almost_equal(cost, expected_cost)

    def test_compute_cost_single_sample(self):
        """Test cost calculation with single sample."""
        X = np.array([[2, 3]])
        y = np.array([10])
        w = np.array([1, 2])
        b = 1

        # Prediction: [2*1+3*2+1] = [9]
        # Error: [9-10] = [-1]
        # Cost: (1/(2*1)) * ((-1)^2) = 0.5 * 1 = 0.5
        expected_cost = 0.5
        cost = compute_cost(X, y, w, b)

        np.testing.assert_almost_equal(cost, expected_cost)

    def test_compute_cost_large_errors(self):
        """Test cost calculation with large prediction errors."""
        X = np.array([[1], [2], [3]])
        y = np.array([1, 2, 3])
        w = np.array([10])  # Large weight causing large predictions
        b = 0

        # Predictions: [10, 20, 30]
        # Errors: [10-1, 20-2, 30-3] = [9, 18, 27]
        # Cost: (1/(2*3)) * (9^2 + 18^2 + 27^2) = (1/6) * (81 + 324 + 729) = 1134/6 = 189
        expected_cost = 189.0
        cost = compute_cost(X, y, w, b)

        np.testing.assert_almost_equal(cost, expected_cost)

    def test_compute_cost_zero_samples(self):
        """Test cost calculation with zero samples."""
        X = np.array([]).reshape(0, 2)
        y = np.array([])
        w = np.array([1, 2])
        b = 1

        # With zero samples, cost should be 0 (handled by the function)
        cost = compute_cost(X, y, w, b)

        # Cost should be exactly 0.0 for empty dataset
        np.testing.assert_equal(cost, 0.0)

    def test_compute_cost_negative_targets(self):
        """Test cost calculation with negative target values."""
        X = np.array([[1], [2]])
        y = np.array([-1, -2])
        w = np.array([1])
        b = 0

        # Predictions: [1, 2]
        # Errors: [1-(-1), 2-(-2)] = [2, 4]
        # Cost: (1/(2*2)) * (2^2 + 4^2) = 0.25 * (4 + 16) = 5.0
        expected_cost = 5.0
        cost = compute_cost(X, y, w, b)

        np.testing.assert_almost_equal(cost, expected_cost)

    def test_compute_cost_always_positive(self):
        """Test that cost is always non-negative."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        w = np.random.randn(3)
        b = np.random.randn()

        cost = compute_cost(X, y, w, b)

        assert cost >= 0, "Cost should always be non-negative"

    def test_compute_cost_return_type(self):
        """Test that compute_cost returns a scalar."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2])
        w = np.array([0.5, 0.5])
        b = 0.5

        cost = compute_cost(X, y, w, b)

        assert np.isscalar(cost), "Cost should be a scalar value"
        assert isinstance(cost, (float, np.floating)), "Cost should be a float"

    def test_compute_cost_dimension_mismatch(self):
        """Test error handling for dimension mismatches."""
        X = np.array([[1, 2]])
        y = np.array([1, 2])  # Wrong length
        w = np.array([1, 1])
        b = 1

        with pytest.raises((ValueError, IndexError)):
            compute_cost(X, y, w, b)

    def test_compute_cost_uses_prediction_function(self):
        """Test that compute_cost uses the prediction function correctly."""
        X = np.array([[1, 2]])
        y = np.array([5])
        w = np.array([1, 1])
        b = 1

        # Manually compute what the cost should be
        pred = prediction(X, w, b)  # Should be [4]
        expected_cost = (1 / (2 * 1)) * np.sum((pred - y) ** 2)

        cost = compute_cost(X, y, w, b)

        np.testing.assert_almost_equal(cost, expected_cost)


if __name__ == "__main__":
    pytest.main([__file__])
