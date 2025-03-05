"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""
# Imports
import pytest
from regression import LogisticRegressor, loadDataset
import numpy as np

def test_prediction():
    # Initialize model with different parameters
    log_model = LogisticRegressor(num_feats=2)
    
    # Set fixed weights for reproducibility 
    log_model.W = np.array([1, -1, 0.1])  # for the 2 features + 1 bias term
    
    # Create test input data
    X = np.array([[0, 1],
                  [-1, 0],
                  [0, -1]])
    
    # Add bias term at the end (always 1)
    X_with_bias = np.hstack([X, np.ones((X.shape[0], 1))])
    

    dot_XW = np.dot(X_with_bias, log_model.W)
    expected_pred = 1 / (1 + np.exp(-dot_XW))
    actual_pred = log_model.make_prediction(X_with_bias)
    
    assert np.all(np.abs(actual_pred - expected_pred) < 1e-6), "Predictions should match expected values"
    assert actual_pred.shape == (3,), "Prediction shape should match number of samples"
    assert np.all((actual_pred >= 0) & (actual_pred <= 1)), "Predictions should be between 0 and 1"

def test_loss_function():
    # Initialize model
    log_model = LogisticRegressor(num_feats=2)
    
    y_true = np.array([1, 0, 1])
    y_pred = np.array([0.9, 0.1, 0.8])
    
    actual_loss = log_model.loss_function(y_true, y_pred)
    
    # Calculate expected loss using the formula: -1/N * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    expected_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    assert abs(actual_loss - expected_loss) < 0.00001

    # Test that loss is non-negative
    assert actual_loss >= 0, "Loss should be non-negative"

def test_gradient():
    # Initialize model
    log_model = LogisticRegressor(num_feats=2)
    
    # Create test input data
    X = np.array([[0, 0],
                  [1, 1],
                  [2, 2]])
    
    # Add bias term at the end (always 1)
    X_with_bias = np.hstack([X, np.ones((X.shape[0], 1))])
    log_model.W = np.array([1, -1, 0.2])  # for 2 features + 1 bias term

    # True label & test data
    y_true = np.array([0, 1, 0])
    y_pred = log_model.make_prediction(X_with_bias)

    # Calculate gradients
    actual_grad = log_model.calculate_gradient(y_true, X_with_bias)
    expected_grad = np.dot(X_with_bias.T, y_pred - y_true) / len(y_true)

    assert np.all(np.abs(actual_grad - expected_grad) < 1e-6), "Gradients should match expected values"    

def test_training():
    log_model = LogisticRegressor(num_feats=2)
    
    X = np.array([[0, 0],
                  [1, 1],
                  [2, 2]])

    y = np.array([0, 1, 0])

    initial_W = log_model.W.copy()  # copy the initial weights before training
    
    log_model.train_model(X, y, X, y)

    final_W = log_model.W.copy()

    assert np.sum(abs(final_W - initial_W)) != 0, "Weights should change after training"