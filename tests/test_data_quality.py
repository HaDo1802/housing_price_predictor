import pandas as pd
import pytest
def test_no_data_leakage():
    """Ensure train/val/test have no overlap"""
    train = pd.read_csv('data/processed/train.csv')
    val = pd.read_csv('data/processed/val.csv')
    test = pd.read_csv('data/processed/test.csv')
    assert len(set(train.Order.values) & set(val.Order.values)) == 0
    assert len(set(train.Order.values) & set(test.Order.values)) == 0
    assert len(set(val.Order.values) & set(test.Order.values)) == 0