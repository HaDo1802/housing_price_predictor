import pandas as pd
import pytest
def test_no_data_leakage():
    """Ensure train/val/test have no overlap"""
    train = pd.read_csv('data/processed/train.csv')
    val = pd.read_csv('data/processed/val.csv')
    test = pd.read_csv('data/processed/test.csv')
    assert len(set(train.index) & set(val.index)) == 0
    assert len(set(train.index) & set(test.index)) == 0
    assert len(set(val.index) & set(test.index)) == 0