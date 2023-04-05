import pytest as pt
import pandas as pd

import deep_hybrid_recommender.pipelines.data_processing.nodes as dtp

def test_prepare_categorical_encoders():
    # Given
    data = pd.DataFrame({
      'categorical_col': [{'a': 'A1', 'b': 'B1'}, {'a': 'A2', 'b': 'B2'}],
        'other_col': ['a', 'b'],
    })

    # When
    encoders = dtp.prepare_categorical_encoders(data, 'categorical_col')

    # Then
    assert encoders.keys() == {'a', 'b'}
    assert (encoders['a'].classes_ == ['A1', 'A2', 'missing']).all()
    assert (encoders['b'].classes_ == ['B1', 'B2', 'missing']).all()


def test_encode_categorical_data():
    # Given
    data = pd.DataFrame({
        'categorical_col': [{'a': 'A1', 'b': 'B1'}, {'a': 'A2', 'b': 'B2'}, {'a': 'A1', 'b': 'B3'}],
        'other_col': ['a', 'b', 'c'],
    })
    encoders = dtp.prepare_categorical_encoders(data, 'categorical_col')
    expected_data = pd.DataFrame({
        'categorical_col': [{'a': 'A1', 'b': 'B1'}, {'a': 'A2', 'b': 'B2'}, {'a': 'A1', 'b': 'B3'}],
        'other_col': ['a', 'b', 'c'],
        'a': [0, 1, 0],
        'b': [0, 1, 2],
    })

    # When
    encoded_data = dtp.encode_categorical_data(data, encoders, 'categorical_col')

    # Then
    pd.testing.assert_frame_equal(encoded_data.sort_values(by='other_col'), expected_data.sort_values(by='other_col'))