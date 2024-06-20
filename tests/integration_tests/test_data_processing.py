import pytest
from app.data_processor import DataProcessor
from app.config import DEFAULT_VALUES

def test_process_data():
    data_processor = DataProcessor()
    input_data = [1, 2, 3, 4, 5]
    expected_output = [2, 3, 4, 5, 6]  # Assuming the processor adds 1 to each element
    assert data_processor.process(input_data) == expected_output

def test_load_and_process_data():
    data_processor = DataProcessor()
    with patch('builtins.open', mock_open(read_data="1\n2\n3\n4\n5")):
        data = data_processor.load_data(DEFAULT_VALUES['csv_input_path'])
    processed_data = data_processor.process(data)
    expected_output = [2, 3, 4, 5, 6]  # Assuming the processor adds 1 to each element
    assert processed_data == expected_output
