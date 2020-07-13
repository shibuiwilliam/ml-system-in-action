import pytest
from app.ml.load_model import get_model_files, get_model_file


@pytest.mark.parametrize(
    ('name', 'fallback_name', 'model_directory', 'expected'),
    [('a.pkl', 'b.pkl', '/tmp/', ['a.pkl', 'b.pkl']),
     ('a.pkl', 'b.pkl', '/tmp/', ['c.pkl', 'd.pkl'])]
)
def test_get_model_files(
        mocker,
        name,
        fallback_name,
        model_directory,
        expected):
    mocker.patch('os.listdir', return_value=[expected[0], expected[1]])
    files = get_model_files(model_directory)
    assert files == expected


@pytest.mark.parametrize(
    ('name', 'model_directory', 'files', 'expected'),
    [('a.pkl', '/tmp/', ['a.pkl', 'b.pkl'], '/tmp/a.pkl'),
     ('a.pkl', '/tmp/', ['b.pkl'], '/tmp/a.pkl')]
)
def test_get_model_file(
        mocker,
        name,
        model_directory,
        files,
        expected):
    mocker.patch('app.ml.load_model.get_model_files', return_value=files)
    found_model_filepath = get_model_file(name, model_directory)
    assert found_model_filepath == expected
