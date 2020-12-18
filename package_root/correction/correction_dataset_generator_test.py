from correction import correction_dataset_generator

def test_xa(mocker):
    tokenizer = mocker.MagicMock()
    tokenizer.get_vocab().keys = lambda: ['a', 'b', '##a', '##b', 'my_pad']
    tokenizer.pad_token = 'my_pad'
    generator = correction_dataset_generator.CorrectionDatasetGenerator(tokenizer, mocker.MagicMock(), 5)
    assert generator.vocab == ['##a', '##b', 'a', 'b']


def test_x(mocker):
    assert 0 == 0