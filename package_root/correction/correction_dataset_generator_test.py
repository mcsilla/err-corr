from correction import correction_dataset_generator
from transformers import BertTokenizerFast

tokenizer_hu = BertTokenizerFast("data/tokenizer/hu/alphabet", do_lower_case=False)
make_old = correction_dataset_generator.MakeTextOld(tokenizer_hu)

# def test_xa(mocker):
#     tokenizer = mocker.MagicMock()
#     tokenizer.get_vocab().keys = lambda: ['a', 'b', '##a', '##b', 'my_pad']
#     tokenizer.pad_token = 'my_pad'
#     generator = correction_dataset_generator.CorrectionDatasetGenerator(tokenizer, mocker.MagicMock(), 5)
#     assert generator.vocab == ['##a', '##b', 'a', 'b']


test_str = [
    "Össze",
    "csipet és a",
    "Szerecsenként",
    "jövőbeli",
]

test_old_tokens = [
    ["Ö", "##f", "##z", "##f", "##z", "##e"],
    ["t", "##s", "##i", "##p", "##e", "##t", "'", "s", "a", "'"],
    ["F", "##z", "##e", "##r", "##e", "##t", "##s", "##e", "##n", "-", "k", "##é", "##n", "##t"],
    ["j", "##ö", "##v", "##ő", "-", "b", "##é", "##l", "##i"]
]

test_correction_tokens = [
    ["Ö", "##s", "[PAD]", "##s", "##z", "##e"],
    ["c", "##s", "##i", "##p", "##e", "##t", "é", "##s", "a", "[PAD]"],
    ["S", "##z", "##e", "##r", "##e", "##c", "##s", "##e", "##n", "[PAD]", "##k", "##é", "##n", "##t"],
    ["j", "##ö", "##v", "##ő", "[PAD]", "##b", "##e", "##l", "##i"]
]
correction_to_old_list = [make_old.make_tokens_old(tokenizer_hu.tokenize(text))[0] for text in test_str]
old_list = [make_old.make_tokens_old(tokenizer_hu.tokenize(text))[1] for text in test_str]

def test_old_correction(mocker):
    assert all([a == b for a, b in zip(correction_to_old_list, test_correction_tokens)])

def test_old(mocker):
    assert all([a == b for a, b in zip(old_list, test_old_tokens)])