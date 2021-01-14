from correction import correction_dataset_generator
from transformers import BertTokenizerFast
import pytest

@pytest.fixture
def tokenizer_hu():
    return BertTokenizerFast("data/tokenizer/hu/alphabet", do_lower_case=False)

@pytest.fixture
def old_gen(tokenizer_hu):
    return correction_dataset_generator.MakeTextOld(tokenizer_hu)

@pytest.fixture
def ocr_error_table(tokenizer_hu):
    return correction_dataset_generator.ErrorTable(tokenizer_hu)

@pytest.fixture
def seq_length():
    return 64

@pytest.fixture
def data_gen(tokenizer_hu, ocr_error_table, seq_length, old_gen):
    return correction_dataset_generator.CorrectionDatasetGenerator(tokenizer_hu, ocr_error_table, seq_length, old_gen)

# def test_xa(mocker):
#     tokenizer = mocker.MagicMock()
#     tokenizer.get_vocab().keys = lambda: ['a', 'b', '##a', '##b', 'my_pad']
#     tokenizer.pad_token = 'my_pad'
#     generator = correction_dataset_generator.CorrectionDatasetGenerator(tokenizer, mocker.MagicMock(), 5)
#     assert generator.vocab == ['##a', '##b', 'a', 'b']

test_data_correction_gen = (
    ("Össze", ["Ö", "##s", "[PAD]", "##s", "##z", "##e"]),
    ("csipet és a cs", ["c", "##s", "##i", "##p", "##e", "##t", "é", "##s", "a", "[PAD]", "c", "##s"]),
    ("Szerecsenként", ["S", "##z", "##e", "##r", "##e", "##c", "##s", "##e", "##n", "[PAD]", "##k", "##é", "##n", "##t"]),
    ("jövőbeli", ["j", "##ö", "##v", "##ő", "[PAD]", "##b", "##e", "##l", "##i"]),
    ("menjen", ["m", "##e", "[PAD]", "##n", "##j", "##e", "##n"]),
    ("állja", ["á", "##l", "##l", "##j", "##a"]),
    ("üljön", ["ü", "[PAD]", "##l", "##j", "##ö", "##n"]),
    ("különb", ["k", "##ü", "##l", "##ö", "##n", "##b"]),
    ("megszentel", ["m", "##e", "##g", "[PAD]", "##s", "##z", "##e", "##n", "##t", "##e", "##l"]),
    ("tömegel", ["t", "##ö", "##m", "##e", "##g", "##e", "##l"]),
    ("ezt is sz", ["e", "##z", "##t", "[PAD]", "i", "##s", "s", "##z"]),
    ("ismer", ["i", "##s", "##m", "##e", "##r"]),
    ("ész", ["é", "##s", "##z"]),
    ("apad", ["a", "##p", "##a", "##d"]),
    ("legjobb r", ["l", "##e", "##g", "[PAD]", "##j", "##o", "##b", "##b", "r"]),
    ("legjobban", ["l", "##e", "##g", "[PAD]", "##j", "##o", "##b", "##b", "##a", "##n"]),
)

test_data_old_gen = (
    ("Össze", ["Ö", "##f", "##z", "##f", "##z", "##e"]),
    ("csipet és a cs", ["t", "##s", "##i", "##p", "##e", "##t", "'", "s", "a", "'", "t", "##s"]),
    ("Szerecsenként", ["F", "##z", "##e", "##r", "##e", "##t", "##s", "##e", "##n", "-", "k", "##é", "##n", "##t"]),
    ("jövőbeli", ["j", "##ö", "##v", "##ő", "-", "b", "##é", "##l", "##i"]),
    ("menjen", ["m", "##e", "##n", "##n", "##y", "##e", "##n"]),
    ("állja", ["á", "##l", "##l", "##y", "##a"]),
    ("üljön", ["ü", "##l", "##l", "##y", "##ö", "##n"]),
    ("különb", ["k", "##ü", "##l", "##ö", "##m", "##b"]),
    ("megszentel", ["m", "##e", "##g", "-", "##f", "##z", "##e", "##n", "##t", "##e", "##l"]),
    ("tömegel", ["t", "##ö", "##m", "##e", "##g", "##e", "##l"]),
    ("ezt is sz", ["e", "##z", "##t", "-", "i", "##s", "f", "##z"]),
    ("ismer", ["i", "##s", "##m", "##e", "##r"]),
    ("ész", ["é", "##f", "##z"]),
    ("apad", ["a", "##p", "##a", "##d"]),
    ("legjobb r", ["l", "##e", "##g", "-", "##j", "##o", "##b", "'", "r"]),
    ("legjobban", ["l", "##e", "##g", "-", "##j", "##o", "##b", "##b", "##a", "##n"]),
)


# ((old_tokens, correct_tokens), tokens_with_ocr_errors)
test_data_ocr_typo_error = (
    ((["s", "##s", "##z"], ["s", "##s", "##z"]), [["s", "##z", "##s", "##z"], ["s", "##z", "-", "s", "##z"], ["s", "##z", "s", "##z"]]),
    ((["i", "##i"], ["i", "##i"]), [["ű"], ["ü"], ["l", "##l"], ["u"], ["t", "##i"], ["i", "##t"], ["n"]]),
    ((["f"], ["s"]), [["i"], ["l"], ["1"], ["í"], ["ï"], ["j"], ["t"], ["r"], ["!"], ["|"], ["I"], ["J"], ["Í"]]),
)

test_data_ocr_typo_correction = (
    ((["s", "##s", "##z"], ["s", "##s", "##z"]), [
        [["s"], ["[PAD]"], ["##s"], ["##z"]],
        [["s"], ["[PAD]"], ["[PAD]"], ["##s"], ["##z"]]
    ]),
    ((["i", "##i"], ["i", "##i"]), [
        [["i", "##i"]],
        [["i"], ["##i"]]
    ]),
    ((["f"], ["s"]), [
        [["s"]],
    ]),
)




@pytest.mark.parametrize("test_input,expected", [(inp, outp) for inp, outp in test_data_old_gen])
def test_old_gen(test_input, expected, tokenizer_hu, old_gen):
    assert old_gen.make_tokens_old(tokenizer_hu.tokenize(test_input))[1] == expected


@pytest.mark.parametrize("test_input,expected", [(inp, outp) for inp, outp in test_data_correction_gen])
def test_correction_gen(test_input, expected, tokenizer_hu, old_gen):
    assert old_gen.make_tokens_old(tokenizer_hu.tokenize(test_input))[0] == expected


def test_get_error_1(ocr_error_table):
    with open("ocr_errors/hu/ocr_errors.txt", encoding="utf-8") as f:
        ocr_error_table.load_table_from_file(f)
    assert ocr_error_table.get_error2(["s", "##s", "##z"])[1] in [("s", "##z", "##s", "##z"), ("s", "##z", "-", "s", "##z"), ("s", "##z", "s", "##z")]


def test_get_error_0(ocr_error_table):
    with open("ocr_errors/hu/ocr_errors.txt", encoding="utf-8") as f:
        ocr_error_table.load_table_from_file(f)
    assert ocr_error_table.get_error2(["s", "##s", "##z"])[0] in [("s", "[PAD]", "##s", "##z"), ("s", "[PAD]", "[PAD]", "##s", "##z")]


@pytest.mark.parametrize("test_input,expected", [(inp, outp) for inp, outp in test_data_ocr_typo_error])
def test_ocr_typo_error(test_input, expected, tokenizer_hu, data_gen, ocr_error_table):
    data_gen._error_tokens = []
    data_gen._correct_tokens = []
    with open("ocr_errors/hu/ocr_errors.txt", encoding="utf-8") as f:
        ocr_error_table.load_table_from_file(f)
    data_gen.make_ocr_typo(0, test_input[0], test_input[1])
    assert data_gen._error_tokens in expected


@pytest.mark.parametrize("test_input,expected", [(inp, outp) for inp, outp in test_data_ocr_typo_correction])
def test_ocr_typo_correction(test_input, expected, tokenizer_hu, data_gen, ocr_error_table):
    data_gen._error_tokens = []
    data_gen._correct_tokens = []
    with open("ocr_errors/hu/ocr_errors.txt", encoding="utf-8") as f:
        ocr_error_table.load_table_from_file(f)
    data_gen.make_ocr_typo(0, test_input[0], test_input[1])
    assert data_gen._correct_tokens in expected

