from lora_serve.task import LABELS, build_messages, parse_label


def test_parse_valid():
    assert parse_label('{"label": "Sports"}') == "Sports"


def test_parse_with_surrounding_text():
    assert parse_label('Sure! {"label": "Sci/Tech"} done') == "Sci/Tech"


def test_parse_rejects_unknown_label():
    assert parse_label('{"label": "Weather"}') is None


def test_parse_rejects_extra_keys():
    assert parse_label('{"label": "World", "why": "x"}') is None


def test_parse_rejects_non_json():
    assert parse_label("Sports") is None
    assert parse_label("") is None
    assert parse_label("{label: Sports}") is None


def test_messages_zero_and_few_shot():
    zero = build_messages("hello world")
    few = build_messages("hello world", few_shot=True)
    assert zero[0]["role"] == "system" and zero[-1]["content"].endswith("hello world")
    assert len(few) == len(zero) + 8
    assert all(parse_label(m["content"]) in LABELS for m in few if m["role"] == "assistant")
