from envs.gridworld_simple.instructions.tokenizer import InstructionTokenizer


def test_instruction_tokenizer_should_calculate_vocabulary_size():
    tokenizer = InstructionTokenizer(["Go to the blue object, go to the red object", "Go to the red object"])

    assert(tokenizer.get_vocabulary_size() == 10) # 3 reserved and 7 unique


def test_instruction_tokenizer_text_to_ids_should_have_start_token():
    tokenizer = InstructionTokenizer(["Go to the blue object, go to the red object", "Go to the red object"])

    ids = tokenizer.text_to_ids("please, go to the red object")
    assert(ids[0] == tokenizer.ID_START)


def test_instruction_tokenizer_text_to_ids_should_have_last_token():
    tokenizer = InstructionTokenizer(["Go to the blue object, go to the red object", "Go to the red object"])

    ids = tokenizer.text_to_ids("please, go to the red object")
    assert(ids[-1] == tokenizer.ID_END)


def test_instruction_tokenizer_text_to_ids_should_handle_unknown_tokens():
    tokenizer = InstructionTokenizer(["Go to the blue object, go to the red object", "Go to the red object"])

    ids = tokenizer.text_to_ids("please, go to the red object")
    assert(ids[1] == tokenizer.ID_UNK)


def test_instruction_tokenizer_should_handle_padding():
    padding_len = 20
    tokenizer = InstructionTokenizer(["Go to the blue object, go to the red object", "Go to the red object"], padding_len)

    ids = tokenizer.text_to_ids("please, go to the red object")
    assert(len(ids) == padding_len)