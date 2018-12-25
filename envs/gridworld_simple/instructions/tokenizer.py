class InstructionTokenizer:
    ID_UNK = 0
    ID_START = 1
    ID_END = 2

    def __init__(self, instructions, padding_len=None):
        '''
        instructions - an array of natural language instructions
        '''
        self._instructions = instructions
        self._padding_len = padding_len
        self._token_to_id = {}


        # Find all unique tokens
        for instruction in instructions:
            tokenized = self._tokenize(instruction)
            for token in tokenized:
                if token not in self._token_to_id:
                    self._token_to_id[token] = len(self._token_to_id) + 3

    def _tokenize(self, text):
        text = str.lower(text)
        text = text.replace(",", " ,")
        return str.split(text, " ")

    def text_to_ids(self, text):
        tokenized = self._tokenize(text)
        ids = [self.ID_START]
        for token in tokenized:
            if token not in self._token_to_id:
                ids.append(self.ID_UNK)
            else:
                ids.append(self._token_to_id[token])
        ids.append(self.ID_END)

        if self._padding_len:
            if len(ids) < self._padding_len:
                ids.extend([self.ID_UNK for _ in range(self._padding_len - len(ids))])
                
        return ids

    def get_vocabulary_size(self):
        return len(self._token_to_id) + 3