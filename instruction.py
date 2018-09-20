import torch

class FindInstructionEncoder:
    '''
    Encodes a sequence of targets, e.g ["Triangle", "Circle", "Square"]
    Meaning that, triangle must be found first, then circle, and then square.
    '''
    def __init__(self, vocabulary, sequence_length=3):
        '''
        vocabulary - a set of possible items
        sequence_length - maximum length of the instruction
        '''
        self._inverted_index = {word:index for index, word in enumerate(vocabulary)}
        self._sequence_length = sequence_length
    
    def encode(instruction):
        '''
        input:
            instruction - an array of items, e.g ["Triangle", "Circle", "Square"]
        output:
            encoding - a binary encoding, e.g [1, 0, 0, 0, 1, 0, 0, 0, 1]
        '''
        num_words = len(self._inverted_index)
        encoding = torch.zero(1, num_words * self.sequence_length, dtype=torch.LongTensor)
        for seq_index, word in enumerate(instruction):
            seq_offset = num_words * sequence_length
            encoding[0, seq_offset + self._inverted_index[word]] = 1



class FindInstructionNaturalLanguageEncoder():
    raise NotImplementedError()