from typing import List


def masked_tensor_from_sentences(sentences: List[List[int]]):
    try:
        sentences[0][0]
    except TypeError:
        raise ValueError("masked_tensor_from_sentences() consumes List of Lists (batch X time)")
