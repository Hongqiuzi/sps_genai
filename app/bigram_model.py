import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable


class BigramModel:
    def __init__(self, corpus: Iterable[str]):
        self.next_word_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.vocabulary: set[str] = set()

        for text in corpus:
            for sentence in re.split(r"[.!?]+", text.lower()):
                words = re.findall(r"[\w']+", sentence)
                self.vocabulary.update(words)
                for current, following in zip(words, words[1:]):
                    self.next_word_counts[current][following] += 1

    def generate_text(self, start_word: str, length: int) -> str:
        if length < 1:
            raise ValueError("length must be at least 1")

        start_words = re.findall(r"[\w']+", start_word.lower())
        if len(start_words) != 1:
            raise ValueError("start_word must contain exactly one word")

        current = start_words[0]
        if current not in self.vocabulary:
            raise ValueError(f"start_word '{start_word}' is not in the corpus")

        generated = [current]
        for _ in range(length - 1):
            next_words = self.next_word_counts.get(current)
            if not next_words:
                break

            words, counts = zip(*next_words.items())
            current = random.choices(words, weights=counts, k=1)[0]
            generated.append(current)

        return " ".join(generated)
