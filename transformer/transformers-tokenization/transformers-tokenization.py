class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self.word_to_id = {token: idx for idx, token in enumerate(special_tokens)}
        self.id_to_word = {idx: token for idx, token in enumerate(special_tokens)}

        unique_words = set()
        for text in texts:
            words=text.lower().split()
            unique_words.update(words)

        for word in sorted(unique_words):
            idx = len(self.word_to_id)
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word

        self.vocab_size = len(self.word_to_id)
        
    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        unk_id = self.word_to_id[self.unk_token]
        encoding = []
        for word in text.lower().split():
            encoding.append(self.word_to_id.get(word, unk_id))
        return encoding

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        words = []
        for token_id in ids:
            words.append(self.id_to_word.get(token_id, self.unk_token))
        return " ".join(words)
            
            