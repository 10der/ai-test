import numpy as np
import logging
from typing import Callable
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging
import re

class IntentClassifier:
    def __init__(self, model_path: str = "../MiniLM", threshold: float = 0.7):
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        self.model = SentenceTransformer(model_path)
        self.threshold = threshold
        self.embeddings_matrix = np.array([], dtype=np.float32)
        self.metadata = []
        self.raw_intents = []

    def add_intent(self, texts: str | list[str], tool: Callable, params: dict = {}):
        if isinstance(texts, str):
            texts = [texts]
        for text in texts:
            self.raw_intents.append(
                {"text": text, "tool": tool, "params": params})

    def build_index(self):
        if not self.raw_intents:
            return

        texts = [i['text'] for i in self.raw_intents]
        embeddings = self.model.encode(
            texts, convert_to_numpy=True).astype('float32')

        # L2-нормалізація
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings_matrix = embeddings / norms
        self.metadata = [{"tool": i['tool'], "params": i['params']}
                         for i in self.raw_intents]
        logging.debug(f"Index built with {len(self.metadata)} intents.")

    def normalize(self, text: str):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text

    def predict(self, user_text: str) -> dict | None:
        if self.embeddings_matrix.size == 0:
            return None

        user_text = self.normalize(user_text)
        query_vec = self.model.encode(
            user_text, convert_to_numpy=True).astype('float32')
        query_vec /= np.linalg.norm(query_vec)

        similarities = np.dot(self.embeddings_matrix, query_vec)
        best_idx = np.argmax(similarities)
        score = similarities[best_idx]

        # logging.debug(f"'{user_text}' -> Raw Score: {score:.3f}")

        if score >= self.threshold:
            return self.metadata[best_idx]

        return None
