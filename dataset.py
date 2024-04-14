import os
import numpy as np
import torch
from torch.utils.data import Dataset

class LentaDataBank:
    '''
    Wrapper for accessing Lenta Dataset
    
    Parses dataset, gives each token and index and provides lookups
    from string token to index and back
    
    Allows to generate random context with sampling strategy described in
    word2vec paper:
    https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
    '''
    def __init__(self):
        self.index_by_token = {}
        self.token_by_index = []

        self.sentences = []

        self.token_freq = {}

        self.token_reject_by_index = None

    def load_dataset(self, folder):
        filename = os.path.join(folder, "sentences.txt")

        with open(filename, "r", encoding="utf-8") as f:
            for l in f:
                splitted_line = l.strip().split()
                words = [w.lower() for w in splitted_line]

                self.sentences.append(words)
                for word in words:
                    if word in self.token_freq:
                        self.token_freq[word] += 1
                    else:
                        index = len(self.token_by_index)
                        self.token_freq[word] = 1
                        self.index_by_token[word] = index
                        self.token_by_index.append(word)
        self.compute_token_prob()

    def compute_token_prob(self):
        words_count = np.array([self.token_freq[token] for token in self.token_by_index])
        words_freq = words_count / np.sum(words_count)

        self.token_reject_by_index = 1 - np.sqrt(1e-5/words_freq)

    def check_reject(self, word):
        '''
        Returns True if the token must NOT be rejected
        '''
        return np.random.rand() > self.token_reject_by_index[self.index_by_token[word]]

    def get_random_context(self, context_length=5):
        """
        Returns tuple of center word and list of context words
        """   
        sentence_sampled = []
        while len(sentence_sampled) <= 2:
            sentence_index = np.random.randint(len(self.sentences))
            sentence = self.sentences[sentence_index]
            sentence_sampled = [word for word in sentence if self.check_reject(word)]

        center_word_index = np.random.randint(len(sentence_sampled))

        words_before = sentence_sampled[max(center_word_index - context_length//2,0):center_word_index]
        words_after = sentence_sampled[center_word_index+1: center_word_index+1+context_length//2]
        
        return sentence_sampled[center_word_index], words_before+words_after

    def num_tokens(self):
        return len(self.token_by_index)


class Word2VecDataset(Dataset):
    '''
    PyTorch Dataset for Word2Vec with Negative Sampling.
    '''
    def __init__(self, data, num_negative_samples, num_contexts=30000):

        self.data = data
        self.num_negative_samples = num_negative_samples
        self.num_contexts = num_contexts
        self.num_tokens = data.num_tokens()
        self.samples = []

    def generate_dataset(self):
        self.samples = []
        for _ in range(self.num_contexts):
            word, context = self.data.get_random_context()
            for target in context:
                word_index = self.data.index_by_token[word]
                target_index = self.data.index_by_token[target]

                output_indices = np.random.randint(self.num_tokens, size=self.num_negative_samples+1)
                output_indices = torch.from_numpy(output_indices)
                output_indices[0] = target_index

                output_target = torch.zeros(self.num_negative_samples+1, dtype=torch.float32)
                output_target[0] = 1.0

                sample = (word_index, output_indices, output_target)
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
