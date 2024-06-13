import torch
import torch.nn as nn


class WordWindowClassifier(nn.Module):
    def __init__(self, hyperparameters, vocab_size, padding_idx=0):
        super(WordWindowClassifier, self).__init__()

        self.window_size = hyperparameters['window_size']
        self.embed_dim = hyperparameters['embed_dim']
        self.hidden_dim = hyperparameters['hidden_dim']
        self.freeze_embeddings = hyperparameters['freeze_embeddings']
        self.word_to_idx = hyperparameters['word_to_idx']
        self.idx_to_word = hyperparameters['idx_to_word']

        # Embedding Layer
        self.embeds = nn.Embedding(vocab_size, self.embed_dim, padding_idx=padding_idx)
        if self.freeze_embeddings: self.embed_layer.weight.requires_grad = False

        # Hidden Layer
        full_window_size = 2 * self.window_size + 1
        self.hidden_layer = nn.Sequential(
            nn.Linear(full_window_size * self.embed_dim, self.hidden_dim),
            nn.Tanh()
        )

        # Output layer
        self.output_layer = nn.Linear(self.hidden_dim, 1)

        # Probabilities
        self.probabilities = nn.Sigmoid()

    def forward(self, inputs):
        B, _ = inputs.size()

        # Reshaping
        token_windows = inputs.unfold(1, 2 * self.window_size + 1, 1)
        _, adjusted_length, _ = token_windows.size()
        assert token_windows.size() == (B, adjusted_length, 2 * self.window_size + 1)

        # Embedding
        embedded_windows = self.embeds(token_windows)

        # Reshaping
        embedded_windows = embedded_windows.view(B, adjusted_length, -1)

        # Layer 1
        layer_1 = self.hidden_layer(embedded_windows)

        # Layer 2
        output = self.output_layer(layer_1)

        # Softmax Score
        output = self.probabilities(output)
        output = output.view(B, -1)

        return output

    def predict(self, input):
        for punc in set([',', ';', '.', '?', '!', '/', "'", '-', '_']): input = input.replace(punc, '')
        tokens = input.lower().split()
        window = ['<pad>'] * self.window_size
        padded_tokens = window + tokens + window
        tokens_idx = [self.word_to_idx[token] for token in padded_tokens]
        output = self.forward(torch.tensor([tokens_idx, tokens_idx]))
        mask = output[0] > 0.5
        target_index = mask.nonzero(as_tuple=True)[0]
        if len(target_index) == 0: return None
        pred_tokens = [padded_tokens[idx] for idx in target_index]
        pred = ' '.join(token for token in pred_tokens)
        return pred
