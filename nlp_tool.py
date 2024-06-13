__all__ = ['tokenize', 'pad_window', 'train']


from typing import List


def tokenize(sentence: str) -> List[str]:
    """
        Return a list of tokens extracted from the input sentence.

        Args
        ----
            sentence, `str`: input string
        
        Output
        ------
            `List[str]`: list of tokens

        Example
        -------
            tokenize(sentence='The sky is blue.') &rarr; ['the', 'sky', 'is', 'blue']
    """
    for punc in set([',', ';', '.', '?', '!', '/', "'", '-', '_']): sentence = sentence.replace(punc, '')
    return sentence.lower().split()


def pad_window(sentence: List[str], window_size: int) -> List[str]:
    """
        Return a list of padded tokens.\\
        The padding token is `'<pad>'`.

        Args
        ----
            sentence, `List[str]`: input list of tokens
            window_size, `int`: desired size of the padding window

        Output
        ------
             `List[str]`: padded list of tokens
        
        Example
        -------
            pad_window(sentence=['the', 'sky', 'is', 'blue'], window_size=2)
                &rarr; ['<pad>', '<pad>', 'the', 'sky', 'is', 'blue', '<pad>', '<pad>']
    """
    window = ['<pad>'] * window_size
    return window + sentence + window


def _train_epoch(loss_function, optimizer, model, loader):
    """
        Compute an epoch for the input model with the input loader data, loss function and optimizer.
    """
    total_loss = 0
    for batch_inputs, batch_labels, batch_lengths in loader:
        optimizer.zero_grad()  # clear gradient
        outputs = model.forward(batch_inputs)  # forward pass
        loss = loss_function(outputs, batch_labels, batch_lengths)  # batch loss
        loss.backward()  # gradient
        optimizer.step()
        total_loss += loss.item()
    return total_loss


def train(loss_function, optimizer, model, loader, num_epochs):
    """
        Compute num_epochs epochs for the input model with the input loader data, loss function and optimizer.
    """
    for epoch in range(0, num_epochs):
        epoch_loss = _train_epoch(loss_function, optimizer, model, loader)
        if epoch % 100 == 0: print(f'Epoch: {epoch+1} - Loss: {epoch_loss}')
