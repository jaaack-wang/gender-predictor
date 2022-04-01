from random import seed, shuffle 
from paddlenlp.datasets import MapDataset
from paddle.io import BatchSampler, DataLoader
from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.data import Vocab


def relabel(g, reverse=False):
    if reverse:
        return {0: 'M', 1: 'F', 2: 'U'}[g]
    return {'M': 0, 'F': 1, 'U': 2}[g]


def data_loader(filepath, num_row_skip=1, first_name_only=True, exclude_u=False):
    name_idx = -3 if first_name_only else -2

    def readFile(path):
        f = open(path, 'r')
        for _ in range(num_row_skip):
            next(f)
        out = []
        for line in f:
            line = line.split('\t')
            name, gender = line[name_idx], relabel(line[-1].strip())
            if exclude_u:
                if gender != 2:
                    out.append([name, gender])
            else:
                out.append([name, gender])
        return out
    
    if isinstance(filepath, str):
        return readFile(filepath)
    elif isinstance(filepath, list):
        return [readFile(path) for path in filepath]
    else:
         raise TypeError('filepath must be either a str or a list.')


def train_dev_test_split(data, train=0.6, dev=0.2, test=0.2, seed_idx=5):
    seed(seed_idx)
    shuffle(data)
    length = len(data)
    boundary1 = round(length * train)
    boundary2 = round(length * (train + dev))    
    return data[:boundary1], data[boundary1: boundary2], data[boundary2:]


class TextVectorizer:
     
    def __init__(self, tokenizer=None):
        self.tokenize = tokenizer
        self.vocab_to_idx = None
        self._V = None
    
    def build_vocab(self, text):
        tokens = list(map(self.tokenize, text))
        self._V = Vocab.build_vocab(tokens, unk_token='[UNK]', pad_token='[PAD]')
        self.vocab_to_idx = self._V.token_to_idx
        
    def text_encoder(self, text):
        if isinstance(text, list):
            return [self(t) for t in text]
        
        tks = self.tokenize(text)
        out = [self.vocab_to_idx[tk] for tk in tks]
        return out

    def __len__(self):
        return len(self.vocab_to_idx)

    def __getitem__(self, w):
        return self.vocab_to_idx[w]
    
    def __call__(self, text):
        if self.vocab_to_idx:
            return self.text_encoder(text)
        raise ValueError("No vocab is built!")


def example_converter(example, text_encoder, include_seq_len):
    
    text, label = example
    encoded = text_encoder(text)
    if include_seq_len:
        text_len = len(encoded)
        return encoded, text_len, label
    return encoded, label


def get_trans_fn(text_encoder, include_seq_len):
    return lambda ex: example_converter(ex, text_encoder, include_seq_len)


def get_batchify_fn(include_seq_len):
    
    if include_seq_len:
        stack = [Stack(dtype="int64")] * 2
    else:
        stack = [Stack(dtype="int64")]
    
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=0),  
        *stack
    ): fn(samples)
    
    return batchify_fn


def create_dataloader(dataset, 
                      trans_fn, 
                      batchify_fn, 
                      batch_size=128, 
                      shuffle=True, 
                      sampler=BatchSampler):

    if not isinstance(dataset, MapDataset):
        dataset = MapDataset(dataset)
        
    dataset.map(trans_fn)
    batch_sampler = sampler(dataset, 
                            shuffle=shuffle, 
                            batch_size=batch_size)
    
    dataloder = DataLoader(dataset, 
                           batch_sampler=batch_sampler, 
                           collate_fn=batchify_fn)
    
    return dataloder
