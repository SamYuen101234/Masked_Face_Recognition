from torch.utils.data.sampler import Sampler
import itertools
import numpy as np

def samples(df):
    label_to_samples = []
    samples = []
    label = 0
    for index, row in df.iterrows():
        if index == 0:
            samples.append(index)
            label = row['target']
        else:
            if row['target'] != label:
                label_to_samples.append(samples)
                samples = []
                label = row['target']
            samples.append(index)
    return label_to_samples

class PKSampler(Sampler):

    def __init__(self, data_source, p=15, k=20):
        super().__init__(data_source)
        self.p = p
        self.k = k
        self.data_source = data_source

    def __iter__(self):
        pk_count = len(self) // (self.p * self.k)
        for _ in range(pk_count):
            labels = np.random.choice(np.arange(len(self.data_source.label_to_samples)), self.p, replace=False)
            for l in labels:
                indices = self.data_source.label_to_samples[l]
                replace = True if len(indices) < self.k else False
                for i in np.random.choice(indices, self.k, replace=replace):
                    yield i

    def __len__(self):
        pk = self.p * self.k
        samples = ((len(self.data_source) - 1) // pk + 1) * pk
        return samples

def grouper(iterable, n):
    it = itertools.cycle(iter(iterable))
    for _ in range((len(iterable) - 1) // n + 1):
        yield list(itertools.islice(it, n))

# full label coverage per 'epoch'
class PKSampler2(Sampler):

    def __init__(self, data_source, p=15, k=20):
        super().__init__(data_source)
        self.p = p
        self.k = k
        self.data_source = data_source

    def __iter__(self):
        rand_labels = np.random.permutation(np.arange(len(self.data_source.label_to_samples)))
        for labels in grouper(rand_labels, self.p):
            for l in labels:
                indices = self.data_source.label_to_samples[l]
                replace = True if len(indices) < self.k else False
                for j in np.random.choice(indices, self.k, replace=replace):
                    yield j

    def __len__(self):
        num_labels = len(self.data_source.label_to_samples)
        samples = ((num_labels - 1) // self.p + 1) * self.p * self.k
        return samples