import numpy as np

def shuffle_sample_idx(sample_indices):
    """"""
    sample_shuffled = sample_indices[:]
    np.random.shuffle(sample_shuffled)
    return sample_shuffled

def split_sample_idx(sample_indices, batch_size):
    """"""
    num_samples = len(sample_indices)
    sample_batch = []
    num_batch = num_samples//batch_size
    for i in xrange(num_batch):
        bat_i = batch_size*i
        bat_f = batch_size*(i+1) if i < num_batch-1 else num_samples
        sample_batch.append(sample_indices[bat_i:bat_f])
    return sample_batch

def shuffle_dataset(dataset, labels, sample_indices=None):
    """"""
    num_samples = len(dataset)
    if not sample_indices:
        sample_shuffled = shuffle_sample_idx(range(num_samples))
    return dataset[sample_shuffled], labels[sample_shuffled]

def group_dataset_by_label(dataset, labels, label_size=10):
    """"""
    xs = [[] for l in xrange(label_size)]
    ys = [[] for l in xrange(label_size)]
    for l in xrange(label_size):
        for i in xrange(len(dataset)):
            if labels[i] == l:
                xs[l] += [dataset[i]]
                ys[l] += [l]
    return xs, ys

def subset_dataset_by_label(dataset, labels, samples_per_label, label_size=10, shuffle=False):
    """"""
    xs, ys = group_dataset_by_label(dataset, labels, label_size)
    label_order = range(label_size)
    if not shuffle:
        xij = [xs[i][j] for j in xrange(samples_per_label) for i in label_order]
        yij = [ys[i][j] for j in xrange(samples_per_label) for i in label_order]
    else:
        xij = []
        yij = []
        for j in xrange(samples_per_label):
            np.random.shuffle(label_order)
            for i in label_order:
                xij.append(xs[i][j])
                yij.append(ys[i][j])
    return np.array(xij), np.array(yij)
