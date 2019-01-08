def decode_with_labels(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    label = [int(line[0]) for line in lines] 
    words = [np.fromstring(line[2:], sep=' ') for line in lines]
    return words, label

def decode_without_labels(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    words = [np.fromstring(line[2:], sep=' ') for line in lines]
    return words
