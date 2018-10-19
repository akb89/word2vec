def get_ctx_items(tokens, target_id, window_size):
    ctx_1 = tokens[max(0, target_id-window_size):target_id]
    ctx_2 = tokens[target_id+1:min(len(tokens), target_id+window_size+1)]
    return [*ctx_1, *ctx_2]

def get_example(training_data_filepath, window_size):
    with open(training_data_filepath, 'r') as training_datastream:
        for line in training_datastream:
            tokens = line.strip().split()
            for target_id, target in enumerate(tokens):
                for ctx in get_ctx_items(tokens, target_id, window_size):
                    yield self._word2id[target], self._word2id[ctx]

if __name__ == '__main__':
    window_size = 15
    training_data_filepath = '/Users/AKB/GitHub/nonce2vec/data/wikipedia/wiki.all.utf8.sent.split.lower'
    for 
