if __name__ == '__main__':
    vocab_filepath = '/Users/AKB/GitHub/nonce2vec/models/enwiki.20180920.utf8.lower.txt.vocab'
    freq_path = '/Users/AKB/GitHub/nonce2vec/models/enwiki.20180920.utf8.lower.txt.freq'
    word_count = {}
    with open(vocab_filepath, 'r') as vocab_stream:
        for line in vocab_stream:
            word, count = line.strip().split('\t')
            word_count[word] = int(count)
    total_count = sum(count for count in word_count.values())
    print('total count = {}'.format(total_count))
    with open(freq_path, 'w') as freq_stream:
        for word, count in word_count.items():
            print('{}\t{}'.format(word, count/total_count), file=freq_stream)
