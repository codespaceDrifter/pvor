#download oxford txt from  https://github.com/first20hours/google-10000-english/blob/master/20k.txt
# add whitespace for tokenization

def add_spaces_to_words(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            word = line.strip()
            if word:  # skip empty lines
                fout.write(word + ' \n')

# Example usage
add_spaces_to_words('20k.txt', '20kSpaced.txt')
