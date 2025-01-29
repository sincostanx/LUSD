import re
import nltk
from nltk import pos_tag

resources = [
    'punkt',
    'punkt_tab',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng'
]

downloader = nltk.downloader.Downloader()
for res in resources:
    if not downloader.is_installed(res):
        nltk.download(res)

def tokenize(text):
    return re.findall(r'\w+', text)

def find_prompt_difference(source_prompt, target_prompt):
    source_tokens = tokenize(source_prompt)
    target_tokens = tokenize(target_prompt)

    target_pos_tags = pos_tag(target_tokens)

    # longest common suffix
    min_len = min(len(source_tokens), len(target_tokens))
    suffix_length = 0
    for i in range(1, min_len + 1):
        if source_tokens[-i] != target_tokens[-i]:
            break
        suffix_length += 1

    # find differing substring
    differing_tokens = []
    differing_pos_tags = []
    for i, token in enumerate(target_tokens[:len(target_tokens) - suffix_length]):
        if i >= len(source_tokens) or source_tokens[i] != token:
            differing_tokens.append(token)
            differing_pos_tags.append(target_pos_tags[i][1])

    prior_text = ' '.join(target_tokens[:len(target_tokens) - suffix_length - len(differing_tokens)])
    
    is_noun = [tag in ('NN', 'NNS') for tag in differing_pos_tags]

    ARTICLES = {'a', 'an', 'the'}
    PREPOSITIONS = {'IN'}
    if len(differing_tokens) > 0:
        last_token = differing_tokens[-1]
        last_pos_tag = differing_pos_tags[-1]
        if (not is_noun[-1]) and (last_token not in ARTICLES) and (last_pos_tag not in PREPOSITIONS):
            start_idx = len(target_tokens[:len(target_tokens) - suffix_length - len(differing_tokens)]) + len(differing_tokens)
            for token, pos in target_pos_tags[start_idx:]:
                differing_tokens.append(token)
                is_noun.append(pos in ('NN', 'NNS'))
                if is_noun[-1]: break

    return prior_text, differing_tokens, is_noun

def compute_target_index(tokenizer, prior_text, diff_tokens, is_noun):
    num_prior_tokens = len(tokenizer._tokenize(prior_text))
    num_diff_tokens = [len(tokenizer._tokenize(word)) for word in diff_tokens]

    target_index = []
    counter = num_prior_tokens
    for i in range(len(num_diff_tokens)):
        counter += num_diff_tokens[i]

        if not is_noun[i]: continue
        else:
            target_index.append(counter)

    return target_index