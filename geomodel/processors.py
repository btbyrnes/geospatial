def words_to_index_generator(word_dict, sentences):

    i = 1 + len(word_dict.keys())

    for sentence in sentences:
        for word in sentence.split(' '):
            if word not in word_dict.keys():
                word_dict[word] = i
                i += 1

    return word_dict

def cbow_context_target_generator(word_dict, sentences, window, verbose=False):

    targets = []
    contexts = []

    for sentence in sentences:

        sentence = sentence.split(' ')
        sentence_length = len(sentence)
        if sentence_length >= 2*window + 1:

            for i in range(window,sentence_length-window):

                context = []
                context_words = []
                target = word_dict[sentence[i]]

                for j in range(i - window, i):

                    context_words.append(sentence[j])
                    context.append(word_dict[sentence[j]])
                
                for j in range(i + 1, i + 1 + window):

                    context_words.append(sentence[j])
                    context.append(word_dict[sentence[j]])
                
                if verbose == True:
                    s = " "
                    s = s.join(context_words)
                    print(f"context: {s} --> {sentence[i]} :target")

                targets.append(target)
                contexts.append(context)

    return (targets,contexts)