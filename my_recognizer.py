import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    for word_id in range(0, len(test_set.get_all_Xlengths())):
        x, lengths = test_set.get_item_Xlengths(word_id)
        scores = {}

        # Calculate Log Likelihood score for each word and model and append to probability list
        for word, model in models.items():
            try:
                score = model.score(x, lengths)
                scores[word] = score
            except:
                scores[word] = float("-inf")
                continue
        probabilities.append(scores)
        guesses.append(max(scores, key = scores.get))

    return probabilities, guesses

