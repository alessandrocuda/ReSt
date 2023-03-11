from importlib import resources
import re

from toolz.functoolz import pipe
from tqdm import tqdm
tqdm.pandas()



from src.data.preprocessing.emoji import translate_emoji
from src.data.preprocessing.string_manipulation import normalize_text, remove_symb, traslate_emoticons, remove_punctuation
from src.data.preprocessing.string_manipulation import add_space_before_capital_words, add_space_numb_char, add_space_before_hashtag
from src.data.preprocessing.string_manipulation import split_hashtags, fix_wrong_word
from src.data.preprocessing.string_manipulation import clean_url, clean_tag, clean_censured_bad_words, clean_hashtag_symbol, clean_consonants, clean_vowels, clean_laughs
from src.data.preprocessing.string_manipulation import replace_abbreviation, replace_acronyms
from src.data.preprocessing.polarity import get_polarity

from nltk.stem import SnowballStemmer
import spacy_udpipe

spacy_udpipe.download("it-postwita")

def print_verbose(text, verbose = 0):
    if verbose:
        print(text)


class PreProcessor:
    def __init__(self):
        with resources.path("src.resources", "it-sentiment_lexicon.lmf.xml") as bad_words:
            self.__bad_words = {word.rstrip().lower() for word in open(bad_words, 'r', encoding='utf8') if word.rstrip().lower() != ''}
        self.__stemmer = SnowballStemmer('italian')
        self.__nlp = spacy_udpipe.load("it-postwita")

    def __text_length(self, text):
        """
        Returns the length of the text.

        Parameters
        ----------
        text : str

        Returns
        -------
        integer
        """
        return len(text)

    def __count_hashtags(self, text):
        """
        Returns the number of hashtags inside the text.

        Parameters
        ----------
        text : str

        Returns
        -------
        integer
        """
        result = re.findall(r'#\S+', add_space_before_hashtag(text))

        return len(result)

    def __caps_lock_words(self, text):
        """
        Returns the percentage of words written in CAPS-LOCK.

        Parameters
        ----------
        text : str

        Returns
        -------
        float
        """
        words = text.split()
        count_caps_lock = 0
        number_of_words = len(words)
        
        for word in words:
            if word.isupper() == True:
                count_caps_lock = count_caps_lock + 1
                
        return ((count_caps_lock*100)//number_of_words)


    def __esclamations(self, text):
        """
        Returns the number of '!' inside the string.

        Parameters
        ----------
        text : str

        Returns
        -------
        int
        """
        return text.count('!')

    def __questions(self, text):
        """
        Returns the number of '?' inside the string.

        Parameters
        ----------
        text : str

        Returns
        -------
        int
        """
        return text.count('?')

    def __udpipe(self, text):
        """
        Returns the lemmas, pos, dep of the words in the text.
        
        Example: 'Io mangio una mela.' -> ['Io', 'mangiare', 'una', 'mela', '.']
                 'Io mangio una mela.' -> ['PRON', 'VERB', 'DET', 'NOUN', 'PUNCT']
                 'Io mangio una mela.' -> ['nsubj', 'ROOT', 'det', 'obj', 'punct']
        
        Parameters
        ----------
        text : str

        Returns
        -------
        list
            list of lemmas.

        list
            list of pos.
        
        list
            list of deps.
        """
        tokens = []
        lemma  = []
        pos    = []
        dep    = []

        for token in self.__nlp(text):
            tokens.append(token.text)
            lemma.append(token.lemma_)
            pos.append(token.pos_)
            dep.append(token.dep_)
        return tokens, lemma, pos, dep
        
    def get_token(self, text):
       tokens, lemma, pos, dep =  self.__udpipe(text)
       return tokens

    def __percentage_bad_words(self, tokens):
        """
        Returns the percentage of bad words.

        Parameters
        ----------
        tokens : list
            list of strings.

        Returns
        -------
        float
        """
        n_words = 0
        n_bad_words = 0
        
        for word in tokens:
            if word != '<' and word != '>':
                n_words = n_words + 1
        
        for word in tokens:
            if word.lower() in self.__bad_words:
                n_bad_words = n_bad_words + 1
            
        return ((n_bad_words*100)//n_words)

    def __stemming(self, tokens):
        """
        Stemming: reduces the inflected form of a word to its root form.

        Parameters
        ----------
        tokens : list
            list of strings.

        Returns
        -------
        list
            list of strings.
        """
        result = []
        
        for word in tokens:
            if word != '<' and word != '>':
                stemmed_word = self.__stemmer.stem(word)
                result.append(stemmed_word)
            else:
                result.append(word)
                
        return result

    def preprocess_df(self, df, column_name, path, verbose = 0):
        """
        Preprocesses the dataset and save the new dataset on the specified path.

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe to be preprocessed.

        column_name : str
            Name of the df's column containing the string to be preprocessed.

        stemming : bool
            If True, the stemming is applied.

        path : str
            Path where to save the preprocessed dataset.
            Example: 'train_val_datasets/train_val_no_hashtag_stemmed.csv'
        """
        print_verbose("Text Processing", verbose=verbose)
        df["processed_text"] =  df[column_name].progress_apply(self.process_text)

        #Feature extraction: length of the comment
        print_verbose("Feature extraction: length of the comment", verbose=verbose)
        df['text_length'] = df[column_name].progress_apply(self.__text_length)

        #Feature extraction: number of hashtags
        print_verbose("Feature extraction: number of hashtags", verbose=verbose)
        df['hashtags'] = df[column_name].progress_apply(self.__count_hashtags)

        #Feature extraction: percentage of words written in CAPS-LOCK
        print_verbose("Feature extraction: percentage of words written in CAPS-LOCK", verbose=verbose)
        df['%CAPS-LOCK words'] = df[column_name].progress_apply(self.__caps_lock_words)
        
        #Feature extraction: number of ‘!’ inside the text
        print_verbose("Feature extraction: esclamations", verbose=verbose)
        df['esclamations'] = df[column_name].progress_apply(self.__esclamations)

        #Feature extraction: number of ‘?’ inside the text
        print_verbose("Feature extraction: number of questions mark", verbose=verbose)
        df['questions'] = df[column_name].progress_apply(self.__questions)
        
        #Lemma
        print_verbose("generate Lemma, PoS, Dep", verbose=verbose)
        df["tokens"], df['lemma'],  df['pos'], df['dep'] = zip(*df["processed_text"].progress_apply(self.__udpipe))

        #Word polarity
        print_verbose("Generate Word polarity", verbose=verbose)
        df['word_polarity'], df['sentence_positive'], df['sentence_negative'], df['sentence_neutral']  = zip(*df['lemma'].progress_apply(get_polarity))

        #Stemming
        print_verbose("Generate Stemming", verbose=verbose)
        df['stem'] = df['tokens'].progress_apply(self.__stemming)

        #Feature extraction: percentage of Bad Words
        print_verbose("Feature extraction: percentage of Bad Words", verbose=verbose)
        df['%bad_words'] = df['tokens'].progress_apply(self.__percentage_bad_words)

        #Saving the preprocessed dataframe
        self.__save_df_csv(df.copy(), path)

    def process_text(self, text):
        return pipe(text,
                    add_space_before_hashtag,
                    add_space_numb_char,
                    #clean_url,
                    #clean_tag,
                    translate_emoji,
                    traslate_emoticons,
                    remove_symb,
                    split_hashtags,
                    remove_punctuation,
                    add_space_before_capital_words,
                    normalize_text,
                    clean_censured_bad_words,
                    clean_hashtag_symbol,
                    clean_laughs,
                    clean_vowels,
                    replace_abbreviation,
                    clean_consonants,
                    replace_acronyms,
                    fix_wrong_word
                    #self.__stick_apostrophe_text,
                    ) 

    def __save_df_csv(self, df, path):
        list_features = ["lemma", "pos", "dep", "word_polarity", "tokens", "stem"]
        
        for feature in list_features:
            df[feature] = df[feature].apply(lambda s : ' '.join([str(elem) for elem in s]))
        df.to_csv(path, sep="\t", index=False)