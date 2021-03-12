from bs4 import BeautifulSoup
import re
from importlib import resources

def load_word_polarity(word_polarity_xml):

    #Word polarity dict

    # Reading the data inside the xml 
    # file to a variable under the name  
    # data 
    with open(word_polarity_xml, 'r') as f: 
        data = f.read() 

    # Passing the stored data inside 
    # the beautifulsoup parser, storing 
    # the returned object  
    Bs_data = BeautifulSoup(data, "xml") 

    word_polarity = {}

    lemma_unique = Bs_data.find_all('Lemma')         #Finding all instances of tag 'Lemma'
    sentiment_unique = Bs_data.find_all('Sentiment') 

    if len(lemma_unique) != len(sentiment_unique):
        print('ERRORE')

    for i in range(0, len(lemma_unique)):
        word = lemma_unique[i].get('writtenForm') #Extracting the data stored in a specific attributes of the 'Lemma' tag
        word = re.sub(r'_', ' ', word)
        
        polarity = sentiment_unique[i].get('polarity')
        if polarity is None:
            polarity = 'neutral'
        
        word_polarity[word] = polarity

    # fix word polarity typo
    word_polarity['aspettare'] = 'neutral'
    return word_polarity


with resources.path("src.resources", "it-sentiment_lexicon.lmf.xml") as word_polarity_xml:
    word_polarity = load_word_polarity(word_polarity_xml)

polarity_map = {
    'negative': -1, 
    'neutral': 0,
    'positive': 1
}

def get_polarity(lemma_text):
    """
    Returns the word polarity of each lemma ('positive'/'neutral'/'negative').
    The polarity of unfamiliar words is set to 'neutral'.

    Example: ['Io', 'amare', 'mangiare'] -> ['neutral', 'positive', 'neutral']

    Parameters
    ----------
    lemmas : list

    Returns
    -------
    list
        list of word polarity.
    """
    polarity = []
    
    for word in lemma_text:
        if word in word_polarity:
            polarity.append(polarity_map[word_polarity[word]])
        else:
            polarity.append(0)
    
    positive = polarity.count(1)
    negative = polarity.count(-1)
    neutral = polarity.count(0)
    

    sentence_positive = positive/len(lemma_text)
    sentence_negative = negative/len(lemma_text)
    sentence_neutral  = neutral/len(lemma_text)
    
    return polarity, sentence_positive, sentence_negative, sentence_neutral



