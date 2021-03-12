import re
pattern_space_before_capital = re.compile(r'((?<=[^\W[A-Z])[A-Z]|(?<=\S)[A-Z](?=[a-z]))')

from importlib import resources


############################
#       Segmenter
###########################
import wordninja
with resources.path("src.resources", "italian_words.txt.gz") as italian_words_gz:
    segmenter = wordninja.LanguageModel(italian_words_gz)

############################
#   unique_italian_words
###########################
with resources.path("src.resources", "parole_uniche.txt") as unique_italian_words_path:
    unique_italian_words   = {word.rstrip().lower() for word in open(unique_italian_words_path, 'r', encoding='utf8') if word.rstrip().lower() != ''}

# from ekphrasis.classes.preprocessor import TextPreProcessor
# from ekphrasis.classes.tokenizer import SocialTokenizer

from src.data.preprocessing.dicts.emoticons import emoticons
from src.data.preprocessing.dicts.wrong_word import wrong_word
from src.data.preprocessing.dicts.abbreviations import abbr_word, acronyms
# import spacy_udpipe
# spacy_udpipe.download("it-postwita")
# nlp = spacy_udpipe.load("it-postwita")

# social_tokenizer = lambda text : [  token.text for token in nlp(text)]

from ekphrasis.classes.exmanager import ExManager

regexes = ExManager().get_compiled()
backoff = ['url', 'email', 'user', 'percent', 'money', 'phone', 'time', 'date']


# text_processor = TextPreProcessor(
#     normalize=['url', 'email', 'user', 'percent', 'money', 'phone', 'time', 'date'],
#     fix_html=True,  # fix HTML tokens
    
#     # select a tokenizer. You can use SocialTokenizer, or pass your own
#     # the tokenizer, should take as input a string and return a list of tokens
#     tokenizer= social_tokenizer, #SocialTokenizer(lowercase=False).tokenize,
    
#     # list of dictionaries, for replacing tokens extracted from the text,
#     # with other expressions. You can pass more than one dictionaries.
#     dicts=[emoticons]
# )

token_syb = {'< url >'     :   "_URL_", 
             '< email >'   :   "_EMAIL_",
             '< user >'    :   "_USER_", 
             '< percent >' :   "_PERCENT_",
             '< money >'   :   "_MONEY_",
             '< phone >'   :   "_PHONE_",
             '< time >'    :   "_TIME_", 
             '< date >'    :   "_DATE_"
}

############################
#       UTILS
###########################
def is_integer(n):
    try:
        return int(n)
    except ValueError:
        return False
    
def is_float(n):
    try:
        return float(n)
    except ValueError:
        return False 

############################
#       General
###########################
def add_space_before_hashtag(text):
    return ' '.join(re.sub("#", " #", text).split())

def add_space_numb_char(text):
    return ' '.join([re.sub(r"(?i)(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)"," ", word) if word[0] != "#" else word for word in text.split() ])

def add_space_before_capital_words(text):
    """
    Adds a space after a lowercase letter if it is followed by an uppercase letter.

    Example: 'lAriaCheTira' -> 'l Aria Che Tira'

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    return re.sub(pattern_space_before_capital, r" \1", text)

def traslate_emoticons(text):
    for item in ["ltr_face", "rtl_face"]:
        text = re.sub(regexes[item], r' \g<0> ', text)

    return ' '.join([emoticons[word] if word in emoticons else word for word in text.split()])

def fix_wrong_word(text):
    text = ' '.join([ wrong_word[word] if word in wrong_word else word for word in text.split()])
    return text
    
############################
#   Normalize Text
############################

def normalize_numb(text):
    """
    Normalizes numbers. 
    - Integer numbers between 0 and 2100 were kept as original.
    - Each integer number greater than 2100 is mapped in a string which 
        represents the number of digits needed to store the number (ex: 10000 -> DIGLEN_5)
    - Each digit in a string that is not convertible to a number must be converted with the 
        following char: @Dg. This is an example of replacement (ex: 10,234 -> @Dg@Dg,@Dg@Dg@Dg)

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    val = is_integer(text)
    if val and val >= 0 and val < 2100:
        return str(val)
    elif val:
        #return "DIGLEN" + str(len(str(val)))
        return " "
    elif is_float(text):
        #return re.sub(r'\d', '@Dg', text)
        return " "
    return text

def remove_symb(text):
    text = ' ' + text + ' '
    text = text.replace("‘’", ' " ')
    text = text.replace("''", ' " ')
    text = text.replace('""', ' " ')
    text = text.replace("'", "' ")
    text = text.replace("’", "' ")
    text = text.replace("`", "' ")
    text = text.replace("‘", "' ")
    text = text.replace("ʼ", "' ")
    text = text.replace("´", "' ")
    text = fix_wrong_word(text) 
    text = text.replace("' ", "'")   
    text = re.compile(r"[']([\w]+)[']", re.UNICODE).sub(r"' \1 ' ", text)
    text = text.replace("'", "' ")
    
    text = ' '.join([word.replace("'", " '") if "'" in word and len(word) > 5 else word for word in text.split()])



    text = text.replace("�", ' ')
    text = text.replace("<", ' ')
    text = text.replace(">", ' ')
    text = text.replace("<", ' ')
    text = text.replace(r"…", ' ')

    text = text.replace("•", " ")
    text = text.replace("=", " ")
    text = text.replace("|", " ")
    text = text.replace("&", " ")
    text = text.replace("*"," ")
    text = text.replace("“", ' " ')    
    text = text.replace("”", ' " ')




    text = re.sub(r'\\n', '. ', text)
    text = re.sub(r'\\', ' ', text)
    text = re.sub(r'/', ' ', text)
    text =  re.sub(r'_', ' ', text)

    return text

def remove_punctuation(text):
    text = text.replace(r'-', ' ')
    text = text.replace(r'—', ' ')
    text = text.replace(r'–', ' ')
    return text


def normalize_text(text):
    """
    Normalizes text.
    - A string starting with lower case character must be lowercased 
        (e.g.: (“aNtoNio” -> “antonio”), (“cane” -> “cane”))
    - A string starting with an upcased character must be capitalized 
        (e.g.: (“CANE” -> “Cane”, “Antonio”-> “Antonio”))

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    #check order
    for item in backoff:
        text = regexes[item].sub(lambda m: " ", text)
        #text = regexes[item].sub(lambda m: " " + "_" + item.upper() + "_" + " ", text)
    text = regexes["normalize_elong"].sub(r'\1\1', text)
    text = re.compile(r'([!?.,;]){2,}', re.UNICODE).sub(r' \1\1 ', text)
    text = re.compile(r'([+∞¶°ç◊\(\)€$/:?.,;{~!"^`\[\]])', re.UNICODE).sub(r' \1 ', text)
    text = text.replace("»", " » ")
    text = text.replace("«", " « ")
     

    result_words = []

    for word in text.split():
        if len(word) > 26:
             norm_word = "__LONG-LONG__"
        elif word == "@user":
            norm_word = " "
        elif word == "URL":
            norm_word = " "
        elif word in token_syb:
            norm_word = token_syb[word]
        elif is_integer(word) or is_float(word):
             norm_word = normalize_numb(word)
        elif word[0].isupper():
            norm_word = word.capitalize()
        else:
            norm_word = word
        result_words.append(norm_word)
    return ' '.join(result_words)

def add_space_before_hashtag(text):
    return ' '.join(re.sub("#", " #", text).split())

############################
#        Hashtag
############################
from src.data.preprocessing.dicts.hashtags import know_hashtag_words, fixed_hashtags

def normalize_hashtags( text):
    """
    Normalizes hashtags into '#hashtag'.

    Example: '#iostoconsalvini' -> '#hashtag'

    Parameters
    ----------
    text : str

    Returns
    -------
    integer
    """
    return re.sub(r'#\S+', '#hashtag', text)


def split_hashtag(hashtag, delimiter = ("<",">")):
    """
    Splits the hashtags in the text into words and encloses them between < and >.

    Example: #iostoconsalvini -> < io sto con salvini >

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    
    #remove hashtag symb
    hashtag = re.sub("#","",hashtag)
    if hashtag.lower() not in know_hashtag_words:
        splitted_hashtag = ' '.join(segmenter.split(add_space_before_capital_words(hashtag)))
        if splitted_hashtag in fixed_hashtags:
            splitted_hashtag = fixed_hashtags[splitted_hashtag]
        return " "+delimiter[0]+" "+splitted_hashtag+" "+delimiter[1]+" "
    else:
        return " "+delimiter[0]+" "+hashtag+" "+delimiter[1]+" "

def get_splitted_hashtags(text):
    hashtags = re.findall(r'#\S+', add_space_before_hashtag(text))
    return {hashtag: split_hashtag(hashtag) for hashtag in hashtags}

def split_hashtags(text):
    hashtags = get_splitted_hashtags(text)
    return ' '.join([hashtags[word] if word in hashtags else word for word in add_space_before_hashtag(text).split()])



############################
#        Cleaning
############################
def clean_url(text):
    """
    Removes URLs from the text.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    return re.sub(r'URL', ' ', text)

def clean_tag(text):
    """
    Removes tag from the text.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    return re.sub(r'@user', ' ', text)

def clean_hashtag_symbol(text):
    """
    Removes # from the text.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    text = ' ' + text + ' '

    return re.sub(r'#', ' ', text)

def clean_censured_bad_words(text):
    """
    Uncensors the bad words.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    text = " " + text + " "
    text = re.sub(r' c[.x*@%#$^]+i ', ' coglioni ', text, flags=re.IGNORECASE)
    text = re.sub(r' c[.x*@%#$^]+e ', ' coglione ', text, flags=re.IGNORECASE)
    text = re.sub(r' c[.x*@%#$^]+o ', ' cazzo ', text, flags=re.IGNORECASE) 
    text = re.sub(r' c[.x*@%#$^]+i ', ' cazzi ', text, flags=re.IGNORECASE) 
    text = re.sub(r' m[.x*@%#$^]+a ', ' merda ', text, flags=re.IGNORECASE) 
    text = re.sub(r' m[.x*@%#$^]+e ', ' merde ', text, flags=re.IGNORECASE) 
    text = re.sub(r' c[.x*@%#$^]+ulo ', ' culo ', text, flags=re.IGNORECASE) 
    text = re.sub(r' p[.x*@%#$^]+a ', ' puttana ', text, flags=re.IGNORECASE)
    text = re.sub(r' p[.x*@%#$^]+e ', ' puttane ', text, flags=re.IGNORECASE)
    text = re.sub(r' t[.x*@%#$^]+a ', ' troia ', text, flags=re.IGNORECASE)
    text = re.sub(r' t[.x*@%#$^]+e ', ' troie ', text, flags=re.IGNORECASE)
    text = re.sub(r' s[.x*@%#$^]+o ', ' stronzo ', text, flags=re.IGNORECASE)
    text = re.sub(r' s[.x*@%#$^]+i ', ' stronzi ', text, flags=re.IGNORECASE)

    return text

def clean_laughs(text):
    """
    Removes laughs.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """

    laughs = ['ah', 'eh', 'he' 'ih', 'hi', 'oh'] 
    vowels = ['a', 'e', 'i', 'o', 'u']

    text_words = text.split()
    new_words  = [word for word in text_words if word.lower() not in laughs]
    
    new_text = ' '.join(new_words)
    
    for i in new_words:
        j = i.lower()
        for k in vowels:
            if ('h' in j) and (len(j) >= 4):
                if (len(j) - 2) <= (j.count(k) + j.count('h')):
                    new_text = new_text.replace(i, '')
    return new_text

def clean_vowels(text):
    """
    Removes nearby equal vowels.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    correct_words_vowels = ['coop', 'facebook', 'canaan', 'canaaniti', 'tweet', 'voodoo', 'book', 'isee', 'speech', 'woolfe',
                            'coffee', 'ffoo', 'refugees', 'google', 'shooting', 'hooligans', 'desiree', 'retweeted', 'microaree',
                            'keep']

    vowels = ['a', 'e', 'i', 'o', 'u']

    new_text = text
    words = text.split()
    
    for word in words:
        word_lower = word.lower()
        if word_lower not in unique_italian_words and word_lower not in correct_words_vowels:
            new_string = word[0]
            for i in range(1, len(word)):
                if word[i].lower() not in vowels:
                    new_string = new_string + word[i]
                else:
                    if(word[i].lower() != word[i-1].lower()):
                        new_string = new_string + word[i] 

            new_text = new_text.replace(word, new_string)

    return new_text

def clean_consonants(text):
    """
    Removes nearby equal consonants if they are more than 2.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """

    consonants = ['b','c','d','f','g','h','k','l','m','n','p','q','r','s','t','v','x','y','z']

    new_text = text
    words = text.split()
    
    for word in words:
        new_string = word[0]
        for i in range(1, len(word)):
            if word[i].lower() not in consonants:
                new_string = new_string + word[i]
            else:
                if(word[i].lower() != word[i-1].lower()):
                    new_string = new_string + word[i]
                elif i>=2 and (word[i].lower() != word[i-2].lower()):
                    new_string = new_string + word[i]

        new_text = new_text.replace(word, new_string)
    return new_text


############################
#        Replace
############################

def replace_abbreviation(text):
    """
    Replaces abbreviations.

    Example: 'cmq' -> 'comunque'

    Parameters
    ----------
    tokens : list
        list of strings.

    Returns
    -------
    list
        list of strings.
    """

    #new_tokens  = [abbr_word.get(ele, ele) for ele in tokens]
    result = []

    for word in text.split():
        if word.lower() in abbr_word:
            result.append(abbr_word[word.lower()])
        else:
            result.append(word)

    return ' '.join(result)

def replace_acronyms(text):
    """
    Replaces acronyms.

    Example: [..., 'onu', ...] -> [..., 'organizzazione', 'nazioni', 'unite', ...]

    Parameters
    ----------
    tokens : list
        list of strings.

    Returns
    -------
    list
        list of strings.
    """


    results = [' '.join(acronyms[word.lower()]) if word.lower() in acronyms else word for word in text.split()]
    return ' '.join(results)