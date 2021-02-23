import pandas as pd
import csv
import nltk
import re
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

import wordninja

from nltk.stem import SnowballStemmer

import ast

import emoji
import unicodedata

import gzip

import spacy_udpipe
import spacy
import language_tool_python


spacy_udpipe.download("it")

text_processor = TextPreProcessor(
    fix_html=True,  # fix HTML tokens
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=False).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

missing_emoji = {}
for key in emoji.UNICODE_EMOJI_ENGLISH:
    if key not in emoji.UNICODE_EMOJI_ITALIAN:
        missing_emoji[key.encode('unicode-escape').decode("latin_1")] = key
print(len(missing_emoji))
print(len(emoji.UNICODE_EMOJI_ITALIAN))
decoded_unicode_ita = {k.encode('unicode-escape').decode("latin_1"): emoji.UNICODE_EMOJI_ITALIAN[k] for k in emoji.UNICODE_EMOJI_ITALIAN}

missing_emoji_tralation = {}

EMOJI_PATTERN = re.compile(
    "(["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "])"
)
    
def add_space_between_emojies(text):
    text = re.sub(EMOJI_PATTERN, r' \1 ', text)
    return text

for elem in missing_emoji:
    for k in decoded_unicode_ita:
        if elem in k:
            missing_emoji_tralation[missing_emoji[elem]] = decoded_unicode_ita[k].replace(':','').replace('_', ' ')
            break

class Preprocessing:
    def __init__(self, italian_words, italian_words_gz, bad_words, word_polarity_dict, text_processor = text_processor, 
                 stemmer = SnowballStemmer('italian')):

        self.__italian_words = italian_words
        self.__lm = wordninja.LanguageModel(italian_words_gz)
        self.__bad_words = bad_words
        self.__word_polarity_dict = word_polarity_dict
        self.__text_processor = text_processor
        self.__stemmer = stemmer
        self.__nlp = spacy_udpipe.load("it")

    def __clean_url(self, text):
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

    def __clean_tag(self, text):
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

    def __translate_emoticons(self, text):
        """
        Converts emoticons:

        Example: :-) -> <happy>

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """
        return " ".join(text_processor.pre_process_doc(text))

    def __replace_symbol_double_dots(self, text):
        """
        Converts ':' into 'double_dots'.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """
        return re.sub(r':', ' double_dots ', text)

    def __translate_emoji(self, text):
        """
        Translates the emojis into text.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """

        text_result = emoji.demojize(text, language='it', delimiters=(" ", " "))
        text_result = add_space_between_emojies(text_result)
        text_result = text_result.split()
        text_result = [elem.replace(':','').replace('_', ' ') if ":"+elem+":" in emoji.EMOJI_UNICODE_ITALIAN else elem for elem in text_result]
        text_result = [ missing_emoji_tralation[word] if word in missing_emoji_tralation else word for word in text_result]
        return ' '.join(text_result)

    def __remove_double_dots(self, text):
        """
        Removes ':' from the text.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """
        return re.sub(r':', ' ', text)

    def __replace_word_double_dots(self, text):
        """
        Converts 'double_dots' into ':'.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """
        return re.sub(r'double_dots', ' : ', text)

    def __add_space_before_hashtag(self, text): ##io#vado -> # #io #vado
        """
        Translates the emoticons into text.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """
        words = text.split()
        newwords = []
        for word in words:
            for i in range(0, len(word)):
                if i != 0:
                    if word[i] == '#':
                        word = self.__add_space_before_hashtag(word[:i]) + ' ' + self.__add_space_before_hashtag(word[i:])
            newwords.append(word)

        return ' '.join(newwords)

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
        result = re.findall(r'#\S+', text)

        return len(result)

    def __normalize_hashtags(self, text):
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

    def __add_space(self, text):
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
        words = text.split()
        newwords = []

        for word in words:
            for i in range(0, len(word)):
                if i != len(word)-1 and word[i] != ' ':
                    if word[i].islower() and word[i+1].isupper():
                        word = word[:i+1] + ' ' + word[i+1:]
            newwords.append(word)

        return ' '.join(newwords)

    def __replace_hashtags(self, text):
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
        hashtag_words = ['lidl', 'roma', 'caritas', 'syria', 'isis', 'm5s', 'apatridi', 'brexit', 'sinta', 'msna', 'yacme',
                 'ckan', 'dimartedi', 'karak', 'cojoni', 'uae', 'scampia', 'onsci', 'hamas', 'ncd', 'olbes', 'fdian',
                 'acquarius', 'aquarius', 'macron', 'barbarians', 'kyenge', 'kienge', 'mef', 'muslim', 'error', 'soros',
                 'italexit', 'sprar', 'ahvaz', 'nsa', 'enez', 'daspo', 'cpr', 'desire', 'boldrina', 'msf', 'belgium',
                 'piddino', 'piddina', 'fdi', 'zarzis', 'eliminiamolo', 'strasbourg', 'isee', 'sophia', 'unit', 'oeshh',
                 'porrajmos', 'dibba', 'ciociaria', 'cie', 'junker', 'is', 'syriza', 'linate', 'raqqa', 'ama', 'cesedi',
                 'aicds', 'heidelberg', 'ffoo', 'cvd', 'forex', 'docufilm', 'reyn', 'hooligans', 'anpal', 'rdc', 'rohingya',
                 'nwo', 'def', 'cattivisti', 'vauro', 'sorosiane', 'libya', 'censis']
    
        text = ' ' + text + ' '
        result = re.findall(r'#\S+', text)
        
        for word in result:
            new_word = '< '
            if word[1:].lower() not in hashtag_words:
                spaced_word = self.__add_space(word)
                splitted = self.__lm.split(spaced_word)
                
                for i in range(0, len(splitted)):
                    if i == 0:
                        new_word = new_word + splitted[i]
                    else:
                        new_word = new_word + ' ' + splitted[i]
            else:
                new_word = new_word + word[1:]
            new_word = new_word + ' >'
            
            text = text.replace(word, new_word)
            
        return text

    def __hashtag_fix(self, text):
        """
        Fixes the wrong hashtags splitting. 

        Example: 'in comin g' -> 'incoming'

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """
        fixed_hashtags = {'je sui s Charlie':'je suis Charlie', 'libere t cogita n s':'libere t cogitans', 'm a g g i o':'maggio', 
                  'ha te speech':'hate speech', 'w ed din g tour i sm':'wedding tourism', 'in comin g':'incoming',
                  'dimarte d':'dimarted', 'dall a vostra':'dalla vostra', 'Trattati Rom a':'Trattati Roma',
                  'I u venta':'Iuventa', 'Woolf e':'Woolfe', 'Attuali t':'Attualit', 'Morni ng':'Morning',
                  'Fort Lau derda l e':'Fort Lauderdale', 'mi gran ts':'migrants', 'a p r i l e':'aprile',
                  'E n e r g y':'Energy', 'I g les':'Igles', 'Christ mas':'Christmas', 'Sud Tiro l':'Sud Tirol',
                  'Paler m':'Palerm', 'esma red ze po va':'esma redzepova', 'gip sy':'gipsy', 'auster it':'austerit',
                  'immigrati on ban':'immigration ban', 'Financial Time s':'Financial Times', 'metro rom a':'metro roma',
                  'Su ed deutsch e Ze i tung':'Sueddeutsche Zeitung', 'porta aporta':'porta a porta', 'terro r':'terror',
                  'immi gran ts':'immigrants', 'giornata dell a memoria':'giornata della memoria', 'm a r z o':'marzo',
                  'dusse l dor f':'dusseldorf', 'riscopriamo l i':'riscopriamoli', 'ultimo ra':'ultima ora',
                  'Cercate l i':'Cercateli', 'Islam op ho bia':'Islamophobia', 'd i c e m b r e':'dicembre',
                  'g e n n a i o':'gennaio', 'f e b b r a i o':'febbraio', 'g i u g n o':'giugno', 'l u g l i o':'luglio',
                  'a g o s t o':'agosto', 's e t t e m b r e':'settembre', 'o t t o b r e':'ottobre',
                  'n o v e m b r e':'novembre', 'ta gad al a 7':'tagada la 7', 'Coffe e break l a 7':'Coffee break la 7',
                  'A S Rom a':'Associazione Sportiva Roma', 'Stadio Dell a Rom a':'Stadio Della Roma',
                  'best songo f movie':'best song of movie', 'un avita in metro':'una vita in metro', 'l e iene':'le iene',
                  'l a zanzara':'la zanzara', 'charlie h e b do':'charlie hebdo', 'Loo king':'Looking',
                  'Rom a Non Alza Muri':'Roma Non Alza Muri', 'r e fu gee s':'refugees', 'non una dime mo':'non una di meno',
                  'i t a l i a':'italia', 'r ss':'rss', 'at tack':'attack', 'a t t u a l i t a':'attualità',
                  'no tengo din ero':'no tengo dinero', 'Hi jr i':'Hijri', 'Asl i Erdogan':'Asli Erdogan',
                  'i si s fu c k you':'isis fuck you', 'L a Ru spetta':'La Ruspetta', 'Go o g l e Aler ts':'Google Alerts',
                  'A mi s':'Amis', 'Bal on Mundial':'Balon Mundial', 'l i dl':'lidl', 'racis m':'racism ',
                  'C y ber':'Cyber', 'Sili con Valle y':'Silicon Valley', 'Medio Campi dan':'Medio Campidan',
                  'w e l comere fu gee s':'welcome refugees', 'Ta gada':'Tagada', 'Grande sy n the':'Grande synthe',
                  'contentivo i':'contenti voi', 'at tack s':'attacks', 'REY N':'REYN', 'Marine L e Pen':'Marine Le Pen',
                  't ru e story':'true story', 'brex it':'brexit', 'delledonne':'delle donne', 'fu c kis i s':'fuck isis',
                  'islam i s the pro ble m':'islam is the problem', 'l a gabbia':'la gabbia', 'fu c k islam':'fuck islam',
                  'fu c k':'fuck', 'fu c k musli ms':'fuck muslims', 'sapeva telo':'sapevatelo', 'R in y':'Riny',
                  'A f g han Con f':'Afghan Con f', 'Ch al i e H e b do':'Chelie Hebdo', 's k y':'sky', 'H e b do':'Hebdo',
                  'S ho ot in g':'Shooting', 'Islam i c State':'Islamic State', 'l a 7':'la 7', 'Daisy Os akue':'Daisy Osakue',
                  'laria che tira':'l aria che tira', 'i phon e X S max':'iphone XSmax', 'L e Ga':'lega',
                  'laria che tirala':'l aria che tira la', 'Casa po un d':'Casa pound', 'R M C NEWS':'RMC NEWS',
                  'm i l i o n i':'milioni', 'un altro cucchia in odi merda':'un altro cucchiaino di merda',
                  'omnibus l a 7':'omnibus la 7', 'job sa c t':'jobs act', 'Mi grati on':'Migration',
                  'Movi men t Onesti':'Moviment Onesti', 'none larena':'non è l arena' ,'Non Un ad i Meno':'Non Una di Meno',
                  'Fil c ams Collettiva':'Filcams Collettiva', 'Time s':'Times', 'ci vi l t allo sbando':'civiltà allo sbando',
                  'Am ne st y International':'Amnesty International', 'C H I U D E T E':'CHIUDETE', 'Open Arm s':'Open Arms',
                  'Gilet s J a un e s':'Gilets Jaunes', 'Mi grant I':'MigrantI', 'Horst Se e hofer':'Horst Seehofer',
                  '5s':'cinque stelle', 'rd c':'rdc', 'piazza delpopolo':'piazza del popolo'}

        for word in fixed_hashtags:
            text = re.sub(re.escape(word), fixed_hashtags[word], text, flags=re.IGNORECASE)

        return text

    def __normalize_numbers(self, text):
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
        try:
            val = int(text)
        except:
            text = re.sub(r'\d', '@Dg', text)
            return text
        if val >= 0 and val < 2100:
            return str(val)
        else:
            return "DIGLEN_" + str(len(str(val)))

    def __clean_some_punctuation(self, text):
        """
        Removes some punctuation.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """
        text = ' ' + text + ' '
        text = re.sub(r'\\n', '. ', text)
        text = re.sub(r'\\', ' ', text)
        text = re.sub(r'/', ' ', text)
        return re.sub(r'_', ' ', text) 

    def __clean_emoticon_text(self, text):
        """
        Translates the emoticons written as <emoticon> into text.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """
        emoticons_text = {
            '<kiss>': 'bacio',
            '<happy>': 'felice',
            '<laugh>': 'risata',
            '<sad>': 'triste',
            '<surprise>': 'sorpreso',
            '<wink>': 'occhiolino',
            '<tong>': 'faccia con lingua',
            '<annoyed>': 'annoiato',
            '<seallips>': 'labbra sigillate',
            '<angel>': 'angelo',
            '<devil>': 'diavolo',
            '<highfive>' : 'batti il cinque',
            '<heart>': 'cuore',
            '<user>' : 'persona'
            }

        text_words = text.split()
        new_words  = [emoticons_text.get(ele, ele) for ele in text_words]

        return ' '.join(new_words)

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

    def __normalize_text(self, text):
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
        words = text.split()
        result_words = []
        
        for word in words:
            if len(word) > 26:
                return "__LONG-LONG__"
            new_word = self.__normalize_numbers(word)
            if new_word != word:
                word = new_word
            if word[0].isupper():
                word = word.capitalize()
            else:
                word = word.lower()
            result_words.append(word)
            
        return ' '.join(result_words)

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

    def __clean_censured_bad_words(self, text):
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

    def __clean_hashtag_symbol(self, text):
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

    def __clean_laughs(self, text):
        """
        Removes laughs.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """

        laughs = ['ah', 'eh', 'he' 'ih', 'hi'] 
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

    def __clean_vowels(self, text):
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
            if word.lower() not in self.__italian_words and word.lower() not in correct_words_vowels:
                new_string = word[0]
                for i in range(1, len(word)):
                    if word[i].lower() not in vowels:
                        new_string = new_string + word[i]
                    else:
                        if(word[i].lower() != word[i-1].lower()):
                            new_string = new_string + word[i] 

                new_text = new_text.replace(word, new_string)
            
        return new_text

    def __clean_consonants(self, text):
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

    def __stick_apostrophe_text(self, text):
        """
        Attaches the apostrophe to the word preceding it.

        Example: "l ' uomo." -> "l' uomo."

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """
        text = re.sub(r" ’", "’", text)
        return re.sub(r" '", "'", text)

    def __lemma(self, text):
        """
        Returns the lemmas of the words in the text.

        Example: 'Io mangio una mela.' -> ['Io', 'mangiare', 'una', 'mela', '.']

        Parameters
        ----------
        text : str

        Returns
        -------
        list
            list of lemmas.
        """
        lemmas = []
    
        doc = self.__nlp(text)
        
        for token in doc:
            lemmas.append(token.lemma_)
            
        return lemmas

    def __pos(self, text):
        """
        Returns the PoS of the words in the text.

        Example: 'Io mangio una mela.' -> ['PRON', 'VERB', 'DET', 'NOUN', 'PUNCT']

        Parameters
        ----------
        text : str

        Returns
        -------
        list
            list of PoS.
        """
        pos_list = []
        
        doc = self.__nlp(text)
        
        for token in doc:
            pos_list.append(token.pos_)
            
        return pos_list

    def __dep(self, text):
        """
        Returns the dep of the words in the text.

        Example: 'Io mangio una mela.' -> ['nsubj', 'ROOT', 'det', 'obj', 'punct']

        Parameters
        ----------
        text : str

        Returns
        -------
        list
            list of dep.
        """
        dep_list = []
    
        doc = self.__nlp(text)
        
        for token in doc:
            dep_list.append(token.dep_)
            
        return dep_list

    def __get_word_polarity(self, lemmas):
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
        
        for word in lemmas:
            if word in self.__word_polarity_dict:
                polarity.append(self.__word_polarity_dict[word])
            else:
                polarity.append('neutral')
                
        return polarity

    def __tokenization(self, text):
        """
        Tokenizes the input string.

        Example: 'Io sto con Salvini' -> ['Io', 'sto', 'con', 'Salvini']

        Parameters
        ----------
        text : str

        Returns
        -------
        list
            list of strings.
        """
        tknzr=SocialTokenizer(lowercase=False)

        return tknzr.tokenize(text)

    def __stick_apostrophe(self, tokens):
        """
        Attaches the apostrophe to the token preceding it, 
        if it is in the 'pre_char' list. It then deletes the token containing only the apostrophe.

        Example: ["l", "'", "uomo"] -> ["l'", "uomo"]

        Parameters
        ----------
        tokens : list
            list of strings.

        Returns
        -------
        list
            list of strings.
        """

        apostrophes = ["'", "’"]

        result_tokens = []
        pos = 0
        while pos < len(tokens):
            if pos !=(len(tokens)-1) and tokens[pos+1] in apostrophes:
                result_tokens.append(tokens[pos] + '\'')
                pos+=2
            else:
                result_tokens.append(tokens[pos])
                pos+=1

        # pre_char = ['l', 'un', 'dell', 'all', 'dall', 'nell', 'sull', 'c', 'n']

        # to_pop = []
        # for i in range(0, len(tokens)-1):
        #     if tokens[i].lower() in pre_char and tokens[i+1] in apostrophes:
        #         tokens[i] = tokens[i] + "'"
        #         to_pop.append(i+1)
        
        # result_tokens = []
        # for i in range(0, len(tokens)):
        #     if i not in to_pop:
        #         result_tokens.append(tokens[i])  
            
        return result_tokens

    def __replace_abbreviation(self, tokens):
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
        abbr_word = {'cmq':'comunque', 'gov':'governatori', 'fb':'facebook', 'tw':'twitter', 'juve':'juventus', 'ing':'ingegnere', 
             'sx':'sinistra', 'qdo':'quando', 'rep':'repubblica', 'grz':'grazie', 'ita':'italia', 'mln':'milioni', 
             'mld':'miliardi', 'pke':'perche', 'anke':'anche', 'cm':'come', 'dlla':'della', 'dlle':'delle', 'qst':'questa',
             'ke':'che', 'nn':'non', 'sn':'sono', 'cn':'con', 'xk':'perche', 'xke':'perche', 'art':'articolo',
             'tv':'televisore', '€':'euro', 'xché':'perché', 'xké':'perché', 'pkè':'perché'} 

        #new_tokens  = [abbr_word.get(ele, ele) for ele in tokens]
        result = []

        for word in tokens:
            if word.lower() in abbr_word:
                result.append(abbr_word[word.lower()])
            else:
                result.append(word)

        return result

    def __replace_acronyms(self, tokens):
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
        acronyms = {'unhcr':['alto', 'commissariato', 'nazioni', 'unite', 'rifugiati'], 
            'onu':['organizzazione', 'delle', 'nazioni', 'unite'],
            'fdi':['fratelli', 'italia'], 
            'msna':['minori', 'stranieri', 'accompagnati'], 
            'rdc':['reddito', 'di', 'cittadinanza'],
            'gus':['gruppo', 'umana', 'solidarieta'], 
            'sprar':['sistema', 'protezione', 'richiedenti', 'asilo'],
            'anpi':['associazione', 'nazionale', 'partigiani', 'italia'], 
            'anac':['autorita', 'nazionale', 'anticorruzione'],
            'lgbt':['lesbiche', 'gay', 'bisessuali', 'transgender'], 
            'ln':['lega', 'nord'], 
            'ue':['unione', 'europea'],
            'msf':['medici','senza','frontiere'], 
            'ispi':['istituto','studi','politica','internazionale'],
            'cpr':['centri','permanenza','rimpatri'], 
            'pd':['partito', 'democratico'], 
            'gc':['guardia', 'costiera'],
            'inps':['istituto','nazionale','previdenza','sociale'],
            'cdm':['consiglio', 'dei', 'ministri'], 
            'pdl':['popolo', 'della', 'liberta'], 
            'atac':['azienda', 'tramvie', 'autobus', 'comune', 'roma'],
            'tav':['treno', 'alta', 'velocita'], 
            'isee':['situazione', 'economica', 'equivalente'],
            'usa':['stati', 'uniti', 'd', 'america'], 
            'onlus':['organizzazione', 'lucrativa', 'utilita', 'sociale'],
            'acsim':['associazione', 'centro', 'servizi', 'immigrati', 'marche'], 
            'aids':['sindrome', 'immuno', 'deficienza', 'acquisita'], 
            'eu':['unione', 'europea'],
            'ong':['organizzazione', 'governativa'], 
            'nwo':['nuovo', 'ordine', 'mondiale'],
            'pil':['prodotto', 'interno', 'lordo'], 
            'cgil':['confederazione', 'generale', 'lavoro'],
            'cdt':['corriere', 'ticino'], 
            'ptv':['societa', 'televisiva', 'pakistan'],
            'syriza':['coalizione', 'sinistra', 'radicale'], 
            'fiom':['federazione', 'impiegati', 'operai', 'metallurgici'],
            'lgbtq':['lesbiche', 'gay', 'bisessuali', 'transgender', 'queer'], 
            'rpl':['radio', 'padania', 'libera'],
            'arci':['associazione', 'ricreativa', 'culturale', 'italiana'],
            'ofcs':['osservatorio', 'focus', 'cultura', 'sicurezza'],
            'm5s':['movimento', 'cinque', 'stelle'],
            'wm5s':['movimento', 'cinque', 'stelle'],
            'mef':['ministero', 'dell', 'economia', 'e', 'delle', 'finanze'],
            'cnel':['consiglio', 'nazionale', 'dell', 'economia', 'e', 'del', 'lavoro'],
            'fdian':['fratelli', 'di', 'italia', 'alleanza', 'nazionale'],
            'ecm':['educazione', 'continua', 'in', 'medicina'],
            'cie':['carta', 'di', 'identità', 'elettronica'],
            'tg':['telegiornale'],
            'rai':['radiotelevisione', 'italiana'],
            'anpal':['agenzia', 'nazionale', 'politiche', 'attive', 'lavoro'],
            'def':['documento', 'di', 'economia', 'e', 'finanza'],
            'cr':['consiglio', 'regionale'],
            'ama':['azienda', 'municipale', 'ambiente'],
            'cesedi':['centro', 'servizi', 'didattici'],
            'ffoo':['forze', 'dell', 'ordine'],
            'reyn':['rete', 'per', 'la', 'prima', 'infanzia', 'rom'],
            'rmc':['radio', 'monte', 'carlo'],
            'ddl':['disegno', 'di', 'legge']}

        for i in range(0, len(tokens)):
            word = tokens[i]
            if word.lower() in acronyms:
                tokens[i] = acronyms[word.lower()][0]
                if len(acronyms[word.lower()]) > 1:
                    for j in range(1, len(acronyms[word.lower()])):
                        tokens.insert(i+j, acronyms[word.lower()][j])

        return tokens

    def __replace_others_emojis(self, tokens):
        """
        Translates the emojis into text.

        Parameters
        ----------
        tokens : list
            list of strings.

        Returns
        -------
        list
            list of strings.
        """
        symbols = {'✔':['segno', 'di', 'spunta'],
                   '♻':['simbolo', 'del', 'riciclaggio'],
                   '▶':['pulsante', 'di', 'riproduzione'],
                   '🖊':['penna', 'a', 'sfera'],
                   '❤':['cuore', 'rosso']}

        for i in range(0, len(tokens)):
            word = tokens[i]
            if word in symbols:
                tokens[i] = symbols[word][0]
                if len(symbols[word]) > 1:
                    for j in range(1, len(symbols[word])):
                        tokens.insert(i+j, symbols[word][j])
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

    def preprocess_df(self, df, column_name, path, stemming=False):
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
        raw_cl = "raw"+column_name
        df[raw_cl] = df[column_name]

        #Removing URLs
        print("Removing URLs")
        df[column_name] = df[column_name].apply(self.__clean_url)
        
        #Removing Tags
        print("Removing Tags")
        df[column_name] = df[column_name].apply(self.__clean_tag)
        
        #Support dataframe
        df_app = df.copy()

        #Feature extraction: length of the comment
        print("Feature extraction: length of the comment")
        df['text_length'] = df[column_name].apply(self.__text_length)

        #Translation of emoji
        print("Translation of emoji")
        df[column_name] = df[column_name].apply(self.__translate_emoji)
        
        #Translating emoticons
        print("Translating emoticons")
        df[column_name] = df[column_name].apply(self.__translate_emoticons)
        
        #Adding space before hashtag symbol '#'
        print("Adding space before hashtag symbol '#'")
        df[column_name] = df[column_name].apply(self.__add_space_before_hashtag)
        
        #Feature extraction: number of hashtags
        print("Feature extraction: number of hashtags")
        df['hashtags'] = df[column_name].apply(self.__count_hashtags)

        #Replacing hashtags
        print("Replacing hashtags")
        df[column_name] = df[column_name].apply(self.__replace_hashtags)
        
        #Normalizing hashtags
        print("Normalizing hashtags")
        df_app[column_name] = df_app[column_name].apply(self.__normalize_hashtags)
        
        #Fixing hashtags
        print("Fixing hashtags")
        df[column_name] = df[column_name].apply(self.__hashtag_fix)
        
        #Normalizing numbers
        print("Normalizing numbers")
        df[column_name] = df[column_name].apply(self.__normalize_numbers)
        
        #Removing _, \\n, \\ and /
        print("clean_some_punctuation")
        df[column_name] = df[column_name].apply(self.__clean_some_punctuation)
        
        #Adding space between lowercase and uppercase
        print("Adding space between lowercase and uppercase")
        df[column_name] = df[column_name].apply(self.__add_space)
        
        #Converting all emoticons written in text
        print("Converting all emoticons written in text")
        df[column_name] = df[column_name].apply(self.__clean_emoticon_text)
        
        #Feature extraction: percentage of words written in CAPS-LOCK
        print("Feature extraction: percentage of words written in CAPS-LOCK")
        df['%CAPS-LOCK words'] = df[column_name].apply(self.__caps_lock_words)
        
        #Normalizing text
        print("Normalizing text")
        df[column_name] = df[column_name].apply(self.__normalize_text)
        
        #Feature extraction: number of ‘!’ inside the text
        print("Feature extraction: esclamations")
        df['esclamations'] = df[column_name].apply(self.__esclamations)

        #Feature extraction: number of ‘?’ inside the text
        print("Feature extraction: number of questions mark")
        df['questions'] = df[column_name].apply(self.__questions)

        #Uncensoring the bad words
        print("Uncensoring the bad words")
        df[column_name] = df[column_name].apply(self.__clean_censured_bad_words)
        
        #Removing hashtag symbol '#'
        print("Removing hashtag symbol")
        df[column_name] = df[column_name].apply(self.__clean_hashtag_symbol)
        
        #Removing laughs
        print("Removing laughs")
        df[column_name] = df[column_name].apply(self.__clean_laughs)
        
        #Removing nearby equal vowels
        print("Removing nearby equal vowels")
        df[column_name] = df[column_name].apply(self.__clean_vowels)
        
        #Removing nearby equal consonants if they are more than 2
        print("Removing nearby equal consonants if they are more than 2")
        df[column_name] = df[column_name].apply(self.__clean_consonants)
        
        #Sticking the apostrophe (text)
        print("Sticking the apostrophe (text)")
        df[column_name] = df[column_name].apply(self.__stick_apostrophe_text)
        
        #Lemma
        print("generate Lemma")
        df['lemma'] = df[column_name].apply(self.__lemma)

        #PoS
        print("genererate PoS")
        df['pos'] = df_app[column_name].apply(self.__pos)

        #Dep
        print("Generate Dep")
        df['dep'] = df_app[column_name].apply(self.__dep)

        #Word polarity
        print("Generate Word polarity")
        df['word_polarity'] = df['lemma'].apply(self.__get_word_polarity)

        #Tokenization
        print("Tokenization")
        df['tokens'] = df[column_name].apply(self.__tokenization)

        #Sticking the apostrophe (tokens)
        print("Sticking the apostrophe (tokens)")
        df['tokens'] = df['tokens'].apply(self.__stick_apostrophe)

        #Stemming
        print("Generate Stemming")
        df['stem'] = df['tokens'].apply(self.__stemming)

        #Replacement of the abbreviations with the respective words
        print("Replacement of the abbreviations with the respective words")
        df['tokens'] = df['tokens'].apply(self.__replace_abbreviation)

        #Replacing Acronyms
        print("Replacing Acronyms")
        df['tokens'] = df['tokens'].apply(self.__replace_acronyms)

        #Replacing other emojis
        #df['tokens'] = df['tokens'].apply(self.__replace_others_emojis)

        #Feature extraction: percentage of Bad Words
        print("Feature extraction: percentage of Bad Words")
        df['%bad_words'] = df['tokens'].apply(self.__percentage_bad_words)

        #Saving the preprocessed dataframe
        self.__save_df_csv(df.copy(), path)
        # df.to_csv(path, index=False)

    def preprocess_str(self, text, hashtags=True, stemming=False):
        """
        Preprocesses the string.

        Parameters
        ----------
        text : str
            String to be preprocessed.

        hashtags : bool
            If False, the function removes the hashtag tags.

        stemming : bool
            If True, the stemming is applied.
        """

        #Removing URLs
        text = self.__clean_url(text)

        #Removing Tags
        text = self.__clean_tag(text)

        #Translating emoticons
        text = self.__translate_emoticons(text)

        #Replacing ':' into 'double_dots'
        text = self.__replace_symbol_double_dots(text)

        #Translation of emoji
        text = self.__translate_emoji(text)

        #Removing ':'
        text = self.__remove_double_dots(text)

        #Adding space before hashtag symbol '#'
        text = self.__add_space_before_hashtag(text)

        #Replacing hashtags
        text = self.__replace_hashtags(text)

        #Fixing hashtags
        text = self.__hashtag_fix(text)

        #Normalizing numbers
        text = self.__normalize_numbers(text)

        #Removing _, \\n, \\ and /
        text = self.__clean_some_punctuation(text)
        
        #Adding space between lowercase and uppercase
        text = self.__add_space(text)

        #Converting all emoticons written in text
        text = self.__clean_emoticon_text(text)

        #Normalizing text
        text = self.__normalize_text(text)

        #Uncensoring the bad words
        text = self.__clean_censured_bad_words(text)

        #Removing hashtag symbol '#'
        text = self.__clean_hashtag_symbol(text)

        #Removing laughs
        text = self.__clean_laughs(text)

        #Removing nearby equal vowels
        text = self.__clean_vowels(text)

        #Removing nearby equal consonants if they are more than 2
        text = self.__clean_consonants(text)

        #Sticking the apostrophe (text)
        text = self.__stick_apostrophe_text(text)

        #Tokenization
        text = self.__tokenization(text)

        #Sticking the apostrophe
        text = self.__stick_apostrophe(text)

        #Replacement of the abbreviations with the respective words
        text = self.__replace_abbreviation(text)

        #Replacing Acronyms
        text = self.__replace_acronyms(text)

        #Replacing other emojis
        text = self.__replace_others_emojis(text)

        return text

    def __save_df_csv(self, df, path):
        list_features = ["lemma", "pos", "dep", "word_polarity", "tokens", "stem"]
        
        for feature in list_features:
            df[feature] = df[feature].apply(lambda s : ' '.join([str(elem) for elem in s]))
        df.to_csv(path, sep="\t", index=False)