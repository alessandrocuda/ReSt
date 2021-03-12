import re

# todo:catch repeating parenthesis
emoticons = {
    ':*': 'bacio',
    ':-*': 'bacio',
    ':x': 'bacio',
    ':-)': 'felice',
    ':-))': 'felice',
    ':-)))': 'felice',
    ':-))))': 'felice',
    ':-)))))': 'felice',
    ':-))))))': 'felice',
    ':)': 'felice',
    ':))': 'felice',
    ':)))': 'felice',
    ':))))': 'felice',
    ':)))))': 'felice',
    ':))))))': 'felice',
    ':)))))))': 'felice',
    ':o)': 'felice',
    ':]': 'felice',
    ':3': 'felice',
    ':c)': 'felice',
    ':>': 'felice',
    '=]': 'felice',
    '8)': 'felice',
    '=)': 'felice',
    ':}': 'felice',
    ':^)': 'felice',
    '|;-)': 'felice',
    ":'-)": 'felice',
    ":')": 'felice',
    '\o/': 'felice',
    '*\\0/*': 'felice',
    ':-D': 'risata',
    ':D': 'risata',
    # '(\':': 'risata',
    '8-D': 'risata',
    '8D': 'risata',
    'x-D': 'risata',
    'xD': 'risata',
    'X-D': 'risata',
    'XD': 'risata',
    '=-D': 'risata',
    '=D': 'risata',
    '=-3': 'risata',
    '=3': 'risata',
    'B^D': 'risata',
    '>:[': 'triste',
    ':-(': 'triste',
    ':-((': 'triste',
    ':-(((': 'triste',
    ':-((((': 'triste',
    ':-(((((': 'triste',
    ':-((((((': 'triste',
    ':-(((((((': 'triste',
    ':(': 'triste',
    ':((': 'triste',
    ':(((': 'triste',
    ':((((': 'triste',
    ':(((((': 'triste',
    ':((((((': 'triste',
    ':(((((((': 'triste',
    ':((((((((': 'triste',
    ':-c': 'triste',
    ':c': 'triste',
    ':-<': 'triste',
    ':<': 'triste',
    ':-[': 'triste',
    ':[': 'triste',
    ':{': 'triste',
    ':-||': 'triste',
    ':@': 'triste',
    ":'-(": 'triste',
    ":'(": 'triste',
    'D:<': 'triste',
    'D:': 'triste',
    'D8': 'triste',
    'D;': 'triste',
    'D=': 'triste',
    'DX': 'triste',
    'v.v': 'triste',
    "D-':": 'triste',
    '(>_<)': 'triste',
    ':|': 'triste',
    '>:O': 'sorpreso',
    ':-O': 'sorpreso',
    ':-o': 'sorpreso',
    ':O': 'sorpreso',
    '°o°': 'sorpreso',
    'o_O': 'sorpreso',
    'o_0': 'sorpreso',
    'o.O': 'sorpreso',
    'o-o': 'sorpreso',
    '8-0': 'sorpreso',
    '|-O': 'sorpreso',
    ';-)': 'occhiolino',
    ';)': 'occhiolino',
    '*-)': 'occhiolino',
    '*)': 'occhiolino',
    ';-]': 'occhiolino',
    ';]': 'occhiolino',
    ';D': 'occhiolino',
    ';^)': 'occhiolino',
    ':-,': 'occhiolino',
    '>:P': 'faccia con lingua',
    ':-P': 'faccia con lingua',
    ':P': 'faccia con lingua',
    'X-P': 'faccia con lingua',
    'x-p': 'faccia con lingua',
    'xp': 'faccia con lingua',
    'XP': 'faccia con lingua',
    ':-p': 'faccia con lingua',
    ':p': 'faccia con lingua',
    '=p': 'faccia con lingua',
    ':-Þ': 'faccia con lingua',
    ':Þ': 'faccia con lingua',
    ':-b': 'faccia con lingua',
    ':b': 'faccia con lingua',
    ':-&': 'faccia con lingua',
    '>:\\': 'annoiato',
    '>:/': 'annoiato',
    ':-/': 'annoiato',
    ':-.': 'annoiato',
    ':/': 'annoiato',
    ':\\': 'annoiato',
    '=/': 'annoiato',
    '=\\': 'annoiato',
    ':L': 'annoiato',
    '=L': 'annoiato',
    ':S': 'annoiato',
    '>.<': 'annoiato',
    ':-|': 'annoiato',
    '<:-|': 'annoiato',
    ':-X': 'labbra sigillate',
    ':X': 'labbra sigillate',
    ':-#': 'labbra sigillate',
    ':#': 'labbra sigillate',
    'O:-)': 'angelo',
    '0:-3': 'angelo',
    '0:3': 'angelo',
    '0:-)': 'angelo',
    '0:)': 'angelo',
    '0;^)': 'angelo',
    '>:)': 'diavolo',
    '>:D': 'diavolo',
    '>:-D': 'diavolo',
    '>;)': 'diavolo',
    '>:-)': 'diavolo',
    '}:-)': 'diavolo',
    '}:)': 'diavolo',
    '3:-)': 'diavolo',
    '3:)': 'diavolo',
    'o/\o': 'batti il cinque',
    '^5': 'batti il cinque',
    '>_>^': 'batti il cinque',
    '^<_<': 'batti il cinque',  # todo:fix tokenizer - MISSES THIS
    '<3': 'cuore'
}

# todo: clear this mess
pattern = re.compile("^[:=\*\-\(\)\[\]x0oO\#\<\>8\\.\'|\{\}\@]+$")
mirror_emoticons = {}
for exp, tag in emoticons.items():
    if pattern.match(exp) \
            and any(ext in exp for ext in [";", ":", "="]) \
            and not any(ext in exp for ext in ["L", "D", "p", "P", "3"]):
        mirror = exp[::-1]

        if "{" in mirror:
            mirror = mirror.replace("{", "}")
        elif "}" in mirror:
            mirror = mirror.replace("}", "{")

        if "(" in mirror:
            mirror = mirror.replace("(", ")")
        elif ")" in mirror:
            mirror = mirror.replace(")", "(")

        if "<" in mirror:
            mirror = mirror.replace("<", ">")
        elif ">" in mirror:
            mirror = mirror.replace(">", "<")

        if "[" in mirror:
            mirror = mirror.replace("[", "]")
        elif "]" in mirror:
            mirror = mirror.replace("]", "[")

        if "\\" in mirror:
            mirror = mirror.replace("\\", "/")
        elif "/" in mirror:
            mirror = mirror.replace("/", "\\")

        # print(exp + "\t\t" + mirror)
        mirror_emoticons[mirror] = tag
emoticons.update(mirror_emoticons)

for exp, tag in list(emoticons.items()):
    if exp.lower() not in emoticons:
        emoticons[exp.lower()] = tag

emoticon_groups = {
    "positive": {'batti il cinque', 'risata', 'cuore', 'felice'},
    "negative": {'annoiato', 'triste', }
}


def print_positive(sentiment):
    for e, tag in emoticons.items():
        if tag in emoticon_groups[sentiment]:
            print(e)

# print_positive("negative")
# print(" ".join(list(emoticons.keys())))
# [print(e) for e in list(emoticons.keys())]