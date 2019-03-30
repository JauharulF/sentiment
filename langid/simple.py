sentences = ["This is a test",
             "This is a simple example.",
             "Ini adalah contoh sederhana.",
             "Kalimat ini adalah contoh sederhana.",
             'Awak nak pergi kemana',
             'Seorang pelajar lelaki lemas ketika mengikuti sesi belajar berenang semalam',
             'Beliau memaklumkan pegawai penyiasat berhubung perkara itu'
            ]

### This part is for TextCat ###
import nltk
# nltk.download('crubadan')
# nltk.download('punkt')

from nltk.classify import textcat
tct = textcat.TextCat()

### This part is for langid.py ###
from langid.langid import LanguageIdentifier, model
lid = LanguageIdentifier.from_modelstring(model, norm_probs=True)
lid.set_languages(['en', 'ms'])

### try to run both identifier for all the sentence 
for sentence in sentences:
    print('%s (%s) %s' % (sentence, tct.guess_language(sentence), lid.classify(sentence)))
