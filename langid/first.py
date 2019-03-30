import nltk
from nltk.corpus import crubadan
from nltk.classify import textcat
from langid.langid import LanguageIdentifier, model

tct = textcat.TextCat()
lid = LanguageIdentifier.from_modelstring(model, norm_probs=True)
# lid.set_languages(['en', 'ms'])

import MySQLdb
import MySQLdb.cursors

dbcon = MySQLdb.connect(host='localhost', user='jauharul', passwd='123456', db='sentiment', cursorclass=MySQLdb.cursors.DictCursor)
cursor = dbcon.cursor()
sql = """ SELECT id, sentence FROM sentiment.socmed ORDER BY id"""
cursor.execute(sql)
rs = cursor.fetchall()
for row in rs: 
    # print('%s (%s) %s' % (row['sentence'], crubadan.iso_to_crubadan(tct.guess_language(row['sentence'])), lid.classify(row['sentence'])))
    sql = """ UPDATE sentiment.socmed_stat SET textcat=%s, langid=%s WHERE id=%s """
    cursor.execute(sql, (crubadan.iso_to_crubadan(tct.guess_language(row['sentence'])), lid.classify(row['sentence'])[0], row['id']))
    dbcon.commit()
    print('.', end='', flush=True)
