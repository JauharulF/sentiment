import nltk
from nltk.corpus import crubadan
from nltk.classify import textcat
from langid.langid import LanguageIdentifier, model

import MySQLdb
import MySQLdb.cursors

def process_lid(conn=None, table=None, lang=None, tct=None, lid=None):
    print(table, lang)
    cursor = conn.cursor()
    sql = """ SELECT id, {}_data FROM sentiment.{} ORDER BY id""".format(lang, table)
    cursor.execute(sql)
    rs = cursor.fetchall()
    for row in rs: 
        # print('%s (%s) %s' % (row['sentence'], crubadan.iso_to_crubadan(tct.guess_language(row['sentence'])), lid.classify(row['sentence'])))
        sql = """ UPDATE sentiment.{} SET {}_textcat=%s, {}_langid=%s WHERE id=%s """.format(table, lang, lang)
        cursor.execute(sql, (crubadan.iso_to_crubadan(tct.guess_language(row['{}_data'.format(lang)])), lid.classify(row['{}_data'.format(lang)])[0], row['id']))
        conn.commit()
        print('.', end='', flush=True)


tct = textcat.TextCat()
lid = LanguageIdentifier.from_modelstring(model, norm_probs=True)
# lid.set_languages(['en', 'ms'])

dbcon = MySQLdb.connect(host='localhost', user='jauharul', passwd='123456', db='sentiment', cursorclass=MySQLdb.cursors.DictCursor)
# cursor = dbcon.cursor()
# process_lid(dbcon, 'imdb_train', 'en', tct, lid)
# process_lid(dbcon, 'imdb_train', 'ms', tct, lid)
# process_lid(dbcon, 'imdb_test', 'en', tct, lid)
# process_lid(dbcon, 'imdb_test', 'ms', tct, lid)
