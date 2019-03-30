SELECT en_textcat, COUNT(*) FROM sentiment.imdb_train group by en_textcat;
SELECT en_langid, COUNT(*) FROM sentiment.imdb_train group by en_langid;
SELECT en_textcat, COUNT(*) FROM sentiment.imdb_test group by en_textcat;
SELECT en_langid, COUNT(*) FROM sentiment.imdb_test group by en_langid;

SELECT ms_textcat, COUNT(*) FROM sentiment.imdb_train group by ms_textcat;
SELECT ms_langid, COUNT(*) FROM sentiment.imdb_train group by ms_langid;
SELECT ms_textcat, COUNT(*) FROM sentiment.imdb_test group by ms_textcat;
SELECT ms_langid, COUNT(*) FROM sentiment.imdb_test group by ms_langid;

SELECT textcat, count(*) FROM sentiment.socmed_stat WHERE textcat=langid group by textcat;
SELECT count(*) FROM sentiment.socmed_stat WHERE textcat=langid;
SELECT count(*) FROM sentiment.socmed_stat WHERE textcat<>langid;
SELECT count(*) FROM sentiment.socmed_stat WHERE textcat<>langid AND (textcat='en' OR langid='en');