# Discussion:

Based on Sultan et. al. (2016), use of Aligned words of both sentences can be use to predict the similarity. The equation given by Sultan et. al. uses the ratio between the number of words aligned to total number of words in both sentences. Using this similarity on it's own did not result in very good predictions. The average score on the train sets was about 40%. However, combining this score with cosine similarity score calculated in the baseline can improve the overall results.

For alignment the code provided in supplementary information of the "A Lightweight and High Performance MonolingualWord Aligner" paper by Xuchen Yao et. al. was used. 

We noticed upon removing the punctution from the sentences, while predicting the baseline, the cosine similarity prediction can be improved by about 0.8% (from scores observed in leaderboard submissions). This can be due to the averaging problem, where less common words causes the average embedding to become more sensitive to the actual words that are different.
