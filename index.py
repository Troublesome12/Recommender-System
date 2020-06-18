# Import Pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
import warnings; warnings.simplefilter('ignore')


# Load Movies Metadata
md = pd.read_csv('./data/movies_metadata.csv')

# Formating the dataset
md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

# Calculate mean of vote average column
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()

# Calculate the minimum number of votes required to be in the chart, m
vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
m = vote_counts.quantile(0.90)

# Filter out all qualified movies into a new DataFrame
qualified = md.copy().loc[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())]

# Function that computes the weighted rating of each movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

# Define a new feature 'score' and calculate its value with `weighted_rating()`
qualified['score'] = qualified.apply(weighted_rating, axis=1)

#Sort movies based on score calculated above
qualified = qualified.sort_values('score', ascending=False).head(300)

s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = md.drop('genres', axis=1).join(s)

def build_chart(genre, percentile=0.90):
	df = gen_md[gen_md['genre'] == genre]
	vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
	vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
	C = vote_averages.mean()
	m = vote_counts.quantile(percentile)

	qualified = df.copy().loc[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())]
	qualified['score'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
	qualified = qualified.sort_values('score', ascending=False).head(100)
	
	return qualified[['title', 'vote_count', 'vote_average', 'score', 'year']]


#Print the top 15 movies
print('Top 20 movies')
print(qualified[['title', 'genres', 'vote_count', 'vote_average', 'score', 'year']].head(20))
print('Top 5 Romance movies')
print(build_chart('Romance').head(5))
print('Top 5 Action movies')
print(build_chart('Action').head(5))
print('Top 5 Comedy movies')
print(build_chart('Comedy').head(5))
print('Top 5 Drama movies')
print(build_chart('Drama').head(5))
