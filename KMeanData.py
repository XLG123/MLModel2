import pandas as pd

# Read the dataset into a pandas dataframe.
orig = pd.read_csv('tmdb_dataset.csv')
# print(orig)

# Remove duplicated index column.
orig.drop(columns=orig.columns[0], axis=1, inplace=True)

# Remove duplicated movies indicated by id.
df = orig.drop_duplicates(subset='id', keep='last')

# Remove rows with vote count as 0.
df = df[df['vote_count']!=0]

# Remove unnecessary columns from the dataset.
df = df[['original_language', 'genre_ids', 'release_date', 'vote_average']]

# A function to get the ascii sum of the language from each movie.
def getASCIISum(language):
	return ord(language[0]) + ord(language[1])
df['original_language'] = df['original_language'].apply(getASCIISum)

# A function to remove month and dates from the release date column.
def removeMonthDate(release_date):
	return int(release_date[0:4])
df['release_date'] = df['release_date'].apply(removeMonthDate)

# A function to get the sum of genres of genre_ids from each movie.
def getGenreIDSum(genre_ids):
	genres = genre_ids[1:-1].split(', ')
	sum = 0
	for genre in genres:
		if len(genre) > 0:
			sum = sum + genre
	return sum
df['genre_ids'] = df['genre_ids'].apply(getGenreIDSum)

# Rename the columns.
df.columns = ['Language', 'Genre', 'Year', 'Vote_Average']
df.to_csv('KMeanDataset.csv', index=False) # Save dataframe as a csv file.

# Normalize the four columns with Min-Max Scaling method.
df['Language'] = (df['Language'] - df['Language'].min())/(df['Language'].max() - df['Language'].min())
df['Genre'] = (df['Genre'] - df['Genre'].min())/(df['Genre'].max() - df['Genre'].min())
df['Year'] = (df['Year'] - df['Year'].min())/(df['Year'].max() - df['Year'].min())
df['Vote_Average'] = (df['Vote_Average'] - df['Vote_Average'].min())/(df['Vote_Average'].max() - df['Vote_Average'].min())
df.to_csv('KMeanDatasetMMS.csv', index=False) # Save dataframe as a csv file.