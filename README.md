# MLModel2

## Progress as on March 18th, 2022:
- [x] Cleaned Dataset and genrated two new datasets for use in ML model.   
- [x] Visualized the clusters on a scatter plot using k-mean clustering with a randomly generated sample dataset from our two newly created datasets.   
- [] Created a training model for both datasets.    

-----
### Cleaned raw data to make sure only the four features will remain.
library used for cleaning data: 
```python
import pandas as pd
```

-----
### Created two datasets from raw data retrieved from TMDB via API.

### Dataset1 Preview:
Language | Genre | Year | Vote_Average
---------|-------|------|--------------
216      |   99  | 1896 |    6.2       
240      |   16  | 1923 |    4.9       
240      |   115 | 2009 |    8.1       

* Language: Convert each letter from an original_language string to its corresponding ASCII number and get their sum.
```python
def getASCIISum(language):
	return ord(language[0]) + ord(language[1])

df['modified_language'] = df['original_language'].apply(getASCIISum)
```   
* Genre: Add all the genre id from genre_ids for each movie.
```python
def getGenreIDSum(genre_ids):
	genres = genre_ids[1:-1].split(', ')
	sum = 0
	for genre in genres:
		if len(genre) > 0:
			genre = int(genre)
			sum = sum + genre
	return sum

df['modified_genre'] = df['genre_ids'].apply(getGenreIDSum)
```

* Year: Only taking the year from the release date of each movie.
```python
def removeMonthDate(release_date):
	return int(release_date[0:4])

df['release_date'] = df['release_date'].apply(removeMonthDate)
```

* Vote_Average: The vote average from the original dataset.   



-----
### Dataset2(With Min-Max Scaling) Preview:
Language | Genre | Year | Vote_Average
---------|-------|------|--------------
1.0      | 0.008 |0.054 |    3.1       
0.134    | 0.135 |0.7   |    0.0       
0.091    | 0.141 |0.47  |    1.7       

The four feature columns are normalized using Min-Max Scaling method to make sure all the data points fall between 0 to 1.   
Min-Max Scaling formula: (df[i] - df.min) / (df.max - df.min)
* Language: 
```python
df['modified_language'] = (df['modified_language'] - df['modified_language'].min())/(df['modified_language'].max() - df['modified_language'].min())
```

* Genre:
```python
df['modified_genre'] = (df['modified_genre'] - df['modified_genre'].min())/(df['modified_genre'].max() - df['modified_genre'].min())
```

* Year:
```python
df['release_date'] = (df['release_date'] - df['release_date'].min())/(df['release_date'].max() - df['release_date'].min())
```

* Vote_Average:
```python
df['vote_average'] = (df['vote_average'] - df['vote_average'].min())/(df['vote_average'].max() - df['vote_average'].min())
```
-----
