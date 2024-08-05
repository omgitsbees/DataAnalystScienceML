import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 

# Sample dataset
data = { 
    'title': ['The Matrix', 'Toy Story', 'Avatar', 'The Lion King', 'Jurassic Park'],
    'genre': ['Action', 'Animation', 'Action', 'Animation', 'Adventure'],
    'description': [
        'A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.',
        'A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy\'s room.',
        'A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world he feels is his home.',
        'Lion prince Simba and his father are targeted by his bitter uncle, who wants to ascend the throne himself.',
        'A pragmatic paleontologist visiting an almost complete theme park is tasked with protecting a couple of kids after a power failure causes the park\'s cloned dinosaurs to run loose.'
    ]
}

df = pd.DataFrame(data)

# Combine features into a single string
df['combined_features'] = df['genre'] + ' ' + df['description']

# Initialize CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(df['combined_features'])

# Compute the cosine similiarity
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Define the recommendation function
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = df[df['title'] == title].index[0]

    # Get the pairwise similiarity scores of all movies with the movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices]

# Test the recommendation system
print(get_recommendations('The Matrix'))