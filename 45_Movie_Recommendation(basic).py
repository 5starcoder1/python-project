import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("movies.csv")

# Convert text data into vectors
cv = CountVectorizer()
vectors = cv.fit_transform(df['genre'])

# Similarity matrix
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie_name):
    movie_index = df[df['title'] == movie_name].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    print(f"\n🎬 Recommended movies for '{movie_name}':\n")

    for i in movies_list:
        print(df.iloc[i[0]].title)

# User input
while True:
    movie = input("\nEnter movie name: ")
    
    if movie in df['title'].values:
        recommend(movie)
    else:
        print("❌ Movie not found in dataset")
