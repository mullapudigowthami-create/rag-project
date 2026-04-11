# =============================
# Movie Recommendation System
# =============================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# STEP 1 - Create Movie Dataset
data = {
    'Movie': [
        'Avengers', 'Iron Man', 'Thor',
        'Titanic', 'Notebook', 'Romeo Juliet',
        'Fast Furious', 'Need for Speed', 'Rush',
        'Lion King', 'Jungle Book', 'Tarzan'
    ],
    'Genre': [
        'action superhero marvel',
        'action superhero marvel',
        'action superhero marvel',
        'romance drama love',
        'romance drama love',
        'romance drama love',
        'action racing cars',
        'action racing cars',
        'action racing cars',
        'animation adventure jungle',
        'animation adventure jungle',
        'animation adventure jungle'
    ]
}

df = pd.DataFrame(data)

# STEP 2 - Convert Genre to Numbers
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['Genre'])

# STEP 3 - Calculate Similarity
similarity = cosine_similarity(tfidf_matrix)

# STEP 4 - Recommendation Function
def recommend_movies(movie_name):
    # Find movie index
    idx = df[df['Movie'] == movie_name].index[0]
    
    # Get similarity scores
    scores = list(enumerate(similarity[idx]))
    
    # Sort by similarity
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    # Get top 3 recommendations
    print(f"\n🎬 Movies similar to {movie_name}:")
    for i, score in scores[1:4]:
        print(f"✅ {df['Movie'][i]} - Match: {score:.2f}")

# STEP 5 - Test Recommendations
recommend_movies('Avengers')
recommend_movies('Titanic')
recommend_movies('Fast Furious')