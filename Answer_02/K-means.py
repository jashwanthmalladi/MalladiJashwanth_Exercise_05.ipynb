from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('/content/Amazon_Unlocked_Mobile.csv')

# Handle missing values in the 'Reviews' column
imputer = SimpleImputer(strategy='constant', fill_value='')
df['Reviews'] = imputer.fit_transform(df[['Reviews']])

# Vectorize the text data (TF-IDF)
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(df['Reviews'])

# Apply K-means clustering
num_clusters = 5  # Adjust as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['kmeans_cluster'] = kmeans.fit_predict(X)

# Print the clusters
print(df[['Reviews', 'kmeans_cluster']])
