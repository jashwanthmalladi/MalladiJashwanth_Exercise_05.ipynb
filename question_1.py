import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Read the data from text files
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    return data

# Load the data
train_data = read_text_file('stsa-train.txt')
test_data = read_text_file('stsa-test.txt')

# Separate features and labels
X_train = [line.split(' ', 1)[1].strip() for line in train_data]
y_train = [int(line.split(' ', 1)[0]) for line in train_data]

X_test = [line.split(' ', 1)[1].strip() for line in test_data]
y_test = [int(line.split(' ', 1)[0]) for line in test_data]

# Split the training data into training and validation sets (80-20 split)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# lassifiers
classifiers = {
    'MultinomialNB': MultinomialNB(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'XGBoost': XGBClassifier()
}

# Perform 10-fold cross-validation
for clf_name, clf in classifiers.items():
    print(f"Training and evaluating {clf_name}...")
    
    # Perform 10-fold cross-validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_train_vec, y_train, cv=kfold, scoring='accuracy')
    
    # Print average accuracy across folds
    print(f"Average Cross-Validation Accuracy for {clf_name}: {scores.mean():.4f}")

    # Train the final model 
    clf.fit(X_train_vec, y_train)

    #Predictions on the validation set
    y_val_pred = clf.predict(X_val_vec)

    #Evaluation
    accuracy = accuracy_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)

    #evaluation metrics on the validation set
    print(f"Accuracy for {clf_name}: {accuracy:.4f}")
    print(f" Recall for {clf_name}: {recall:.4f}")
    print(f"Precision for {clf_name}: {precision:.4f}")
    print(f"F-1 Score for {clf_name}: {f1:.4f}")
    print()

    # Evaluate the final trained model
    y_test_pred = clf.predict(X_test_vec)
    
    # evaluation metrics 
    print(f"Test Accuracy for {clf_name}: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Test Recall for {clf_name}: {recall_score(y_test, y_test_pred):.4f}")
    print(f"Test Precision for {clf_name}: {precision_score(y_test, y_test_pred):.4f}")
    print(f"Test F-1 Score for {clf_name}: {f1_score(y_test, y_test_pred):.4f}")
    print("="*50)
