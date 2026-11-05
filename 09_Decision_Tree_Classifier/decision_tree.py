import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("--- 1. Loading Dataset ---")
    # Load the built-in Iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    print(f"Features: {list(iris.feature_names)}")
    print(f"Classes: {list(iris.target_names)}")
    print(f"Dataset shape: {X.shape}\n")

    print("--- 2. Splitting Data ---")
    # Split the dataset into 70% training data and 30% testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Training instances: {len(X_train)} | Testing instances: {len(X_test)}\n")

    print("--- 3. Training Decision Tree Classifier ---")
    # Initialize the classifier using the Gini impurity criterion
    clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    print("Model trained successfully!\n")

    print("--- 4. Learned Tree Rules ---")
    # Print the logical If-Then rules the AI learned
    tree_rules = export_text(clf, feature_names=list(iris.feature_names))
    print(tree_rules)

    print("--- 5. Model Evaluation ---")
    # Predict the classes for our test set
    y_pred = clf.predict(X_test)
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {accuracy * 100:.2f}%\n")
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred), "\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

if __name__ == "__main__":
    main()