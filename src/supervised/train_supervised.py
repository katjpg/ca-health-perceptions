from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import silhouette_score, davies_bouldin_score
from xgboost import XGBRegressor

def train_model(X, y, model, model_name, scaler=None):
    """
    This function fits a model on the input features and target variable. It first 
    standardizes the features using the provided scaler and then applies the model. 
    It splits the data into train and test sets with a test size of 20% for model validation. 
    The function also evaluates the performance of the model by computing and printing the 
    Mean Squared Error, Mean Absolute Error and R^2 score for both the training and test sets.

    Args:
        X (np.array or pd.DataFrame): The input features to be used for training the model.
        y (np.array or pd.Series): The target variable for the model.
        model (sklearn estimator): The model to train.
        model_name (str): The name of the model (for printouts).
        scaler (sklearn transformer, optional): The scaler to apply before training the model.

    Returns:
        pipeline (sklearn.pipeline.Pipeline): The fitted sklearn Pipeline that includes the scaler 
        for feature scaling (if provided) and the model.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    steps = []
    if scaler:
        steps.append(('scaler', scaler))
    steps.append((model_name, model))
    pipeline = Pipeline(steps)

    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    print(f"----- {model_name} Training Data -----")
    print("Mean Squared Error: ", mean_squared_error(y_train, y_train_pred))
    print("Mean Absolute Error: ", mean_absolute_error(y_train, y_train_pred))
    print("R^2 Score: ", r2_score(y_train, y_train_pred))

    print(f"\n----- {model_name} Test Data -----")
    print("Mean Squared Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Error: ", mean_absolute_error(y_test, y_test_pred))
    print("R^2 Score: ", r2_score(y_test, y_test_pred))

    return pipeline
