import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def lasso_regression(train_df, target_column, test_size=0.1, random_state=42):
    # Split data into features and target
    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]
    
    # Split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Standardize the features using only the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Performing Lasso regression...")
    
    # Lasso regression with cross-validation for hyperparameter tuning of regularization parameter alpha
    lasso = LassoCV(cv=5, random_state=random_state, max_iter=10000).fit(X_train_scaled, y_train)
    
    selected_features_lasso = np.where(lasso.coef_ != 0)[0]
    selected_feature_names_lasso = X.columns[selected_features_lasso]

    print(f"Selected features by Lasso ({len(selected_feature_names_lasso)}): {selected_feature_names_lasso}")

    print("Predicting on the training and test data (Lasso)...")

    y_train_pred_lasso = lasso.predict(X_train_scaled)
    y_test_pred_lasso = lasso.predict(X_test_scaled)

    print("Evaluating the model (Lasso)...")
    
    # Evaluating on train data
    train_rmse_lasso = np.sqrt(mean_squared_error(y_train, y_train_pred_lasso))
    train_mae_lasso = mean_absolute_error(y_train, y_train_pred_lasso)
    train_mape_lasso = mean_absolute_percentage_error(y_train, y_train_pred_lasso)

    # Evaluating on test data
    test_rmse_lasso = np.sqrt(mean_squared_error(y_test, y_test_pred_lasso))
    test_mae_lasso = mean_absolute_error(y_test, y_test_pred_lasso)
    test_mape_lasso = mean_absolute_percentage_error(y_test, y_test_pred_lasso)

    # Collect metrics
    results_lasso_dict = {
        "Model": ["Lasso"] * 6,
        "Metric": ["Training RMSE", "Training MAE", "Training MAPE",
                   "Testing RMSE", "Testing MAE", "Testing MAPE"],
        "Value": [f"{train_rmse_lasso:.2f}", f"{train_mae_lasso:.2f}", f"{train_mape_lasso:.2f}",
                  f"{test_rmse_lasso:.2f}", f"{test_mae_lasso:.2f}", f"{test_mape_lasso:.2f}"]
    }

    results_lasso_df = pd.DataFrame(results_lasso_dict)
    print(results_lasso_df)
    
    # Collect variable importance
    variable_importance_dict = {
        "Feature": selected_feature_names_lasso,
        "Coefficient": lasso.coef_[selected_features_lasso]
    }
    variable_importance_df = pd.DataFrame(variable_importance_dict)
    variable_importance_df = variable_importance_df.sort_values(by="Coefficient", ascending=False).reset_index(drop=True)

    print("\nVariable Importance (Lasso):")
    print(variable_importance_df)
    
    return lasso
