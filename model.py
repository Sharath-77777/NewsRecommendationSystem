import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle

def train_model(df):
    # Map user and news IDs to matrix indices
    user_id_to_index = {user_id: index for index, user_id in enumerate(df['user_id'].unique())}
    news_id_to_index = {news_id: index for index, news_id in enumerate(df['news_id'].unique())}

    # Add matrix index columns to the DataFrame
    df['user_index'] = df['user_id'].map(user_id_to_index)
    df['news_index'] = df['news_id'].map(news_id_to_index)

    # Create user-item matrix
    num_users = len(user_id_to_index)
    num_news = len(news_id_to_index)

    user_item_matrix = np.zeros((num_users, num_news))

    for row in df.itertuples():
        user_item_matrix[row.user_index, row.news_index] = row.rating

    # Define number of folds for cross-validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize RMSE and recommendation lists
    rmse_list = []

    # Perform k-fold cross-validation
    for train_index, test_index in kf.split(df):
        train_data, test_data = df.iloc[train_index], df.iloc[test_index]

        # Create user-item matrices for training and testing
        train_matrix = np.zeros((num_users, num_news))

        for row in train_data.itertuples():
            train_matrix[row.user_index, row.news_index] = row.rating

        test_matrix = np.zeros((num_users, num_news))

        for row in test_data.itertuples():
            test_matrix[row.user_index, row.news_index] = row.rating

        # Perform Singular Value Decomposition (SVD)
        u, sigma, vt = np.linalg.svd(train_matrix, full_matrices=False)

        # Choose the number of latent factors (adjust as needed)
        num_factors = 20

        # Approximate the original matrix using the selected number of latent factors
        u_k = u[:, :num_factors]
        sigma_k = np.diag(sigma[:num_factors])
        vt_k = vt[:num_factors, :]

        predicted_ratings = np.dot(np.dot(u_k, sigma_k), vt_k)

        # Evaluate the model using RMSE
        predicted_ratings_flattened = predicted_ratings[test_matrix.nonzero()]
        test_matrix_flattened = test_matrix[test_matrix.nonzero()]
        rmse_fold = sqrt(mean_squared_error(predicted_ratings_flattened, test_matrix_flattened))
        rmse_list.append(rmse_fold)

    # Calculate and return the average RMSE and user_item_matrix
    average_rmse = round(np.mean(rmse_list),2)

    # Save the trained model and user_item_matrix to a pickle file
    with open('trained_model.pkl', 'wb') as model_file:
        pickle.dump((user_id_to_index, news_id_to_index, u_k, sigma_k, vt_k, user_item_matrix), model_file)

    return average_rmse, user_item_matrix
