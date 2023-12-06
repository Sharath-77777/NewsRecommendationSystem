from flask import Flask, render_template, request

import pandas as pd
import numpy as np
import pickle
from math import sqrt
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# Load the dataset from CSV
file_path = 'news2.csv'
df = pd.read_csv(file_path)

# Train the model and get average RMSE and user_item_matrix
from model import train_model

average_rmse, user_item_matrix = train_model(df)

# Load the trained model and user_item_matrix from the pickle file
with open('trained_model.pkl', 'rb') as model_file:
    user_id_to_index, news_id_to_index, u_k, sigma_k, vt_k, user_item_matrix = pickle.load(model_file)


# Define Flask routes
@app.route('/')
def index():
    return render_template('index.html', average_rmse=average_rmse)


@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    try:
        user_id = int(request.form['user_id'])
        user_index_to_recommend = user_id_to_index[user_id]
        user_ratings = np.dot(u_k[user_index_to_recommend, :], np.dot(sigma_k, vt_k))

        # Get unrated news indices for the specified user using user_item_matrix
        unrated_news = np.where(user_item_matrix[user_index_to_recommend, :] == 0)[0]

        # Get top 5 recommendations for unrated news
        top5_recommendations = unrated_news[np.argsort(user_ratings[unrated_news])[::-1]][:5]

        return render_template('index.html', user_id=user_id, recommendations=top5_recommendations,
                               show_recommendations=True)
    except Exception as e:
        return {'error': str(e)}


if __name__ == '__main__':
    app.run(debug=True)
