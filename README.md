# 🍽️ Restaurant Recommendation in Jakarta using Collaborative Filtering
The rapid growth of the culinary sector in Jakarta has led to the emergence of numerous restaurants offering a wide variety of both local and international cuisines. However, having too many options often makes it difficult for people to decide where to eat. To overcome this problem, a recommendation system can help users find restaurants that match their preferences based on other users’ ratings. This study compares three recommendation system algorithms: SVD (Singular Value Decomposition), VAE (Variational Autoencoder), and Item-based Collaborative Filtering. These algorithms are evaluated using the metrics RMSE, MAE, Precision\@10, and Recall\@10. Experimental results show that SVD outperforms the other models and is deployed via a Streamlit web application.

---

## 🧠 Algorithms Compared

Three collaborative filtering-based recommendation models were implemented and compared:

| Model             | Description                                              |
| ----------------- | -------------------------------------------------------- |
| **SVD**           | Matrix factorization using Singular Value Decomposition  |
| **VAE**           | Deep learning-based Variational Autoencoder model        |
| **Item-based CF** | Collaborative Filtering using Adjusted Cosine Similarity |

---

## 📈 Evaluation Metrics

To evaluate performance, the following metrics were used:

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* Precision\@10
* Recall\@10

**Best results** achieved by SVD:

* 📉 RMSE: `0.4203`
* 📉 MAE: `0.3170`
* 🎯 Precision\@10: `0.7833`
* 🔁 Recall\@10: `0.8014`

---

## 🚀 Deployment

The best model (SVD) was deployed using [**Streamlit**](https://streamlit.io/), allowing users to:

* Enter their 3-5 favorite restaurant
* Get top 10 restaurant recommendations

🔗 **Access the App**: [jakarta-restaurant-recommendation.streamlit.app](https://jakarta-restaurant-recommendation.streamlit.app)

---

## ⚙️ Installation

To run locally:

```bash
git clone https://github.com/joselynss/restaurant-recommendation.git
cd restaurant-recommendation
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧑‍💻 Author

**Joselyn Setiawan**

Email: *[joselynsetiawan@gmail.com](mailto:joselynsetiawan@gmail.com)*
