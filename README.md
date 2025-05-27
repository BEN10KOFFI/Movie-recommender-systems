# 🎬 Movie Recommender System

This project implements a personalized movie recommender system using **Alternating Least Squares (ALS)** enhanced with **bias terms** and **latent factors**, applied on the **[MovieLens 100K-25M Datase](https://grouplens.org/datasets/movielens/)**. It emphasizes both mathematical clarity and practical implementation.

---

## 📌 Motivation

When choosing a movie, we are often overwhelmed by countless options. This system mimics a human assistant who understands your preferences and makes personalized recommendations. Inspired by collaborative filtering techniques such as those used in the **Netflix Prize (2006)**, this work explores the use of **matrix factorization** for scalable and intelligent predictions.

---

## 📊 Datasets Used

* **[MovieLens 100K ](https://grouplens.org/datasets/movielens/)** — 100,836 ratings by 610 users on 9,742 movies
* **[MovieLens 25M ](https://grouplens.org/datasets/movielens/)** — 25 million ratings by 162,000 users on 62,000 movies

Each dataset includes:
`movies.csv`, `ratings.csv`, `tags.csv`, `links.csv`

---

## 🧠 Learning Models

The project contrasts two recommendation paradigms:

* **Content-based filtering**, which builds a profile for each user using item metadata
* **Collaborative filtering**, which exploits the preferences of similar users, even when explicit data is missing

We focus on the latter by using **Probabilistic Matrix Factorization (PMF)** to recover missing ratings from the user–item matrix $R$.

We assume a low-rank approximation:

$R \approx UV^T$

Where:

* $U \in \mathbb{R}^{M \times K}$ is the latent matrix for $M$ users
* $V \in \mathbb{R}^{N \times K}$ is the latent matrix for $N$ items

The rating matrix $R \in \mathbb{R}^{M \times N}$ is sparse. PMF fills in the blanks using **Bayesian learning**:

$p(U, V | R, \sigma^2) \propto p(R | U, V, \sigma^2) p(U | \sigma_U^2) p(V | \sigma_V^2)$

Assuming Gaussian priors and a likelihood term, we obtain the log-likelihood and optimize it with respect to $U$ and $V$ using alternating least squares (ALS).

---

## ⚙️ Core Algorithm – ALS

The complete model includes:

$$
R_{mn} \approx U_m^T V_n + b^{(U)}_m + b^{(V)}_n
$$

Where:

* $b^{(U)}$, $b^{(V)}$ are user and item biases
* ALS alternates between optimizing $U$ and $V$ while holding the other fixed

---

## 📈 Results & Insights

* Models trained on **MovieLens 100K** and **25M** showed strong performance and generalization
* Using both biases and latent factors improves predictions compared to either alone
* The sparsity of $R$ is effectively handled by PMF under ALS, achieving high scalability

---

## 📂 Report For Further Reading

A detailed explanation, including derivations, evaluations and results, is available in the PDF report of this project linked below:

🔗 [Project Report (PDF)](https://drive.google.com/file/d/1bkLjHE7IcfCrQjBmgb9V1KsiC2BMej1L/view?usp=sharing)

---

## 🛠️ Tools Used

* Python (NumPy, Pandas, Matplotlib)
* Scikit-learn
