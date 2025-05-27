# 🎬 Movie Recommender System

This project implements a personalized movie recommender system using **Alternating Least Squares (ALS)** with both **biases** and **latent factors**, built on the **MovieLens 100K and 25M datasets**. It is part of the "Machine Learning at Scale" module at AIMS South Africa.

---

## 📌 Motivation

When choosing a movie, we are often overwhelmed by endless options. This system aims to replicate the decision-making of a human who understands your tastes — recommending films you’re likely to enjoy. Inspired by early recommendation engines like Netflix Prize (2006), this project explores the fundamentals of **collaborative filtering** using matrix factorization.

---

## 📊 Datasets Used

- **MovieLens 100K** — 100,836 ratings by 610 users on 9,742 movies  
- **MovieLens 25M** — 25 million ratings by 162,000 users on 62,000 movies

Each dataset includes:  
`movies.csv`, `ratings.csv`, `tags.csv`, `links.csv`

---

## 🧠 Approach

The project focuses on **Collaborative Filtering**, implemented through:

1. **Baseline model with biases only**  
2. **ALS with latent factors (matrix factorization)**  
3. **Full model: biases + latent factors (optimized via ALS)**

We use **Probabilistic Matrix Factorization (PMF)** under a Bayesian framework to estimate the user–item interaction matrix \( R \approx UV^T \).

---

## ⚙️ Core Algorithm – ALS

The objective function is optimized alternately over:
- User latent factors \( U \)  
- Item latent factors \( V \)  
- User and item biases \( b_U, b_V \)

The optimization steps follow:

```math
R_{mn} \approx U_m^\top V_n + b^{(U)}_m + b^{(V)}_n

