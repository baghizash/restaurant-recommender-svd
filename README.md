# Restaurant Recommender with SVD

A from-scratch recommender system on Yelp restaurant ratings: vector similarity measures, similar-restaurant search, and dimensionality reduction with Truncated SVD.

Coursework from **Data Mining in Python** (University of Michigan, More Applied Data Science with Python specialization). All notebooks run end to end with outputs.

## Data

`data/` holds course-provided Yelp data for Montréal restaurants (IDs randomized for privacy):

- `montreal_business.csv` — restaurant attributes (business_id, name, stars, ...)
- `montreal_user.csv` — user ratings of restaurants (~3.5 MB)

The ratings are pivoted into a **2,770 × 11,937** user-by-restaurant matrix (implemented in Part 1 as `row_col_count`).

## What was done

**Part 1 — Rating matrix** (`notebooks/part1-rating-matrix.ipynb`)
Loaded and cleaned both CSVs and built the sparse user–restaurant rating matrix (2,770 rows, 11,937 columns).

**Part 2 — Vector similarity from scratch** (`notebooks/part2-vector-similarity.ipynb`)
Implemented Manhattan distance, Euclidean distance, and cosine similarity manually (no extra imports), the building blocks for comparing restaurants as rating vectors.

**Part 3 — Find similar restaurants** (`notebooks/part3-find-similar-restaurants.ipynb`)
Used dot products and cosine similarity on real rating vectors to answer: which restaurants are most similar to **Modavie** (a favorite Montréal restaurant) in terms of customer ratings? Also compared mean cosine similarity among nearby restaurants vs. French restaurants.

**Part 4 — Patterns with SVD** (`notebooks/part4-svd-patterns.ipynb`)
Applied Truncated SVD (scikit-learn) to compress the rating matrix, then implemented `svd_transformed_rating`, ranked restaurants by dot product on raw vs. transformed vectors, and measured with Jaccard similarity how much the top-N sets change after dimensionality reduction. This is the collaborative-filtering idea behind Netflix-style recommenders.

## Tech

Python, pandas, NumPy, scikit-learn (TruncatedSVD), matplotlib, Jupyter.
