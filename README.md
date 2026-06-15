# World Cup Match Predictor

---
## Overview

This project is a World Cup match outcome prediction system trained on over 3000 past international matches. It uses structured statistical features combined with a neural network to estimate expected goals for both teams and simulate match outcomes.


It is designed for research and analysis rather than live betting or external data feeds. 

---

## Key Features

* Residual neural network with cross attention between team representations
* Elo rating system with tournament specific weighting
* Head to head historical performance modelling
* Decay weighted recent form over a 10 match window
* Confederation based team encoding
* Tournament importance scaling across different competition levels
* Weighted likelihood training using Poisson goal modelling
* Chronological train validation split
* Monte Carlo simulation using 50,000 samples for stable probabilities
* Expected goals output with scoreline probability distribution

---

## Dataset

The model was trained on over 3000 historical international football matches.

Each match record includes:

* Match date
* Home and away teams
* Goals scored
* Expected goals 
* Shots on target
* Possession statistics
* Tournament information
* Venue information

The dataset is processed chronologically to prevent future information leaking into historical predictions.

---

## Feature Engineering

The model generates features from several sources:

### Elo Ratings

Dynamic Elo ratings are maintained for every team and updated after each match. More important tournaments have a greater influence on rating changes.

### Recent Form

Recent performances are calculated using a decay weighted rolling window of the last 10 matches.

Metrics include:

* Goals scored
* Goals conceded
* Expected goals
* Expected goals conceded
* Shots on target
* Shots on target conceded
* Possession of the ball

### Head to Head Statistics

Historical meetings between teams are tracked and summarised using:

* Average goals scored
* Average goals conceded
* Average expected goals
* Number of previous meetings 

### Confederation Information

Teams are grouped by football confederation:

* UEFA - europe
* CONMEBOL - south america
* CONCACAF - north ameria
* CAF - africa
* AFC - asia
* OFC - oceania

---

## Model Architecture

The prediction engine uses a deep neural network built with PyTorch.

Components include:

### Team Embeddings

Separate attack and defence embeddings are learned for every team.

### Confederation Embeddings

Each confederation receives its own learned representation.

### Cross Attention Layer

Home and away team representations interact through a cross attention mechanism , allowing the model to learn matchup specific relationships.

### Residual Network - prevents degrading 

Several residual blocks process the combined feature representation and improve learning stability.

### Goal Prediction Heads

Two independent output heads estimate expected goals for :

* Home team
* Away team

The outputs are constrained to positive values and also interpreted as Poisson goal expectations.

---

## Training

The model is trained using a weighted Poisson negative log likelihood objective.

Training includes:

* AdamW optimisation
* Learning rate scheduling
* Gradient clipping
* Validation loss tracking
* Early stopping - max epochs

Match importance is incorporated through tournament tier weighting.

---

## Prediction Pipeline

For each matchup:

1. Current team ratings and form are reconstructed
2. Features are passed through the neural network
3. Expected goals are generated for both teams
4. 50,000 Monte Carlo simulations are performed
5. Match outcome probabilities are calculated
6. Most likely scorelines are identified

A Dixon Coles adjustment is also applied to improve low scoring probability estimates.

---

## Example Output

The predictor provides:

* Expected goals for both teams
* Win probability
* Draw probability
* Loss probability
* Most likely scoreline
* Top scoring outcomes
* Elo comparison
* Head to head information
* Venue context

---

## Requirements

```bash
pip install torch pandas numpy openpyxl
```
---

## Usage

Place the dataset file in the project directory:

```text
Advanced_WorldCup_Stats_2014_2026.xlsx
```

Run the script:

```bash
python main.py
```

If a trained model already exists it will be loaded automatically.

Otherwise a new model will be trained and saved for future use.

When prompted, enter two team names to generate a prediction.

```text
Team 1: England
Team 2: Brazil
```

Type `exit` at any prompt to quit.

---

## Project Structure

```text
.
├── main.py
├── Advanced_WorldCup_Stats_2014_2026.xlsx
├── worldcup_v2_model.pt
└── README.md
```

## Notes

This project is intended for football analytics and machine learning experimentation.

Prediction accuracy depends on data quality, historical coverage, and the availability of recent match information.

Results should be treated as probabilistic estimates rather than guaranteed outcomes.

## Licence

This project is released under the MIT Licence.
