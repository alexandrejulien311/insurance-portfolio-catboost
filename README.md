# Non-life insurance portfolio – CatBoost pricing tool

Developed a pricing tool for a non-life motor insurance powered by CatBoosting trained on an already priced portfolio.

1. visualization of the portfolio with graphics
2. evaluation of the trained model with graphics
3. prediction of the trained model on new insured
4. model training

Dataset used: https://data.mendeley.com/datasets/5cxyb5fp4f/1


## Motivation
Developped for a Master's project (M1 level) in actuarial Science.

## Results and limits
The accuracy of the pricing tool is strongly limited by the dataset coherence. Garbage-in, garbage-out as usual...
It worked as wished anyways.

## Installation
```bash
conda env create -f environment.yml
conda activate insurance-portfolio-catboost-conda-env
```

## Usage
Three first points:
```bash
python main.py
```
Training the model (fourth point):
```bash
python src/insurance_portfolio_catboost/catboost_model_training.py
```