import pandas as pd
from catboost import CatBoostRegressor
from pathlib import Path
from data import DATA_FOLDER_PATH
from src.insurance_portfolio_catboost.visualizer.visualize_results import (
    save_prediction_plot,
    save_prediction_plot_pct,
    loss_graph,
)
from src.insurance_portfolio_catboost.visualizer.visualize_data_input import (
    premium_distribution,
    correlation_heatmap,
    segmentation_input,
)
import numpy as np


PATH_CURRENT_FILE = Path(__file__).resolve().parent
PATH_DATA_INPUT_FOLDER = Path(PATH_CURRENT_FILE) / "data_input_visualization"
PATH_DATA_OUTPUT_FOLDER = Path(PATH_CURRENT_FILE) / "results_visualization"


PATH_DATA = Path(DATA_FOLDER_PATH) / "data_cleaned.csv"


data_cleaned_df = pd.read_csv(PATH_DATA, delimiter=";")
data_cleaned_df = data_cleaned_df.drop(
    ["contrat_connue", "Date_last_renewal", "ID", "connu"], axis=1
)

X = data_cleaned_df.drop("Premium", axis=1)
y = data_cleaned_df["Premium"]

X_new = pd.read_csv(Path(DATA_FOLDER_PATH) / "data_new_insured.csv", delimiter=";")


def main():

    # input visualisation
    correlation_heatmap(
        PATH_DATA_INPUT_FOLDER / Path("heatmap.png"), df=data_cleaned_df
    )

    segmentation_input(
        PATH_DATA_INPUT_FOLDER / Path("segmentation.png"), df=data_cleaned_df
    )

    premium_distribution(
        PATH_DATA_INPUT_FOLDER / Path("premium_distribution.png"), df=data_cleaned_df
    )

    # model evaluation
    model_path = Path(PATH_CURRENT_FILE) / "catboost_premium.cbm"

    model_loaded = CatBoostRegressor()
    model_loaded.load_model(str(model_path))
    y_pred = model_loaded.predict(X)

    loss_graph(model_loaded, Path(PATH_DATA_OUTPUT_FOLDER) / "loss_graph.png")

    premium_sumd_df = pd.DataFrame({"Premium": y_pred})
    premium_distribution(
        path=PATH_DATA_OUTPUT_FOLDER / Path("results_premiums.png"), df=premium_sumd_df
    )

    save_prediction_plot(
        y_true=y,
        y_pred=y_pred,
        path=Path(PATH_DATA_OUTPUT_FOLDER) / "comparison_plot.png",
    )

    save_prediction_plot_pct(
        y_true=y,
        y_pred=y_pred,
        path=Path(PATH_DATA_OUTPUT_FOLDER) / "comparison_plot_pct.png",
    )

    # Use case: a premium prediction of some new contracts
    X_new_filtered = X_new.drop(
        ["contrat_connue", "Date_last_renewal", "ID", "connu"], axis=1
    )
    y_new_pred = model_loaded.predict(X_new_filtered)

    df_out = X_new.copy()
    df_out["premium_pred"] = np.ravel(y_new_pred)
    df_out.to_csv(Path(PATH_CURRENT_FILE) / "data_new_insured_priced.csv", index=False)


main()
