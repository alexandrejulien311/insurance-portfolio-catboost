import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from data import DATA_FOLDER_PATH


def correlation_heatmap(path, df) -> None:
    fig = plt.figure(figsize=(14, 8))
    mosaic = fig.subplot_mosaic(
        """
        A
        """,
        gridspec_kw={"height_ratios": [1]},
    )

    num_for_corr = df.select_dtypes(include=["number"]).columns
    if len(num_for_corr) > 1:
        ax_corr = mosaic["A"]
        corr = df[num_for_corr].corr()
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax_corr)
        ax_corr.set_title("Correlation")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)


def segmentation_input(path, df) -> None:
    fig = plt.figure(figsize=(14, 8))
    mosaic = fig.subplot_mosaic(
        """
        ABC
        DEF
        """,
    )

    cat_cols = [
        "Type_risk",
        "Area",
        "Distribution_channel",
        "Second_driver",
        "Type_fuel",
    ]

    axes_cat = ["A", "B", "C"]
    for ax_key, col in zip(axes_cat, cat_cols):
        if col in df.columns:
            ax = mosaic[ax_key]
            order = df[col].value_counts().index[:15]
            sns.countplot(data=df, x=col, order=order, ax=ax)
            ax.set_title(f"Répartition de {col}")
            ax.tick_params(axis="x", rotation=45)

    # D : scatter Premium vs N_claims_year
    if {"Premium", "N_claims_year"}.issubset(df.columns):
        ax = mosaic["D"]
        sns.scatterplot(data=df, x="Premium", y="N_claims_year", alpha=0.3, ax=ax)
        ax.set_title("Premium vs N_claims_year")

    # E : scatter Premium vs Cost_claims_year
    if {"Premium", "Cost_claims_year"}.issubset(df.columns):
        ax = mosaic["E"]
        sns.scatterplot(data=df, x="Premium", y="Cost_claims_year", alpha=0.3, ax=ax)
        ax.set_title("Premium vs Cost_claims_year")

    # F : average premium for Type_risk or Area
    if {"Premium", "Type_risk"}.issubset(df.columns):
        col_group = "Type_risk"
    elif {"Premium", "Area"}.issubset(df.columns):
        col_group = "Area"
    else:
        col_group = None

    if col_group is not None:
        ax = mosaic["F"]
        avg_prem = (
            df.groupby(col_group)["Premium"]
            .mean()
            .sort_values(ascending=False)
            .head(15)
        )
        sns.barplot(x=avg_prem.index, y=avg_prem.values, ax=ax)
        ax.set_title(f"Average premium for {col_group}")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(path, dpi=150)


def premium_distribution(path, df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(14, 8))
    mosaic3 = fig.subplot_mosaic(
        """
        A
        """,
        gridspec_kw={"height_ratios": [1]},
    )

    ax = mosaic3["A"]
    sns.histplot(df["Premium"], kde=True, bins=40, ax=ax)
    ax.set_title(f"Premium Distribution")
    fig.savefig(path, dpi=150)
