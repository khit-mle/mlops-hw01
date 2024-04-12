import os

import numpy as np
import pandas as pd
import plotly.express as px

np.random.seed(42)


def generate_stock_data(
    ticker, start_price, end_price, noise_level, dip_magnitude, corr_high_tech, corr_twitter, corr_websearch, num_dips
):
    dates = pd.date_range(start="2022-01-01", end="2024-03-31", freq="D")

    base_price = np.linspace(start_price, end_price, len(dates)) + np.random.randn(len(dates)) * noise_level

    dip_indices = np.random.choice(len(dates), size=num_dips, replace=False)

    base_price[dip_indices] -= dip_magnitude

    df = pd.DataFrame({"date": dates, f"{ticker}_stock_price": base_price})

    pre_2022_dates = pd.date_range(start="2021-11-01", end="2021-12-31", freq="D")

    pre_2022_prices = (
        np.linspace(start_price - 10, start_price, len(pre_2022_dates)) + np.random.randn(len(pre_2022_dates)) * 2
    )

    df_pre_2022 = pd.DataFrame({"date": pre_2022_dates, f"{ticker}_stock_price": pre_2022_prices})

    df = pd.concat([df_pre_2022, df], ignore_index=True)

    df[f"{ticker}_stock_price_30d_moving_average"] = df[f"{ticker}_stock_price"].rolling(window=30).mean()

    high_tech_index = df[f"{ticker}_stock_price_30d_moving_average"] * corr_high_tech + np.random.randn(len(df)) * 5

    high_tech_index[dip_indices] -= dip_magnitude / 2

    df["high_tech_comp_index_30d_moving_average"] = high_tech_index.rolling(window=30).mean()

    twitter_sentiment = df[f"{ticker}_stock_price"] * corr_twitter + np.random.randn(len(df)) * 2

    twitter_sentiment[dip_indices] -= 0.5

    df["twitter_sentiment"] = (twitter_sentiment - twitter_sentiment.min()) / (
        twitter_sentiment.max() - twitter_sentiment.min()
    ) * 2 - 1

    web_search_interest = df[f"{ticker}_stock_price"] * corr_websearch + np.random.randn(len(df)) * 3

    web_search_interest[dip_indices] -= 0.3

    df["web_search_interest_trend"] = (web_search_interest - web_search_interest.min()) / (
        web_search_interest.max() - web_search_interest.min()
    )

    df = df[df["date"] >= "2022-01-01"]

    train_data = df[df["date"] < "2024-01-01"]

    test_data = df[df["date"] >= "2024-01-01"]

    return df, train_data, test_data


def save_plots(df, ticker):
    os.makedirs(f"plots/{ticker}", exist_ok=True)

    split_date = "2024-01-01"

    split_date_millis = pd.to_datetime(split_date).value / 1e6  # convert nanoseconds to milliseconds

    for col in df.columns[1:]:
        fig = px.line(df, x="date", y=col, title=f"{ticker} - {col}")
        fig.add_vline(
            x=split_date_millis,
            line_width=2,
            line_dash="dash",
            line_color="red",
            annotation_text="Train-Test Split",
            annotation_position="top right",
        )
        fig.update_layout(xaxis_range=[pd.to_datetime("2022-01-01"), pd.to_datetime("2024-03-31")])
        fig.write_image(f"plots/{ticker}/{col}.png", engine="kaleido")


# Ensure the directories exist
os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)

abc_data, abc_train, abc_test = generate_stock_data("ABC", 100, 200, 5, 20, 0.8, 0.7, 0.6, 10)
xyz_data, xyz_train, xyz_test = generate_stock_data("XYZ", 50, 150, 8, 15, 0.9, 0.6, 0.5, 12)

abc_train.to_csv("train/abc_train.csv", index=False)
abc_test.to_csv("test/abc_test.csv", index=False)

xyz_train.to_csv("train/xyz_train.csv", index=False)
xyz_test.to_csv("test/xyz_test.csv", index=False)

save_plots(abc_data, "ABC")
save_plots(xyz_data, "XYZ")
