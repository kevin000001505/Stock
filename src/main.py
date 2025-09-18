from auth import check_authentication, show_logout_button
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from stock import StockDataFetcher
import datetime
import time
import config


BASE_COLUMNS = ["代號", "名稱", "產業別"]

check_authentication()

st.set_page_config(page_title="Stock Data Analysis", layout="wide")

show_logout_button()

# Initialize the StockDataFetcher
stock_fetcher = StockDataFetcher()

st.sidebar.header("參數調整")

# Stock type selection
stock_type = st.sidebar.radio("市場", ["上市", "上櫃"])

# Number of extrema to find
num_extrema = st.sidebar.slider("顯示底或頂數量", 2, 10, 2)

# Extrema type - Modified to include "Both" option
extrema_display = st.sidebar.radio("篩選方式", ["底谷", "頂峰", "兩者"])

# Map Chinese to English for internal use
extrema_mapping = {"底谷": "Minima", "頂峰": "Maxima", "兩者": "Both"}

extrema_type = extrema_mapping[extrema_display]

# Reverse option - Changed to radio button with Chinese labels
reverse_display = st.sidebar.radio("比較方向", ["做多", "做空"])

# Map Chinese to English/Boolean for internal use
reverse_option = True if reverse_display == "做空" else False

# Moving Average controls in sidebar
st.sidebar.header("移動平均數")
window_size = st.sidebar.slider("天數", 1, 10, 1)

# Add data display range control
st.sidebar.header("資料顯示範圍")
display_days = st.sidebar.slider("顯示天數", 5, 100, 20, help="選擇要顯示的資料天數")


def get_today_date():
    """Get today's date in M/D format (cross-platform)"""
    today = datetime.datetime.now()
    return f"{today.month:02d}/{today.day:02d}"


# Get current date dynamically
current_date = get_today_date()

# Check if current date data already exists
if stock_type == "上市":
    market_df = stock_fetcher.listed_data
else:
    market_df = stock_fetcher.counter_data

has_current_data = current_date in market_df.columns

# Button to fetch latest data
if st.sidebar.button("獲取今日股票資料"):
    if has_current_data:
        st.info(f"Data for {current_date} already exists. No fetch needed.")
    else:
        with st.spinner("獲取最新股票資訊..."):
            try:
                if stock_type == "上市":
                    market_df = stock_fetcher.fetch_stock_data(market_df, "listed")
                else:
                    market_df = stock_fetcher.fetch_stock_data(market_df, "counter")

                st.success("資料更新成功！")
                # Force app to rerun and recalculate all variables
                st.rerun()

            except Exception as e:
                st.error(f"獲取資料時發生錯誤: {str(e)}")

# Display current data status
if has_current_data:
    st.sidebar.success(f"✅ Data for {current_date} is available")
else:
    st.sidebar.warning(f"⚠️ Data for {current_date} is missing")

# Show data overview with filtered dataframe based on extrema results
st.header(f"{stock_type} 股票數據概覽")


def prepare_series(
    values: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    display_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply moving average and trim to the requested display window."""

    processed_values = values
    processed_labels = labels

    if window_size > 1 and values.size >= window_size:
        kernel = np.ones(window_size) / window_size
        processed_values = np.convolve(values, kernel, mode="valid")
        processed_labels = labels[window_size - 1 :]

    if display_days and processed_values.size > display_days:
        processed_values = processed_values[-display_days:]
        processed_labels = processed_labels[-display_days:]

    return processed_values, processed_labels


category_options = (
    sorted(market_df["產業別"].dropna().unique()) if "產業別" in market_df.columns else []
)

selected_categories = st.sidebar.multiselect(
    "產業別",
    options=category_options,
    default=category_options,
    placeholder="選擇欲分析的產業別",
    help="依照股票產業別篩選資料，預設為全部產業",
)

if selected_categories:
    filtered_market_df = market_df[market_df["產業別"].isin(selected_categories)].copy()
else:
    filtered_market_df = market_df.iloc[0:0].copy()

filtered_market_df.reset_index(drop=True, inplace=True)

display_df = pd.DataFrame(columns=BASE_COLUMNS)
price_columns = [col for col in filtered_market_df.columns if col not in BASE_COLUMNS]
total_considered = len(filtered_market_df)
has_category_selection = bool(selected_categories)

if total_considered and price_columns:
    numeric_prices = filtered_market_df.loc[:, price_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    price_matrix = numeric_prices.to_numpy(dtype=float)
    price_labels = np.array(price_columns)

    filtered_records: list[dict] = []

    with st.spinner("Analyzing all stocks for positive trends..."):
        for row_idx in range(total_considered):
            row_prices = price_matrix[row_idx]
            valid_mask = ~np.isnan(row_prices)

            if valid_mask.sum() < 3:
                continue

            valid_values = row_prices[valid_mask]
            valid_labels = price_labels[valid_mask]

            processed_values, processed_labels = prepare_series(
                valid_values, valid_labels, window_size, display_days
            )

            if processed_values.size < 3:
                continue

            series_for_analysis = processed_values.tolist()

            try:
                if extrema_type == "Both":
                    minima_result, _ = stock_fetcher.find_local_extrema(
                        series_for_analysis,
                        find_type="minima",
                        number_of_extrema=num_extrema,
                        reverse=reverse_option,
                    )
                    maxima_result, _ = stock_fetcher.find_local_extrema(
                        series_for_analysis,
                        find_type="maxima",
                        number_of_extrema=num_extrema,
                        reverse=reverse_option,
                    )
                    matches_filter = minima_result and maxima_result
                else:
                    matches_filter, _ = stock_fetcher.find_local_extrema(
                        series_for_analysis,
                        find_type=extrema_type.lower(),
                        number_of_extrema=num_extrema,
                        reverse=reverse_option,
                    )
            except Exception:
                continue

            if not matches_filter:
                continue

            stock_info = filtered_market_df.loc[row_idx, BASE_COLUMNS].to_dict()
            record = {**stock_info}
            record.update(
                {
                    label: round(value, 2)
                    for label, value in zip(processed_labels.tolist(), processed_values)
                }
            )
            filtered_records.append(record)

    if filtered_records:
        display_df = pd.DataFrame(filtered_records)

        ordered_date_columns = [
            col for col in price_columns if col in display_df.columns and col not in BASE_COLUMNS
        ]

        if display_days and len(ordered_date_columns) > display_days:
            ordered_date_columns = ordered_date_columns[-display_days:]

        final_columns = BASE_COLUMNS + ordered_date_columns
        display_df = display_df.reindex(columns=final_columns)
        display_df.sort_values(by="代號", inplace=True)
        display_df.reset_index(drop=True, inplace=True)
        display_df["代號"] = display_df["代號"].astype(str)
    else:
        display_df = pd.DataFrame(columns=BASE_COLUMNS)
else:
    if not has_category_selection and category_options:
        st.sidebar.warning("⚠️ 請選擇至少一個產業別進行篩選")

if not display_df.empty:
    st.success(
        f"Found {len(display_df)} stocks with positive trends out of {total_considered} total stocks"
    )
else:
    if not has_category_selection and not category_options:
        st.info("🚫 此市場尚未提供產業別資料")
    elif not has_category_selection:
        st.info("👈 在側邊欄選擇欲分析的產業別後即可顯示結果")
    elif total_considered == 0:
        st.error("❌ 選定的產業別目前沒有可用股票資料")
    else:
        st.error("❌ 沒有股票符合您設定的條件")

        # Show what the user was looking for
        st.info("**您的篩選條件:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"- 市場: **{stock_type}**")
            st.write(f"- 篩選方式: **{extrema_display}**")
            st.write(f"- 比較方向: **{reverse_display}**")
            st.write(
                "- 產業別: "
                + ("全部" if len(selected_categories) == len(category_options) else ", ".join(selected_categories))
            )
        with col2:
            st.write(f"- 底/頂數量: **{num_extrema}**")
            st.write(f"- 移動平均: **{window_size} 天**")
            st.write(f"- 顯示範圍: **{display_days} 天**")

        display_df = pd.DataFrame(columns=BASE_COLUMNS)

st.write(f"股票數: {len(display_df)}")

# Display the filtered dataframe with scrolling capability
if len(display_df) > 0:
    # Add download button for filtered data
    if not display_df.empty:
        price_cols_for_download = [
            col for col in display_df.columns if col not in BASE_COLUMNS
        ]

        if price_cols_for_download:
            download_columns = ["代號", "名稱", price_cols_for_download[-1]]
        else:
            download_columns = ["代號", "名稱", "產業別"]

        download_df = display_df.loc[:, download_columns].copy()

        download_df["代號"] = download_df["代號"].astype(str)
        # Convert to CSV
        csv = download_df.to_csv(index=False, encoding="utf-8-sig")

        st.download_button(
            label="📥 下載當日符合股票 (CSV)",
            data=csv.encode("utf-8-sig"),  # Add this encoding
            file_name=f"{stock_type}_{current_date}.csv",
            mime="text/csv; charset=utf-8",  # Specify charset
            help="下載包含股票代號、名稱和最新價格的資料",
        )

    # Make the dataframe clickable
    event = st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
        on_select="rerun",
        selection_mode="single-row",
    )

    # Get the selected stock from dataframe click
    selected_stock_from_df = None
    if event.selection and len(event.selection.rows) > 0:
        selected_row_index = event.selection.rows[0]
        selected_stock_from_df = display_df.iloc[selected_row_index]["代號"]
else:
    if has_category_selection and total_considered > 0:
        st.info("🚫 沒有股票符合您的篩選條件")
    elif not has_category_selection:
        st.info("👈 在側邊欄選擇產業別後即可顯示股票清單")
    else:
        st.info("🚫 此市場暫時沒有可顯示的股票資料")

# Stock filter (now works on the already filtered data)
st.header("股票分析")
stock_filter = st.text_input("Filter stocks by code or name", "")

filtered_df = display_df
if stock_filter and len(display_df) > 0:
    filtered_df = display_df[
        display_df["代號"].str.contains(stock_filter, na=False)
        | display_df["名稱"].str.contains(stock_filter, na=False)
    ]

if not filtered_df.empty:
    # Use selected stock from dataframe click or selectbox
    if (
        selected_stock_from_df
        and selected_stock_from_df in filtered_df["代號"].tolist()
    ):
        selected_stock = selected_stock_from_df
        # Update selectbox to show the clicked stock
        stock_options = filtered_df["代號"].tolist()
        default_index = (
            stock_options.index(selected_stock)
            if selected_stock in stock_options
            else 0
        )
        selected_stock = st.selectbox(
            "Select a Stock", stock_options, index=default_index
        )
    else:
        # Allow user to select a stock normally
        selected_stock = st.selectbox("Select a Stock", filtered_df["代號"].tolist())

    # Get the selected stock data - USE THE SAME RAW DATA SOURCE
    stock_row = market_df[market_df["代號"] == selected_stock].iloc[
        0
    ]
    stock_name = stock_row["名稱"]

    st.subheader(f"{selected_stock} - {stock_name}")

    # Get the price data columns from original df
    price_columns = [col for col in market_df.columns if col not in BASE_COLUMNS]

    # Extract RAW price data for the selected stock
    raw_price_series = pd.to_numeric(
        stock_row[price_columns], errors="coerce"
    )
    price_array = raw_price_series.to_numpy(dtype=float)
    price_labels = np.array(price_columns)

    valid_mask = ~np.isnan(price_array)
    valid_values = price_array[valid_mask]
    valid_labels = price_labels[valid_mask]

    # Find local extrema
    if valid_values.size >= 3:
        use_moving_average = window_size > 1 and valid_values.size >= window_size
        processed_values, processed_labels = prepare_series(
            valid_values, valid_labels, window_size, display_days
        )

        if processed_values.size < 3:
            st.warning("Not enough processed price data for analysis after trimming.")
        else:
            data_to_analyze = processed_values.tolist()
            labels_to_use = processed_labels.tolist()
            line_label = (
                f"Moving Average ({window_size})" if use_moving_average else "Price"
            )
            line_color = "orange" if use_moving_average else "steelblue"

            # Now the extrema detection uses the SAME processed data as filtering
            if extrema_type == "Both":
                # Find minima on filtered data
                minima_result, minima_locations = stock_fetcher.find_local_extrema(
                    data_to_analyze,
                    find_type="minima",
                    number_of_extrema=num_extrema,
                    reverse=reverse_option,
                )

                # Find maxima on filtered data
                maxima_result, maxima_locations = stock_fetcher.find_local_extrema(
                    data_to_analyze,
                    find_type="maxima",
                    number_of_extrema=num_extrema,
                    reverse=reverse_option,
                )

                # Line plot with highlighted extrema points
                plt.style.use("seaborn-v0_8")
                fig, ax = plt.subplots(figsize=(14, 8))

                # Create x-axis positions based on filtered data length - FIX HERE
                x_positions = range(len(data_to_analyze))
                x_labels = labels_to_use[: len(data_to_analyze)]  # Ensure same length

                sns.lineplot(
                    x=x_positions,
                    y=data_to_analyze,
                    marker="o",
                    linewidth=3,
                    markersize=8,
                    color=line_color,
                    alpha=0.8,
                    ax=ax,
                    label=line_label,
                )

                # Highlight minima points if found
                if minima_locations:
                    minima_values = [data_to_analyze[loc] for loc in minima_locations]
                    ax.scatter(
                        minima_locations,
                        minima_values,
                        color="red",
                        s=200,
                        marker="v",
                        label="Local Minima",
                        zorder=5,
                        edgecolor="darkred",
                        linewidth=2,
                    )

                    # Draw connecting line between minima points
                    if len(minima_locations) > 1:
                        ax.plot(
                            minima_locations,
                            minima_values,
                            color="red",
                            linestyle="--",
                            linewidth=2,
                            alpha=0.7,
                            label="Minima Trend Line",
                        )

                    # Add annotations for minima with price values
                    for i, loc in enumerate(minima_locations):
                        ax.annotate(
                            f"{data_to_analyze[loc]:.2f}",
                            xy=(loc, data_to_analyze[loc]),
                            xytext=(0, -20),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.8),
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                            fontsize=12,
                            color="white",
                            weight="bold",
                            ha="center",
                        )

                # Highlight maxima points if found
                if maxima_locations:
                    maxima_values = [data_to_analyze[loc] for loc in maxima_locations]
                    ax.scatter(
                        maxima_locations,
                        maxima_values,
                        color="green",
                        s=200,
                        marker="^",
                        label="Local Maxima",
                        zorder=5,
                        edgecolor="darkgreen",
                        linewidth=2,
                    )

                    # Draw connecting line between maxima points
                    if len(maxima_locations) > 1:
                        ax.plot(
                            maxima_locations,
                            maxima_values,
                            color="green",
                            linestyle="--",
                            linewidth=2,
                            alpha=0.7,
                            label="Maxima Trend Line",
                        )

                    # Add annotations for maxima with price values
                    for i, loc in enumerate(maxima_locations):
                        ax.annotate(
                            f"{data_to_analyze[loc]:.2f}",
                            xy=(loc, data_to_analyze[loc]),
                            xytext=(0, 20),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.3", fc="green", alpha=0.8),
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                            fontsize=12,
                            color="white",
                            weight="bold",
                            ha="center",
                        )

                ax.set_xlabel("Date", fontsize=14, fontweight="bold")
                ax.set_ylabel("Price", fontsize=14, fontweight="bold")

                # Set x-axis labels - now correctly sized for filtered data
                ax.set_xticks(x_positions)
                ax.set_xticklabels(x_labels, rotation=45, ha="right")

                # Add grid with better styling
                ax.grid(True, alpha=0.3, linestyle="--")

                # Add legend if there are extrema points
                if minima_locations or maxima_locations:
                    ax.legend(fontsize=12, loc="upper left")

                # Add subtle background color
                ax.set_facecolor("#f8f9fa")

                # Tight layout to prevent label cutoff
                plt.tight_layout()

                st.pyplot(fig)

                # Show extrema values for both (using correct labels)
                st.subheader("詳細資訊")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Minima Details:**")
                    if minima_locations:
                        minima_data = []
                        for i, loc in enumerate(minima_locations):
                            minima_data.append(
                                {
                                    "Date": labels_to_use[loc],
                                    "Position": loc,
                                    "Value": data_to_analyze[loc],
                                    "Type": "Minimum",
                                }
                            )
                        minima_df = pd.DataFrame(minima_data)
                        st.dataframe(minima_df, use_container_width=True)
                    else:
                        st.info("No minima found")

                with col2:
                    st.write("**Maxima Details:**")
                    if maxima_locations:
                        maxima_data = []
                        for i, loc in enumerate(maxima_locations):
                            maxima_data.append(
                                {
                                    "Date": labels_to_use[loc],
                                    "Position": loc,
                                    "Value": data_to_analyze[loc],
                                    "Type": "Maximum",
                                }
                            )
                        maxima_df = pd.DataFrame(maxima_data)
                        st.dataframe(maxima_df, use_container_width=True)
                    else:
                        st.info("No maxima found")

            else:
                # Single extrema type visualization
                result, locations = stock_fetcher.find_local_extrema(
                    data_to_analyze,
                    find_type=extrema_type.lower(),
                    number_of_extrema=num_extrema,
                    reverse=reverse_option,
                )

                if result is not None and locations:  # Check for valid results
                    plt.style.use("seaborn-v0_8")
                    fig, ax = plt.subplots(figsize=(14, 8))

                    # Create x-axis positions based on filtered data length - FIX HERE
                    x_positions = range(len(data_to_analyze))
                    x_labels = labels_to_use[: len(data_to_analyze)]  # Ensure same length

                    sns.lineplot(
                        x=x_positions,
                        y=data_to_analyze,
                        marker="o",
                        linewidth=3,
                        markersize=8,
                        color=line_color,
                        alpha=0.8,
                        ax=ax,
                        label=line_label,
                    )

                    # Highlight the extrema points
                    if locations:
                        extrema_values = [data_to_analyze[loc] for loc in locations]
                        if extrema_type == "Minima":
                            ax.scatter(
                                locations,
                                extrema_values,
                                color="red",
                                s=200,
                                marker="v",
                                label="Local Minima",
                                zorder=5,
                                edgecolor="darkred",
                                linewidth=2,
                            )

                            # Draw connecting line between minima points with dashed line
                            if len(locations) > 1:
                                ax.plot(
                                    locations,
                                    extrema_values,
                                    color="red",
                                    linestyle="--",
                                    linewidth=2,
                                    alpha=0.7,
                                    label="Minima Trend Line",
                                )

                            # Add price annotations for minima
                            for i, loc in enumerate(locations):
                                ax.annotate(
                                    f"{data_to_analyze[loc]:.2f}",
                                    xy=(loc, data_to_analyze[loc]),
                                    xytext=(0, -20),
                                    textcoords="offset points",
                                    bbox=dict(
                                        boxstyle="round,pad=0.3", fc="red", alpha=0.8
                                    ),
                                    arrowprops=dict(
                                        arrowstyle="->", connectionstyle="arc3,rad=0"
                                    ),
                                    fontsize=12,
                                    color="white",
                                    weight="bold",
                                    ha="center",
                                )

                        else:
                            ax.scatter(
                                locations,
                                extrema_values,
                                color="green",
                                s=200,
                                marker="^",
                                label="Local Maxima",
                                zorder=5,
                                edgecolor="darkgreen",
                                linewidth=2,
                            )

                            # Draw connecting line between maxima points with dashed line
                            if len(locations) > 1:
                                ax.plot(
                                    locations,
                                    extrema_values,
                                    color="green",
                                    linestyle="--",
                                    linewidth=2,
                                    alpha=0.7,
                                    label="Maxima Trend Line",
                                )

                            # Add price annotations for maxima
                            for i, loc in enumerate(locations):
                                ax.annotate(
                                    f"{data_to_analyze[loc]:.2f}",
                                    xy=(loc, data_to_analyze[loc]),
                                    xytext=(0, 20),
                                    textcoords="offset points",
                                    bbox=dict(
                                        boxstyle="round,pad=0.3", fc="green", alpha=0.8
                                    ),
                                    arrowprops=dict(
                                        arrowstyle="->", connectionstyle="arc3,rad=0"
                                    ),
                                    fontsize=12,
                                    color="white",
                                    weight="bold",
                                    ha="center",
                                )

                    ax.set_xlabel("Date", fontsize=14, fontweight="bold")
                    ax.set_ylabel("Price", fontsize=14, fontweight="bold")

                    # Set x-axis labels - now correctly sized for filtered data
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels(x_labels, rotation=45, ha="right")

                    # Add grid with better styling
                    ax.grid(True, alpha=0.3, linestyle="--")

                    # Add legend
                    ax.legend(fontsize=12, loc="upper left")

                    # Add subtle background color
                    ax.set_facecolor("#f8f9fa")

                    # Tight layout to prevent label cutoff
                    plt.tight_layout()

                    st.pyplot(fig)

                    # Show extrema values (using correct labels)
                    st.subheader("詳細資訊")
                    extrema_data = []
                    for i, loc in enumerate(locations):
                        extrema_data.append(
                            {
                                "日期": labels_to_use[loc],
                                # "Position": loc,
                                "價格": data_to_analyze[loc],
                                "篩選方式": extrema_display,
                            }
                        )

                    extrema_df = pd.DataFrame(extrema_data)
                    st.dataframe(extrema_df, use_container_width=True)
                else:
                    st.warning(f"❌ 無法找到足夠的{extrema_display}進行分析")
                    st.info(f"此股票在設定條件下找不到 {num_extrema} 個{extrema_display}")
    else:
        st.warning("Not enough price data for this stock to perform analysis.")
else:
    st.info("🚫 沒有股票符合您的篩選條件")

    # Show current parameter settings for user reference
    st.markdown("**目前設定:**")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"- 市場: **{stock_type}**")
        st.write(f"- 篩選方式: **{extrema_display}**")
        st.write(f"- 比較方向: **{reverse_display}**")
    with col2:
        st.write(f"- 底/頂數量: **{num_extrema}**")
        st.write(f"- 移動平均: **{window_size} 天**")
        st.write(f"- 顯示範圍: **{display_days} 天**")

# Add CSV Editor Section in Sidebar
st.sidebar.header("📝 資料管理")

with st.sidebar.expander("刪除日期範圍資料"):
    st.info("刪除指定日期範圍的股價資料欄位")

    # Get current year for date context
    current_year = datetime.datetime.now().year

    # Date range input with date picker
    start_date_picker = st.date_input(
        "開始日期", value=datetime.date(current_year, 1, 1), help="選擇開始日期"
    )

    end_date_picker = st.date_input(
        "結束日期", value=datetime.date(current_year, 4, 1), help="選擇結束日期"
    )

    if start_date_picker and end_date_picker:
        # Convert to MM/DD format for matching
        start_date_str = (
            start_date_picker.strftime("%m/%d").lstrip("0").replace("/0", "/")
        )
        end_date_str = end_date_picker.strftime("%m/%d").lstrip("0").replace("/0", "/")

        # Get current dataframe
        current_df = (
            stock_fetcher.listed_data
            if stock_type == "上市"
            else stock_fetcher.counter_data
        )

        # Get date columns
        date_columns = [
            col for col in current_df.columns if col not in BASE_COLUMNS
        ]

        # Filter columns by date range
        columns_to_delete = []
        for col in date_columns:
            try:
                # Parse column date
                col_parts = col.split("/")
                if len(col_parts) == 2:
                    col_month, col_day = int(col_parts[0]), int(col_parts[1])
                    col_date = datetime.date(current_year, col_month, col_day)

                    # Check if in range
                    if start_date_picker <= col_date <= end_date_picker:
                        columns_to_delete.append(col)

            except (ValueError, IndexError):
                continue

        # Show preview and delete functionality
        if columns_to_delete:
            st.warning(f"⚠️ 將刪除 {len(columns_to_delete)} 個日期欄位")

            # Show first few columns as preview
            preview_cols = columns_to_delete[:5]
            if len(columns_to_delete) > 5:
                st.write(
                    f"📋 預覽: {', '.join(preview_cols)}... (+{len(columns_to_delete)-5} 更多)"
                )
            else:
                st.write(f"📋 將刪除: {', '.join(preview_cols)}")

            if st.button("🗑️ 確認刪除", type="secondary"):
                with st.spinner("正在刪除欄位..."):
                    try:
                        # Delete and save
                        if stock_type == "上市":
                            stock_fetcher.listed_data = stock_fetcher.listed_data.drop(
                                columns=columns_to_delete
                            )
                            stock_fetcher.listed_data.to_csv(
                                config.LISTED_CSV, index=False, encoding="utf-8-sig"
                            )
                        else:
                            stock_fetcher.counter_data = (
                                stock_fetcher.counter_data.drop(
                                    columns=columns_to_delete
                                )
                            )
                            stock_fetcher.counter_data.to_csv(
                                config.COUNTER_CSV, index=False, encoding="utf-8-sig"
                            )

                        st.success(f"✅ 成功刪除 {len(columns_to_delete)} 個欄位！")
                        time.sleep(1)
                        st.rerun()

                    except Exception as e:
                        st.error(f"刪除失敗: {str(e)}")
        else:
            st.info("📅 指定範圍內無匹配的日期欄位")

st.sidebar.markdown("---")
