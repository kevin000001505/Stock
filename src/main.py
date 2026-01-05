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
import json
import re
import pytz
from pathlib import Path


BASE_COLUMNS = ["代號", "名稱", "產業別"]
# Keep auto-fetch related configuration together
AUTO_FETCH_LOG_PATH = Path("auto_fetch_log.json")
AUTO_FETCH_HOUR = 18
AUTO_FETCH_MAX_ATTEMPTS = 3
AUTO_FETCH_TIMEZONE = pytz.timezone("Asia/Taipei")
USER_PREFS_PATH = Path("user_preferences.json")


def _read_user_preferences_store() -> dict:
    if USER_PREFS_PATH.exists():
        try:
            with USER_PREFS_PATH.open("r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def load_user_preferences(username: str) -> dict:
    if not username:
        return {}
    store = _read_user_preferences_store()
    prefs = store.get(username, {})
    return prefs.copy() if isinstance(prefs, dict) else {}


def save_user_preferences(username: str, preferences: dict) -> None:
    if not username:
        return
    store = _read_user_preferences_store()
    store[username] = preferences
    try:
        with USER_PREFS_PATH.open("w", encoding="utf-8") as file:
            json.dump(store, file, ensure_ascii=False, indent=2)
    except OSError:
        pass


check_authentication()

st.set_page_config(page_title="Stock Data Analysis", layout="wide")

current_username = st.session_state.get("username", "")
user_prefs = load_user_preferences(current_username)

stock_type_options = ["上市", "上櫃"]
stock_type_default = user_prefs.get("stock_type", stock_type_options[0])
if stock_type_default not in stock_type_options:
    stock_type_default = stock_type_options[0]

extrema_options = ["底谷", "頂峰", "兩者"]
extrema_default = user_prefs.get("extrema_display", extrema_options[0])
if extrema_default not in extrema_options:
    extrema_default = extrema_options[0]

reverse_options = ["做多", "做空"]
reverse_default = user_prefs.get("reverse_display", reverse_options[0])
if reverse_default not in reverse_options:
    reverse_default = reverse_options[0]


def _clamp_int(value, minimum, maximum, fallback):
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, int_value))


num_extrema_default = _clamp_int(user_prefs.get("num_extrema", 2), 2, 10, 2)
window_size_default = _clamp_int(user_prefs.get("window_size", 1), 1, 10, 1)
display_days_default = _clamp_int(user_prefs.get("display_days", 20), 5, 100, 20)

selected_categories_map = user_prefs.get("selected_categories", {})
if not isinstance(selected_categories_map, dict):
    selected_categories_map = {}

observed_default_codes = user_prefs.get("observed_codes", [])
if not isinstance(observed_default_codes, list):
    observed_default_codes = []
observed_default_text = "\n".join(observed_default_codes)

# Initialize the StockDataFetcher
stock_fetcher = StockDataFetcher()

st.sidebar.header("參數調整")

# Stock type selection
stock_type = st.sidebar.radio(
    "市場",
    stock_type_options,
    index=stock_type_options.index(stock_type_default),
)

# Number of extrema to find
num_extrema = st.sidebar.slider("顯示底或頂數量", 2, 10, num_extrema_default)

# Extrema type - Modified to include "Both" option
extrema_display = st.sidebar.radio(
    "篩選方式",
    extrema_options,
    index=extrema_options.index(extrema_default),
)

# Map Chinese to English for internal use
extrema_mapping = {"底谷": "Minima", "頂峰": "Maxima", "兩者": "Both"}

extrema_type = extrema_mapping[extrema_display]

# Reverse option - Changed to radio button with Chinese labels
reverse_display = st.sidebar.radio(
    "比較方向",
    reverse_options,
    index=reverse_options.index(reverse_default),
)

# Map Chinese to English/Boolean for internal use
reverse_option = True if reverse_display == "做空" else False

# Moving Average controls in sidebar
st.sidebar.header("移動平均數")
window_size = st.sidebar.slider("天數", 1, 10, window_size_default)

# Add data display range control
st.sidebar.header("資料顯示範圍")
display_days = st.sidebar.slider(
    "顯示天數",
    5,
    100,
    display_days_default,
    help="選擇要顯示的資料天數",
)


def get_today_date():
    """Get today's date (Taipei time) in MM/DD format."""
    today = datetime.datetime.now(AUTO_FETCH_TIMEZONE)
    return f"{today.month:02d}/{today.day:02d}"


def load_auto_fetch_log() -> dict:
    if AUTO_FETCH_LOG_PATH.exists():
        try:
            with AUTO_FETCH_LOG_PATH.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (json.JSONDecodeError, OSError):
            data = {}
    else:
        data = {}

    data.setdefault("listed", {})
    data.setdefault("counter", {})
    return data


def save_auto_fetch_log(log: dict) -> None:
    try:
        with AUTO_FETCH_LOG_PATH.open("w", encoding="utf-8") as file:
            json.dump(log, file, ensure_ascii=False, indent=2)
    except OSError:
        pass


def maybe_auto_fetch(
    market_key: str,
    market_df: pd.DataFrame,
    current_date_label: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Attempt to auto-fetch data after 18:00 with retry tracking."""

    now = datetime.datetime.now(AUTO_FETCH_TIMEZONE)
    messages: list[str] = []

    if now.hour < AUTO_FETCH_HOUR:
        return market_df, messages

    today_key = now.date().isoformat()
    log = load_auto_fetch_log()
    market_log = log.setdefault(market_key, {})
    entry = market_log.get(today_key, {"attempts": 0})

    if current_date_label in market_df.columns:
        if today_key in market_log:
            del market_log[today_key]
            save_auto_fetch_log(log)
        return market_df, messages

    if entry.get("status") == "skipped":
        messages.append("今日自動更新已跳過。")
        return market_df, messages

    attempts = entry.get("attempts", 0)

    if attempts >= AUTO_FETCH_MAX_ATTEMPTS:
        entry["status"] = "skipped"
        market_log[today_key] = entry
        save_auto_fetch_log(log)
        messages.append("今日自動更新已達嘗試上限，跳過。")
        return market_df, messages

    last_attempt = entry.get("last_attempt")
    if last_attempt:
        try:
            last_attempt_dt = datetime.datetime.fromisoformat(last_attempt)
        except ValueError:
            last_attempt_dt = None
        if last_attempt_dt and now - last_attempt_dt < datetime.timedelta(hours=1):
            wait_minutes = 60 - int((now - last_attempt_dt).total_seconds() // 60)
            messages.append(f"等待重新嘗試自動更新（約 {wait_minutes} 分鐘後）。")
            return market_df, messages

    try:
        updated_df = stock_fetcher.fetch_stock_data(market_df, market_key)
    except Exception as exc:  # noqa: BLE001
        attempts += 1
        entry.update(
            {
                "attempts": attempts,
                "last_attempt": now.isoformat(),
                "status": "error",
                "error": str(exc),
            }
        )
        market_log[today_key] = entry
        save_auto_fetch_log(log)
        messages.append(f"自動更新失敗：{exc}")

        if attempts >= AUTO_FETCH_MAX_ATTEMPTS:
            entry["status"] = "skipped"
            messages.append("已達自動更新上限，今日不再嘗試。")
        return market_df, messages

    if current_date_label in updated_df.columns:
        if today_key in market_log:
            del market_log[today_key]
        save_auto_fetch_log(log)
        messages.append("自動更新成功，已取得今日資料。")
        return updated_df, messages

    attempts += 1
    entry.update(
        {
            "attempts": attempts,
            "last_attempt": now.isoformat(),
            "status": "pending",
        }
    )
    market_log[today_key] = entry
    save_auto_fetch_log(log)
    messages.append("自動更新嘗試後仍無今日資料。")

    if attempts >= AUTO_FETCH_MAX_ATTEMPTS:
        entry["status"] = "skipped"
        messages.append("已達自動更新上限，今日不再嘗試。")

    return market_df, messages


# Get current date dynamically
current_date = get_today_date()

# Check if current date data already exists
if stock_type == "上市":
    market_df = stock_fetcher.listed_data
    market_key = "listed"
else:
    market_df = stock_fetcher.counter_data
    market_key = "counter"

market_df, auto_fetch_messages = maybe_auto_fetch(
    market_key=market_key,
    market_df=market_df,
    current_date_label=current_date,
)

has_current_data = current_date in market_df.columns
market_price_columns = [col for col in market_df.columns if col not in BASE_COLUMNS]
market_code_series = (
    market_df["代號"].astype(str)
    if "代號" in market_df.columns
    else pd.Series(dtype=str)
)

# Button to fetch latest data
if st.sidebar.button("獲取今日股票資料"):
    if has_current_data:
        st.info(f"Data for {current_date} already exists. No fetch needed.")
    else:
        # Create placeholders for progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_text = st.empty()
        detail_text = st.empty()

        def update_progress(
            message: str, current: int, total: int, estimated_remaining: float
        ):
            """Callback function to update progress UI"""
            progress = current / total if total > 0 else 0
            progress_bar.progress(progress)
            status_text.text(f"{message} ({current}/{total})")

            if estimated_remaining > 0:
                minutes = int(estimated_remaining // 60)
                seconds = int(estimated_remaining % 60)
                if minutes > 0:
                    time_text.text(f"⏱️ 預估剩餘時間: {minutes} 分 {seconds} 秒")
                else:
                    time_text.text(f"⏱️ 預估剩餘時間: {seconds} 秒")

            # Show processing rate
            if current > 0:
                detail_text.text(f"📊 處理進度: {progress*100:.1f}% | 每筆約 10 秒")

        try:
            market_df = stock_fetcher.fetch_stock_data(
                market_df, market_key, progress_callback=update_progress
            )

            progress_bar.progress(1.0)
            status_text.empty()
            time_text.empty()
            detail_text.empty()
            st.success("資料更新成功！")
            time.sleep(1)
            st.rerun()

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            time_text.empty()
            detail_text.empty()
            st.error(f"獲取資料時發生錯誤: {str(e)}")

# Display current data status
if has_current_data:
    st.sidebar.success(f"✅ Data for {current_date} is available")
else:
    st.sidebar.warning(f"⚠️ Data for {current_date} is missing")

for message in auto_fetch_messages:
    if any(keyword in message for keyword in ["失敗", "上限", "跳過"]):
        st.sidebar.warning(message)
    elif "成功" in message:
        st.sidebar.success(message)
    else:
        st.sidebar.info(message)


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


def analyze_stock_condition(
    values: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    display_days: int,
    extrema_type: str,
    num_extrema: int,
    reverse_option: bool,
) -> tuple[bool, np.ndarray, np.ndarray, dict[str, list[int]]]:
    processed_values, processed_labels = prepare_series(
        values, labels, window_size, display_days
    )

    if processed_values.size < 3:
        return False, processed_values, processed_labels, {}

    series_for_analysis = processed_values.tolist()
    extrema_details: dict[str, list[int]] = {}

    try:
        if extrema_type == "Both":
            minima_result, minima_locations = stock_fetcher.find_local_extrema(
                series_for_analysis,
                find_type="minima",
                number_of_extrema=num_extrema,
                reverse=reverse_option,
            )
            maxima_result, maxima_locations = stock_fetcher.find_local_extrema(
                series_for_analysis,
                find_type="maxima",
                number_of_extrema=num_extrema,
                reverse=reverse_option,
            )
            extrema_details["minima"] = minima_locations
            extrema_details["maxima"] = maxima_locations
            matches = (
                bool(minima_result)
                and bool(maxima_result)
                and bool(minima_locations)
                and bool(maxima_locations)
            )
        else:
            single_type = extrema_type.lower()
            result, locations = stock_fetcher.find_local_extrema(
                series_for_analysis,
                find_type=single_type,
                number_of_extrema=num_extrema,
                reverse=reverse_option,
            )
            extrema_details[single_type] = locations
            matches = bool(result) and bool(locations)
    except Exception:  # noqa: BLE001
        return False, processed_values, processed_labels, {}

    return matches, processed_values, processed_labels, extrema_details


category_options = (
    sorted(market_df["產業別"].dropna().unique())
    if "產業別" in market_df.columns
    else []
)

if stock_type in selected_categories_map:
    stored_categories_for_market = selected_categories_map[stock_type]
    if not isinstance(stored_categories_for_market, list):
        stored_categories_for_market = []
    preselected_categories = [
        category
        for category in stored_categories_for_market
        if category in category_options
    ]
    if not preselected_categories and stored_categories_for_market and category_options:
        preselected_categories = category_options
else:
    preselected_categories = category_options

selected_categories = st.sidebar.multiselect(
    "產業別",
    options=category_options,
    default=preselected_categories,
    placeholder="選擇欲分析的產業別",
    help="依照股票產業別篩選資料，預設為全部產業",
)

if selected_categories:
    filtered_market_df = market_df[market_df["產業別"].isin(selected_categories)].copy()
else:
    filtered_market_df = market_df.iloc[0:0].copy()

filtered_market_df.reset_index(drop=True, inplace=True)

st.sidebar.header("觀察股票")
observed_input = st.sidebar.text_area(
    "輸入股票代號清單",
    placeholder="例如: 2330\n2603, 0050",
    help="使用逗號、空白或換行分隔多個代號，可同時追蹤多檔股票",
    value=observed_default_text,
)

observed_codes: list[str] = []
if observed_input:
    tokens = re.split(r"[\s,;]+", observed_input)
    observed_codes = [token.strip() for token in tokens if token.strip()]
    observed_codes = list(dict.fromkeys(observed_codes))

if observed_codes:
    results: list[dict[str, str]] = []
    satisfied_count = 0

    for code in observed_codes:
        result = {"代號": code, "名稱": "-", "最新價格": "-", "狀態": "⚠️ 找不到股票"}

        if market_code_series.empty:
            results.append({**result, "狀態": "⚠️ 尚未載入市場資料"})
            continue

        match_mask = market_code_series == code

        if not match_mask.any():
            results.append(result)
            continue

        observed_row = market_df.loc[match_mask].iloc[0]
        stock_name_display = str(observed_row.get("名稱", ""))
        result["名稱"] = stock_name_display if stock_name_display else "-"

        if not market_price_columns:
            result["狀態"] = "⚠️ 無歷史價格資料"
            results.append(result)
            continue

        observed_prices = pd.to_numeric(
            observed_row[market_price_columns], errors="coerce"
        ).to_numpy(dtype=float)
        observed_labels = np.array(market_price_columns)
        valid_mask = ~np.isnan(observed_prices)

        if not valid_mask.any():
            result["狀態"] = "⚠️ 無有效價格"
            results.append(result)
            continue

        latest_price = observed_prices[valid_mask][-1]
        result["最新價格"] = (
            f"{latest_price:.2f}" if not np.isnan(latest_price) else "-"
        )

        if valid_mask.sum() < 3:
            result["狀態"] = "⌛ 資料不足"
            results.append(result)
            continue

        valid_values = observed_prices[valid_mask]
        valid_labels = observed_labels[valid_mask]
        matches, processed_values, _, _ = analyze_stock_condition(
            valid_values,
            valid_labels,
            window_size,
            display_days,
            extrema_type,
            num_extrema,
            reverse_option,
        )

        if processed_values.size < 3:
            result["狀態"] = "⌛ 資料不足"
        elif matches:
            result["狀態"] = "✅ 符合"
            satisfied_count += 1
        else:
            result["狀態"] = "⌛ 尚未符合"

        results.append(result)

    if results:
        st.sidebar.caption(f"符合條件：{satisfied_count} / {len(results)}")
        results_df = pd.DataFrame(results)
        display_height = min(400, max(200, 48 * len(results)))
        st.sidebar.dataframe(
            results_df,
            use_container_width=True,
            height=display_height,
        )
else:
    results_df = pd.DataFrame(columns=["代號", "名稱", "最新價格", "狀態"])

with st.sidebar.expander("偏好設定"):
    store_clicked = st.button("💾 儲存目前條件", use_container_width=True)

if store_clicked and current_username:
    updated_categories_map = dict(selected_categories_map)
    updated_categories_map[stock_type] = selected_categories

    new_preferences = dict(user_prefs)
    new_preferences.update(
        {
            "stock_type": stock_type,
            "num_extrema": num_extrema,
            "extrema_display": extrema_display,
            "reverse_display": reverse_display,
            "window_size": window_size,
            "display_days": display_days,
            "observed_codes": list(observed_codes),
            "selected_categories": updated_categories_map,
        }
    )

    save_user_preferences(current_username, new_preferences)
    st.sidebar.success("✅ 已儲存偏好設定")
    user_prefs = new_preferences
    selected_categories_map = updated_categories_map

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

            matches_filter, processed_values, processed_labels, _ = (
                analyze_stock_condition(
                    valid_values,
                    valid_labels,
                    window_size,
                    display_days,
                    extrema_type,
                    num_extrema,
                    reverse_option,
                )
            )

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
            col
            for col in price_columns
            if col in display_df.columns and col not in BASE_COLUMNS
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
                + (
                    "全部"
                    if len(selected_categories) == len(category_options)
                    else ", ".join(selected_categories)
                )
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
    stock_row = market_df.loc[market_code_series == selected_stock].iloc[0]
    stock_name = stock_row["名稱"]

    st.subheader(f"{selected_stock} - {stock_name}")

    # Get the price data columns from original df
    price_columns = market_price_columns

    # Extract RAW price data for the selected stock
    raw_price_series = pd.to_numeric(stock_row[price_columns], errors="coerce")
    price_array = raw_price_series.to_numpy(dtype=float)
    price_labels = np.array(price_columns)

    valid_mask = ~np.isnan(price_array)
    valid_values = price_array[valid_mask]
    valid_labels = price_labels[valid_mask]

    # Find local extrema
    if valid_values.size >= 3:
        use_moving_average = window_size > 1 and valid_values.size >= window_size
        condition_met, processed_values, processed_labels, extrema_info = (
            analyze_stock_condition(
                valid_values,
                valid_labels,
                window_size,
                display_days,
                extrema_type,
                num_extrema,
                reverse_option,
            )
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

            if extrema_type == "Both":
                minima_locations = extrema_info.get("minima", [])
                maxima_locations = extrema_info.get("maxima", [])

                if not condition_met:
                    st.warning("此股票目前尚未符合條件")

                if not minima_locations or not maxima_locations:
                    st.warning(f"❌ 無法找到足夠的{extrema_display}進行分析")
                    st.info(
                        f"此股票在設定條件下找不到 {num_extrema} 個{extrema_display}"
                    )
                else:
                    plt.style.use("seaborn-v0_8")
                    fig, ax = plt.subplots(figsize=(14, 8))

                    x_positions = range(len(data_to_analyze))
                    x_labels = labels_to_use[: len(data_to_analyze)]

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

                    for loc in minima_locations:
                        ax.annotate(
                            f"{data_to_analyze[loc]:.2f}",
                            xy=(loc, data_to_analyze[loc]),
                            xytext=(0, -20),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.8),
                            arrowprops=dict(
                                arrowstyle="->", connectionstyle="arc3,rad=0"
                            ),
                            fontsize=12,
                            color="white",
                            weight="bold",
                            ha="center",
                        )

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

                    for loc in maxima_locations:
                        ax.annotate(
                            f"{data_to_analyze[loc]:.2f}",
                            xy=(loc, data_to_analyze[loc]),
                            xytext=(0, 20),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.3", fc="green", alpha=0.8),
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
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels(x_labels, rotation=45, ha="right")
                    ax.grid(True, alpha=0.3, linestyle="--")
                    ax.legend(fontsize=12, loc="upper left")
                    ax.set_facecolor("#f8f9fa")
                    plt.tight_layout()

                    st.pyplot(fig)

                    st.subheader("詳細資訊")
                    col1, col2 = st.columns(2)

                    minima_data = [
                        {
                            "Date": labels_to_use[loc],
                            "Position": loc,
                            "Value": data_to_analyze[loc],
                            "Type": "Minimum",
                        }
                        for loc in minima_locations
                    ]
                    maxima_data = [
                        {
                            "Date": labels_to_use[loc],
                            "Position": loc,
                            "Value": data_to_analyze[loc],
                            "Type": "Maximum",
                        }
                        for loc in maxima_locations
                    ]

                    with col1:
                        if minima_data:
                            st.dataframe(
                                pd.DataFrame(minima_data), use_container_width=True
                            )
                        else:
                            st.info("No minima found")

                    with col2:
                        if maxima_data:
                            st.dataframe(
                                pd.DataFrame(maxima_data), use_container_width=True
                            )
                        else:
                            st.info("No maxima found")

            else:
                single_key = extrema_type.lower()
                locations = extrema_info.get(single_key, [])

                if not condition_met:
                    st.warning("此股票目前尚未符合條件")

                if not locations:
                    st.warning(f"❌ 無法找到足夠的{extrema_display}進行分析")
                    st.info(
                        f"此股票在設定條件下找不到 {num_extrema} 個{extrema_display}"
                    )
                else:
                    plt.style.use("seaborn-v0_8")
                    fig, ax = plt.subplots(figsize=(14, 8))

                    x_positions = range(len(data_to_analyze))
                    x_labels = labels_to_use[: len(data_to_analyze)]

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

                        for loc in locations:
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

                        for loc in locations:
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
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels(x_labels, rotation=45, ha="right")
                    ax.grid(True, alpha=0.3, linestyle="--")
                    ax.legend(fontsize=12, loc="upper left")
                    ax.set_facecolor("#f8f9fa")
                    plt.tight_layout()

                    st.pyplot(fig)

                    extrema_data = [
                        {
                            "日期": labels_to_use[loc],
                            "價格": data_to_analyze[loc],
                            "篩選方式": extrema_display,
                        }
                        for loc in locations
                    ]

                    st.subheader("詳細資訊")
                    st.dataframe(pd.DataFrame(extrema_data), use_container_width=True)
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
        # Get current dataframe
        current_df = (
            stock_fetcher.listed_data
            if stock_type == "上市"
            else stock_fetcher.counter_data
        )

        # Get date columns
        date_columns = [col for col in current_df.columns if col not in BASE_COLUMNS]

        # Filter columns by date range (matching YYYY/MM/DD or MM/DD format)
        columns_to_delete = []
        for col in date_columns:
            try:
                # Try parsing as YYYY/MM/DD format first
                try:
                    col_date = datetime.datetime.strptime(col, "%Y/%m/%d").date()
                except ValueError:
                    # If that fails, try MM/DD format and assume year 2025
                    col_date = datetime.datetime.strptime(
                        f"2025/{col}", "%Y/%m/%d"
                    ).date()

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
show_logout_button()
