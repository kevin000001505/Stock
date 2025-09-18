import config as config
import requests
import pandas as pd
from datetime import datetime, timedelta
from pyquery import PyQuery as pq
import config


class StockDataFetcher:
    def __init__(self):
        self.listed_data = pd.read_csv(config.LISTED_CSV, encoding="utf-8")
        self.counter_data = pd.read_csv(config.COUNTER_CSV, encoding="utf-8")
        self.today_date = datetime.now().strftime("%m/%d")
        self.session = requests.Session()
        self.session.cookies.update(config.COOKIES)
        self.session.headers.update(config.HEADERS)

    def fetch_stock_data(self, df: pd.DataFrame, market_type: str) -> pd.DataFrame:
        """
        Fetch stock data for the specified market type.

        Parameters:
        - df: DataFrame with existing stock data
        - market_type: "listed" for 上市 or "counter" for 上櫃

        Returns:
        - Updated DataFrame with new data
        """
        # Configuration mapping
        config_map = {
            "listed": {
                "params": config.LISTED_PARAMS.copy(),
                "csv_path": config.LISTED_CSV,
                "attr_name": "listed_data",
            },
            "counter": {
                "params": config.COUNTER_PARAMS.copy(),
                "csv_path": config.COUNTER_CSV,
                "attr_name": "counter_data",
            },
        }

        if market_type not in config_map:
            raise ValueError("market_type must be 'listed' or 'counter'")

        config_data = config_map[market_type]
        params = config_data["params"]
        csv_path = config_data["csv_path"]
        attr_name = config_data["attr_name"]

        latest_date = df.columns[-1]
        dates_list = self.generate_dates(f"{datetime.now().year}/{latest_date}")

        for date in dates_list:
            params["RPT_TIME"] = date
            response = self.session.post(config.URL, params=params)
            doc = pq(response.text)
            stock_date = (
                doc("table#tblStockList tr#row0 > td:nth-child(4)").text().strip()
            )
            if stock_date == "":
                continue  # Skip if no data for this date
            elif f"{datetime.now().year}/{stock_date}" != date:
                raise ValueError(
                    f"Date mismatch: expected {date}, got {datetime.now().year}/{stock_date}"
                )

            data_list = []
            for item in doc("table#tblStockList tr[id]").items():
                code = item.find("th:nth-child(1) > nobr > a").text().strip()
                price = item.find("td:nth-child(6) > nobr > a").text()
                if code and price:
                    data_list.append({"代號": code, stock_date: price})

            today_df = pd.DataFrame(data_list)
            df = df.merge(today_df, on="代號", how="left")

        # UPDATE CLASS ATTRIBUTE AND SAVE TO CSV
        setattr(self, attr_name, df)  # Dynamically set the class attribute
        df.to_csv(csv_path, index=False, encoding="utf-8")  # Save to file

        return df

    def generate_dates(self, start_date_str) -> list[str]:
        """
        Generate dates from the day after the given start date until today.
        Date format: YYYY/MM/DD

        Args:
            start_date_str (str): The start date in 'YYYY/MM/DD' format.

        Returns:
            list[str]: List of dates in 'YYYY/MM/DD' format.
        """
        start_date = datetime.strptime(start_date_str, "%Y/%m/%d")
        today = datetime.today()

        # Start from the next day
        current_date = start_date + timedelta(days=1)
        result = []

        while current_date <= today:
            result.append(current_date.strftime("%Y/%m/%d"))
            current_date += timedelta(days=1)

        return result

    def find_local_extrema(
        self, data_list: list, find_type="minima", number_of_extrema=2, reverse=False
    ) -> tuple[bool, list]:
        """
        Find the first n local extrema (minima or maxima) from right to left.

        Parameters:
        - data_list: list of numeric values
        - find_type: "minima" or "maxima"
        - number_of_extrema: how many extrema to find
        - reverse: if True, returns True when first extrema < second extrema
                if False, returns True when first extrema > second extrema

        Returns:
        - Tuple of (boolean comparison result, list of locations)
        """
        if len(data_list) < 3:
            return (False, [])

        local_extrema = []
        location = []

        # Set comparison based on extrema type
        if find_type == "minima":
            comparison = lambda current, left, right: current < left and current < right
        else:  # maxima
            comparison = lambda current, left, right: current > left and current > right

        # Start from the second-to-last element and go backwards
        for i in range(len(data_list) - 2, 0, -1):
            current = data_list[i]
            left_neighbor = data_list[i - 1]
            right_neighbor = data_list[i + 1]

            if comparison(current, left_neighbor, right_neighbor):
                if not local_extrema or current != local_extrema[-1]:
                    local_extrema.append(current)
                    location.append(i)

                    # Stop after finding the requested number
                    if len(local_extrema) == number_of_extrema:
                        return (
                            self.recognize_trend(
                                local_extrema,
                                reverse=reverse,
                                latest_price=data_list[-1],
                            ),
                            location,
                        )
        return (False, [])

    def recognize_trend(
        self, data_list: list, reverse=False, latest_price=None
    ) -> bool:
        """
        Recognize the trend based on local extrema.

        Parameters:
        - data_list: list of numeric values
        - find_type: "minima" or "maxima"
        - number_of_extrema: how many extrema to find
        - reverse: if True, returns "一底比一底低" or "一頂比一頂低"

        Returns:
        - String describing the trend
        """

        assert latest_price is not None, "latest_price must be provided"

        if not reverse:  # 一底比一底高 or 一頂比一頂高
            if data_list[0] > latest_price:
                return False  # Latest price is higher than the first local extrema
            for i in range(len(data_list) - 1):
                if data_list[i] < data_list[i + 1]:
                    return False
            return True
        else:  # 一底比一底低 or 一頂比一頂低
            if data_list[0] < latest_price:
                return False  # Latest price is lower than the first local extrema
            for i in range(len(data_list) - 1):
                if data_list[i] > data_list[i + 1]:
                    return False
            return True

    def mean_price_backwards(self, data_list: list, days=3) -> list:
        """
        Calculate mean prices going backwards (most recent first)
        """
        if not data_list or len(data_list) < days:
            return []

        means = []

        # Start from the most recent data
        for i in range(len(data_list) - 1, days - 2, -days):
            window = data_list[i - days + 1 : i + 1]
            mean_value = sum(window) / len(window)
            means.append(round(mean_value, 2))

        return means


if __name__ == "__main__":
    stock_data_fetcher = StockDataFetcher()
    stock_data_fetcher.fetch_today_listed_data()
    stock_data_fetcher.fetch_today_counter_data()
