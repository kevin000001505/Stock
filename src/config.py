URL = "https://goodinfo.tw/tw2/StockList.asp"

LISTED_CSV = "data/StockList.csv"
COUNTER_CSV = "data/StockCounter.csv"

AUTH_FILE = "auth.json"

COOKIES = {
    "CLIENT%5FID": "20250806152624097%5F150%2E117%2E19%2E43",
    "CLIENT%5FID": "20250806152624097%5F150%2E117%2E19%2E43",
    "IS_TOUCH_DEVICE": "F",
    "SCREEN_SIZE": "WIDTH=1512&HEIGHT=982",
    "TW_STOCK_BROWSE_LIST": "0050",
}

COUNTER_PARAMS = {
    "SEARCH_WORD": "",
    "SHEET": "交易狀況",
    "MARKET_CAT": "上櫃",
    "INDUSTRY_CAT": "上櫃全部",
    "STOCK_CODE": "",
    "RANK": "0",
    "STEP": "DATA",
    "SHEET2": "日",
}

LISTED_PARAMS = {
    "SEARCH_WORD": "",
    "SHEET": "交易狀況",
    "MARKET_CAT": "上市",
    "INDUSTRY_CAT": "上市全部",
    "STOCK_CODE": "",
    "RANK": "0",
    "STEP": "DATA",
    "SHEET2": "日",
}


HEADERS = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    # 'content-length': '0',
    "content-type": "application/x-www-form-urlencoded;",
    "dnt": "1",
    "origin": "https://goodinfo.tw",
    "priority": "u=1, i",
    "referer": "https://goodinfo.tw/tw2/StockList.asp?MARKET_CAT=%E4%B8%8A%E6%AB%83&INDUSTRY_CAT=%E4%B8%8A%E6%AB%83%E5%85%A8%E9%83%A8&SHEET=%E4%BA%A4%E6%98%93%E7%8B%80%E6%B3%81&SHEET2=%E8%BF%9112%E6%97%A5%E6%94%B6%E7%9B%A4%E5%83%B9%E4%B8%80%E8%A6%BD&RPT_TIME=2025%2F08%2F06",
    "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
}
