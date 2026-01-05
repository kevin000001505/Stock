URL = "https://goodinfo.tw/tw2/StockList.asp"

LISTED_CSV = "data/StockList.csv"
COUNTER_CSV = "data/StockCounter.csv"

AUTH_FILE = "auth.json"

COOKIES = {
    "CLIENT_KEY": "2.6%7C39966.8613468014%7C46633.528013468%7C300%7C20457.925772766204%7C20457.925864270834",
    "CLIENT%5FID": "20260105110844847%5F108%2E31%2E83%2E77",
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
    "IS_RELOAD_REPORT": "T",
}

LISTED_PARAMS = {
    "STEP": "DATA",
    "SEARCH_WORD": "",
    "SHEET": "交易狀況",
    "MARKET_CAT": "上市",
    "INDUSTRY_CAT": "上市全部",
    "STOCK_CODE": "",
    "RANK": "0",
    "SHEET2": "日",
    "IS_RELOAD_REPORT": "T",
}


HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded;",
    "Accept": "*/*",
    "Referer": "https://goodinfo.tw/tw/StockList.asp?MARKET_CAT=%E4%B8%8A%E5%B8%82&INDUSTRY_CAT=%E4%B8%8A%E5%B8%82%E5%85%A8%E9%83%A8&SHEET=%E4%BA%A4%E6%98%93%E7%8B%80%E6%B3%81&SHEET2=%E6%97%A5&RPT_TIME=%E6%9C%80%E6%96%B0%E8%B3%87%E6%96%99",
    "Origin": "https://goodinfo.tw",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0.1 Safari/605.1.15",
}
