# 📈 Yahoo Finance Historical Scraper

This Python tool allows you to **scrape historical stock price data** from [Yahoo Finance](https://finance.yahoo.com) using **Selenium WebDriver**. It supports both **sequential** and **parallel scraping**, and can optionally be configured with a list of **proxy servers** to bypass rate-limiting or geo-restrictions.

---

## 🚀 Features

- ✅ Scrapes historical stock price tables for multiple tickers
- ✅ Headless Chrome WebDriver support
- ✅ Intelligent handling of Yahoo cookie consent banners
- ✅ Global backoff strategy for rate-limit handling
- ✅ Proxy support (rotated across threads)
- ✅ Parses data into clean, structured **Pandas DataFrames**
- ✅ Supports **multi-threaded parallel scraping** using `ThreadPoolExecutor`

---

## 🧰 Requirements

- Python 3.7+
- Google Chrome installed
- ChromeDriver installed and available in PATH

### Python Packages

Install dependencies with:

```bash
pip install pandas selenium
```

---

## 🛠️ Usage

### 1. Configure Proxies (Optional)

```python
proxy_list = [
    "http://username:password@proxy_ip:port",
    "http://username:password@proxy_ip_2:port",
]
```

### 2. List of Stock Tickers

```python
stock_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "FB", "NFLX", "NVDA"]
```

### 3. Initialize the Scraper

```python
scraper = YahooFinanceHistoricalScraper(headless=True, timeout=10, proxies=proxy_list)
```

### 4. Scrape Sequentially

```python
data_dict_seq = scraper.scrape_multiple_stocks(stock_symbols, workers=1)
for ticker, df in data_dict_seq.items():
    print(f"{ticker}:\n", df.head())
```

### 5. Scrape in Parallel

```python
data_dict_par = scraper.scrape_multiple_stocks(stock_symbols, workers=3)
for ticker, df in data_dict_par.items():
    print(f"{ticker}:\n", df.head())
```

---

## 📦 Output

Each ticker will return a cleaned `pandas.DataFrame` containing:

| Date       | Open   | High   | Low    | Close  | Adj Close | Volume     | Ticker |
|------------|--------|--------|--------|--------|-----------|------------|--------|
| 2024-03-25 | 150.32 | 152.88 | 149.67 | 151.85 | 151.85    | 50,000,000 | AAPL   |

---

## 🧠 Notes

- The scraper uses `pandas.read_html()` to convert the scraped HTML table to a `DataFrame`.
- It intelligently filters out **non-price rows** (like dividends).
- When rate-limited or blocked, a **global backoff** is applied to let all threads cool down before retrying.
- You can assign **proxy rotation** when using multiple threads.

---

## 🧪 Example

To run the example in the script:

```bash
python scraper.py
```

---

## ❗ Disclaimer

This tool is for **educational and research purposes** only. Scraping Yahoo Finance may violate their **terms of service**, so please use responsibly and at your own risk.

---

