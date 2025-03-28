# YahooFinanceHistoricalScraper Documentation 🚀

This documentation provides an overview of the YahooFinanceHistoricalScraper module, installation instructions for the required packages, and examples on how to use it to scrape historical stock data from Yahoo Finance.

⸻

## Table of Contents
*	Overview
*	Requirements & Installation
*	Usage
*	Scraping Data Sequentially
*	Scraping Data in Parallel
*	File Structure
*	Additional Notes

⸻

## Overview

YahooFinanceHistoricalScraper is a Python module that uses Selenium to scrape historical stock data from Yahoo Finance. It supports both sequential and parallel scraping (using multiple threads) and offers an option to use proxies for distributing requests. The module also leverages pandas for data manipulation and tqdm for progress tracking.

⸻

## Requirements & Installation 📝

Before using the scraper, ensure you have the following installed:

### Python Version
*	Python 3.10 or higher

### Python Packages

```bash
# Create the virtual environment
python -m venv stockscraperenv

# Activate the virtual environment:
# On Windows:
stockscraperenv\Scripts\activate
# On macOS/Linux:
source stockscraperenv/bin/activate

# Install Jupyter and ipykernel inside the virtual environment
pip install -r requirements.txt

# Create a new Jupyter kernel for this environment
python -m ipykernel install --user --name stockscraperenv --display-name "stockscraperenv"
```

### WebDriver Setup
*	Google Chrome: Ensure that you have Google Chrome installed.
*	ChromeDriver: Download the ChromeDriver that matches your Chrome version from here and ensure it’s in your system’s PATH.
Note: If ChromeDriver is not in the PATH, you can specify its location when initializing the driver.

⸻

## Usage

After installing the requirements and setting up ChromeDriver, you can use the module in your project. Below are examples demonstrating how to instantiate and use the scraper.

### Importing the Module

If your project directory has the following file structure:

```
your_project/
├── YahooFinanceHistoricalScraper.py
└── __init__.py
```

You can import the scraper as:
```python
from YahooFinanceHistoricalScraper import YahooFinanceHistoricalScraper
```
### Scraping Data Sequentially

This example shows how to scrape data for multiple stock symbols using a single worker (sequential scraping):
```python
from YahooFinanceHistoricalScraper import YahooFinanceHistoricalScraper

# Optional: Provide a list of proxies (if needed).
proxy_list = [
    "http://your_username:your_password@35.239.40.187:3128",
    # Add more proxies if needed.
]

# List of stock symbols to scrape.
stock_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "FB", "NFLX", "NVDA"]

# Initialize the scraper.
scraper = YahooFinanceHistoricalScraper(headless=True, timeout=10, proxies=proxy_list)

# Sequential scraping (one worker).
data_dict_seq = scraper.scrape_multiple_stocks(stock_symbols, workers=1, show_progress=True)

# Display the scraped data.
for ticker, df in data_dict_seq.items():
    print(f"Data for {ticker}:")
    print(df.head())
```
### Scraping Data in Parallel

For faster processing, you can use multiple workers (threads) to scrape data concurrently:
```python
from YahooFinanceHistoricalScraper import YahooFinanceHistoricalScraper

# Optional: Provide proxies if you want to distribute requests.
proxy_list = [
    "http://your_username:your_password@35.239.40.187:3128",
    # Add more proxies as needed.
]

# List of stock symbols to scrape.
stock_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "FB", "NFLX", "NVDA"]

# Initialize the scraper.
scraper = YahooFinanceHistoricalScraper(headless=True, timeout=10, proxies=proxy_list)

# Parallel scraping using 3 workers.
data_dict_par = scraper.scrape_multiple_stocks(stock_symbols, workers=3, show_progress=True)

# Display the scraped data.
for ticker, df in data_dict_par.items():
    print(f"Data for {ticker}:")
    print(df.head())
```

⸻

Additional Notes
*   Dont use too many workers if you do not use proxies as you will get ratelimited, also more workers does not nessesarily mean it will be faster as the browsers will get slower if there is insufficient compute, memory or networking bandwidth.
*	Logging & Progress Bar:
The module uses Python’s built-in logging to provide information on the scraping process and tqdm to display a progress bar.
*	Proxies:
Proxies should be provided in the format "http://username:password@ip:port". If proxies are supplied, they will be evenly distributed among the worker threads.
*	Backoff Mechanism:
A global backoff mechanism is implemented to handle rate-limiting or connection issues gracefully by pausing the scraping process when needed.
