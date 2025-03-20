Yahoo Finance Historical Stock Data Scraper

Overview
This script scrapes historical stock data from Yahoo Finance and returns it as a structured pandas DataFrame. It automates data collection using Selenium WebDriver and processes the extracted data into a clean format.

Features
Scrapes historical stock prices for a given ticker symbol.
Automatically handles webpage loading and consent pop-ups.
Extracts relevant stock price data and structures it into a DataFrame.
Cleans data by filtering out unnecessary rows and converting data types.
Logs key events and errors for easy debugging.
Requirements
To run this script, install the required dependencies:

pip install pandas selenium

Setup
Ensure you have Google Chrome installed.
Download the ChromeDriver that matches your Chrome version from here.
Move the chromedriver executable to a directory in your system's PATH.

Usage
Run the script with a stock ticker symbol:

scraper = YahooFinanceHistoricalScraper("AAPL")  # Apple stock
df_history = scraper.scrape()
print(df_history.head())

Customisation
You can modify parameters such as:

headless: Run browser in headless mode (default True).
timeout: Page load timeout in seconds (default 10).

Example:

scraper = YahooFinanceHistoricalScraper("GOOGL", headless=False, timeout=15)


Troubleshooting
If the script fails to find a table, ensure Yahoo Finance hasn't changed its HTML structure.
Check if chromedriver is correctly installed and accessible.

