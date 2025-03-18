import time
import logging
from io import StringIO
from typing import Optional

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Set up basic logging.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class YahooFinanceHistoricalScraper:
    """
    Scrapes historical stock data from Yahoo Finance.

    Attributes:
        ticker (str): Stock ticker symbol (e.g., "AAPL").
        headless (bool): Whether to run the browser in headless mode.
        timeout (int): Seconds to wait for page elements.
    """

    def __init__(self, ticker: str, headless: bool = True, timeout: int = 10) -> None:
        self.ticker = ticker.upper()
        self.headless = headless
        self.timeout = timeout
        self.url = f"https://finance.yahoo.com/quote/{self.ticker}/history?p={self.ticker}"

    def _init_driver(self) -> webdriver.Chrome:
        """Initializes the Chrome WebDriver with desired options."""
        options = Options()
        if self.headless:
            options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--log-level=3")
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        try:
            driver = webdriver.Chrome(options=options)
            logger.info("Chrome WebDriver initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize Chrome WebDriver: %s", e)
            raise Exception("Chrome WebDriver initialization failed.") from e
        return driver

    def _fetch_page(self, driver: webdriver.Chrome) -> webdriver.Chrome:
        """Navigates to the target URL, handles consent, and waits for the page to load."""
        driver.get(self.url)
        logger.info("Navigated to URL: %s", self.url)
        try:
            accept_button = WebDriverWait(driver, self.timeout).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button.accept-all"))
            )
            logger.info("Consent button found; clicking it.")
            accept_button.click()
            time.sleep(2)  # Allow time for overlay to vanish.
        except Exception as e:
            logger.info("No consent button found or unable to click it: %s", e)
        try:
            WebDriverWait(driver, self.timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
        except Exception as e:
            driver.quit()
            logger.error("No table found on the page: %s", e)
            raise Exception("Page did not load any table elements.") from e
        return driver

    def _extract_target_table(self, driver: webdriver.Chrome) -> str:
        """
        Searches all table elements on the page and returns the HTML of the table
        that contains the expected headers.
        """
        expected_headers = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        tables = driver.find_elements(By.TAG_NAME, "table")
        target_table_html: Optional[str] = None
        for table in tables:
            try:
                thead = table.find_element(By.TAG_NAME, "thead")
                header_text = thead.text
                if all(header in header_text for header in expected_headers):
                    target_table_html = table.get_attribute("outerHTML")
                    logger.info("Found target table with expected headers.")
                    break
            except Exception:
                continue
        if target_table_html is None:
            logger.error("Could not find target table with headers: %s", expected_headers)
            raise Exception("Target table not found based on header text.")
        return target_table_html

    def _parse_table(self, table_html: str) -> pd.DataFrame:
        """Parses the HTML table into a pandas DataFrame using StringIO."""
        try:
            # Wrap the HTML string in a StringIO object.
            dfs = pd.read_html(StringIO(table_html))
            if not dfs:
                raise Exception("No tables parsed from HTML.")
            df = dfs[0]
            logger.info("Table parsed into DataFrame successfully.")
        except Exception as e:
            logger.error("Error parsing table HTML: %s", e)
            raise Exception("Failed to parse table HTML into DataFrame.") from e
        return df

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Renames columns using a simple mapping approach.
        For example:
            - 'Close Close price adjusted for splits.' -> 'Close'
            - 'Adj Close Adjusted close price adjusted for splits and dividend and/or capital gain distributions.' -> 'Adj Close'
        """
        mapping = {}
        for col in df.columns:
            col = col.strip()
            if col.startswith("Adj Close"):
                mapping[col] = "Adj Close"
            elif col.startswith("Close"):
                mapping[col] = "Close"
        if mapping:
            logger.info("Renaming columns using mapping: %s", mapping)
            df = df.rename(columns=mapping)  # Return new DataFrame to avoid SettingWithCopyWarning.
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the DataFrame by:
          - Removing dividend rows.
          - Renaming columns via a simple mapping.
          - Converting column types.
          - Selecting desired columns (including 'Adj Close' if available).
        """
        try:
            # Filter out dividend rows by creating an explicit copy.
            df = df[pd.to_numeric(df['Open'], errors='coerce').notnull()].copy()
            logger.info("Filtered out dividend rows.")
        except Exception as e:
            logger.error("Error filtering dividend rows: %s", e)
            raise Exception("Failed to filter dividend rows.") from e

        df = self._rename_columns(df)
        logger.info("Columns after renaming: %s", list(df.columns))

        # Convert data types.
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            for col in ["Open", "High", "Low", "Close"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # Remove commas from the Volume column before conversion.
            df['Volume'] = df['Volume'].astype(str).str.replace(',', '')
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        except Exception as e:
            logger.error("Error converting data types: %s", e)
            raise Exception("Data type conversion failed.") from e

        # Select desired columns; include 'Adj Close' if it exists.
        desired_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        df = df[[col for col in desired_cols if col in df.columns]]
        df.reset_index(drop=True, inplace=True)
        logger.info("Final DataFrame columns: %s", list(df.columns))
        return df

    def scrape(self) -> pd.DataFrame:
        """
        Scrapes Yahoo Finance for historical stock data and returns a cleaned pandas DataFrame.
        """
        driver = self._init_driver()
        try:
            driver = self._fetch_page(driver)
            table_html = self._extract_target_table(driver)
        except Exception as e:
            logger.error("Error during page fetch/extraction: %s", e)
            raise
        finally:
            driver.quit()
            logger.info("WebDriver closed.")
        df = self._parse_table(table_html)
        df = self._clean_data(df)
        return df


# Example usage:
if __name__ == "__main__":
    try:
        scraper = YahooFinanceHistoricalScraper("AAPL")
        df_history = scraper.scrape()
        logger.info("Historical Data (first 5 rows):\n%s", df_history.head())
    except Exception as ex:
        logger.error("An error occurred during scraping: %s", ex)