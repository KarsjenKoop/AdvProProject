import time
import logging
import threading
from io import StringIO
from typing import Optional, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    Optionally uses a provided list of proxies (in the format "http://username:password@ip:port")
    that are distributed evenly over the workers.
    """
    backoff_until: float = 0  # Epoch time until which all workers should back off.
    backoff_lock = threading.Lock()

    def __init__(self, headless: bool = True, timeout: int = 10, proxies: Optional[List[str]] = None) -> None:
        """
        Args:
            headless: Run Chrome in headless mode.
            timeout: Timeout in seconds for page loads.
            proxies: A list of proxy strings (e.g. "http://username:password@ip:port").
                     If provided, these will be assigned evenly across workers.
        """
        self.headless = headless
        self.timeout = timeout
        self.proxies = proxies or []
        # Flag to indicate if cookie consent has been accepted in a driver session.
        self.consent_accepted = False

    def _check_backoff(self) -> None:
        now = time.time()
        with self.__class__.backoff_lock:
            if now < self.__class__.backoff_until:
                wait_time = self.__class__.backoff_until - now
                logger.info("Global backoff active. Waiting for %.1f seconds.", wait_time)
        now = time.time()
        if now < self.__class__.backoff_until:
            time.sleep(self.__class__.backoff_until - now)

    def _init_driver(self, proxy: Optional[str] = None) -> webdriver.Chrome:
        """Initializes the Chrome WebDriver with desired options and, if provided, the given proxy."""
        options = Options()
        if self.headless:
            options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--incognito")
        options.add_argument("--disable-application-cache")
        # Disable image loading for faster performance.
        prefs = {"profile.managed_default_content_settings.images": 2}
        options.add_experimental_option("prefs", prefs)
        options.add_argument("--log-level=3")
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        if proxy:
            options.add_argument(f"--proxy-server={proxy}")
            logger.info("Using proxy: %s", proxy)
        try:
            driver = webdriver.Chrome(options=options)
            logger.info("Chrome WebDriver initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize Chrome WebDriver: %s", e)
            raise Exception("Chrome WebDriver initialization failed.") from e
        return driver

    def _fetch_page(self, driver: webdriver.Chrome, url: str) -> None:
        """Navigates to the URL, handles global backoff and cookie consent, and waits for the table element."""
        self._check_backoff()
        driver.get(url)
        logger.info("Navigated to URL: %s", url)
        if not self.consent_accepted:
            try:
                accept_button = WebDriverWait(driver, self.timeout).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "button.accept-all"))
                )
                logger.info("Consent button found; clicking it.")
                accept_button.click()
                self.consent_accepted = True
            except Exception as e:
                logger.info("No consent button found or unable to click it: %s", e)
        wait_time = self.timeout if not self.consent_accepted else max(2, self.timeout // 2)
        try:
            WebDriverWait(driver, wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
        except Exception as e:
            driver.quit()
            logger.error("No table found on the page: %s", e)
            raise Exception("Page did not load any table elements.") from e

    def _extract_target_table(self, driver: webdriver.Chrome) -> str:
        """Searches for the HTML table with expected headers and returns its outer HTML."""
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
            dfs = pd.read_html(StringIO(table_html))
            if not dfs:
                raise Exception("No tables parsed from HTML.")
            df = dfs[0]
        except Exception as e:
            logger.error("Error parsing table HTML: %s", e)
            raise Exception("Failed to parse table HTML into DataFrame.") from e
        return df

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renames columns using a simple mapping approach."""
        mapping = {}
        for col in df.columns:
            col = col.strip()
            if col.startswith("Adj Close"):
                mapping[col] = "Adj Close"
            elif col.startswith("Close"):
                mapping[col] = "Close"
        if mapping:
            df = df.rename(columns=mapping)
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the DataFrame by filtering out dividend rows,
        converting data types, and selecting the desired columns.
        """
        try:
            df = df[pd.to_numeric(df['Open'], errors='coerce').notnull()].copy()
        except Exception as e:
            logger.error("Error filtering dividend rows: %s", e)
            raise Exception("Failed to filter dividend rows.") from e

        df = self._rename_columns(df)

        try:
            df['Date'] = pd.to_datetime(df['Date'])
            for col in ["Open", "High", "Low", "Close"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['Volume'] = df['Volume'].astype(str).str.replace(',', '')
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        except Exception as e:
            logger.error("Error converting data types: %s", e)
            raise Exception("Data type conversion failed.") from e

        desired_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        df = df[[col for col in desired_cols if col in df.columns]]
        df.reset_index(drop=True, inplace=True)
        logger.info("Final DataFrame columns: %s", list(df.columns))
        return df

    def scrape_stock(self, ticker: str, proxy: Optional[str] = None) -> pd.DataFrame:
        """
        Scrapes historical data for a single ticker.
        Optionally uses the specified proxy for this driver session.
        """
        driver = self._init_driver(proxy=proxy)
        ticker = ticker.upper()
        url = f"https://finance.yahoo.com/quote/{ticker}/history?p={ticker}"
        try:
            self._fetch_page(driver, url)
            table_html = self._extract_target_table(driver)
            df = self._parse_table(table_html)
            df = self._clean_data(df)
            df['Ticker'] = ticker
        except Exception as e:
            logger.error("Error scraping ticker %s: %s", ticker, e)
            raise
        finally:
            driver.quit()
            logger.info("WebDriver closed for ticker %s.", ticker)
        return df

    def _scrape_ticker_chunk(self, tickers_chunk: List[str], proxy: Optional[str]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """
        Scrapes a chunk of tickers sequentially using one driver instance assigned the given proxy.
        Returns a tuple of:
          - a dictionary with successful ticker data,
          - a list of tickers that failed.
        """
        results: Dict[str, pd.DataFrame] = {}
        failed_tickers: List[str] = []
        driver = self._init_driver(proxy=proxy)
        self.consent_accepted = False  # Reset consent for this driver session.
        try:
            for ticker in tickers_chunk:
                ticker = ticker.upper()
                url = f"https://finance.yahoo.com/quote/{ticker}/history?p={ticker}"
                logger.info("Scraping ticker: %s", ticker)
                try:
                    self._check_backoff()
                    self._fetch_page(driver, url)
                    table_html = self._extract_target_table(driver)
                    df = self._parse_table(table_html)
                    df = self._clean_data(df)
                    df['Ticker'] = ticker
                    results[ticker] = df
                except Exception as e:
                    if "Connection refused" in str(e):
                        backoff_time = 10  # seconds to back off (adjust as needed)
                        with self.__class__.backoff_lock:
                            self.__class__.backoff_until = time.time() + backoff_time
                        logger.info("Rate limit detected while scraping %s. Backing off all workers for %s seconds.", ticker, backoff_time)
                    else:
                        logger.error("Error scraping ticker %s: %s", ticker, e)
                    failed_tickers.append(ticker)
                    time.sleep(2)
                    continue
        finally:
            driver.quit()
            logger.info("WebDriver closed for ticker chunk: %s", tickers_chunk)
        return results, failed_tickers

    def scrape_multiple_stocks(self, tickers: List[str], workers: int = 1) -> Dict[str, pd.DataFrame]:
        """
        Scrapes historical data for multiple tickers.
          - If workers == 1, it reuses a single driver instance for all tickers sequentially.
          - If workers > 1, it splits the tickers into chunks and uses a ThreadPoolExecutor so each worker processes a chunk,
            with each worker assigned a proxy (if provided) in an even, round-robin manner.
        """
        results: Dict[str, pd.DataFrame] = {}
        global_failed_tickers: List[str] = []
        if workers == 1:
            # If proxies are provided, use the first proxy.
            proxy = self.proxies[0] if self.proxies else None
            failed_tickers: List[str] = []
            self.consent_accepted = False
            driver = self._init_driver(proxy=proxy)
            try:
                for ticker in tickers:
                    ticker = ticker.upper()
                    url = f"https://finance.yahoo.com/quote/{ticker}/history?p={ticker}"
                    logger.info("Scraping ticker: %s", ticker)
                    try:
                        self._check_backoff()
                        self._fetch_page(driver, url)
                        table_html = self._extract_target_table(driver)
                        df = self._parse_table(table_html)
                        df = self._clean_data(df)
                        df['Ticker'] = ticker
                        results[ticker] = df
                    except Exception as e:
                        if "Connection refused" in str(e):
                            backoff_time = 10
                            with self.__class__.backoff_lock:
                                self.__class__.backoff_until = time.time() + backoff_time
                            logger.info("Rate limit detected while scraping %s. Backing off all workers for %s seconds.", ticker, backoff_time)
                        else:
                            logger.error("Error scraping ticker %s: %s", ticker, e)
                        failed_tickers.append(ticker)
                        time.sleep(2)
                        continue
            finally:
                driver.quit()
                logger.info("WebDriver closed after sequential scraping.")
            if failed_tickers:
                logger.info("Failed tickers (sequential): %s", failed_tickers)
            global_failed_tickers.extend(failed_tickers)
        else:
            # Split tickers into chunks for each worker.
            def chunk_list(lst: List[str], n: int) -> List[List[str]]:
                k, m = divmod(len(lst), n)
                return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
            ticker_chunks = chunk_list(tickers, workers)
            logger.info("Splitting %s tickers into %s chunks: %s", len(tickers), workers, ticker_chunks)
            
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = []
                # Assign proxies evenly: if there are proxies provided, assign worker i the proxy proxies[i % len(proxies)]
                for i, chunk in enumerate(ticker_chunks):
                    proxy = self.proxies[i % len(self.proxies)] if self.proxies else None
                    futures.append(executor.submit(self._scrape_ticker_chunk, chunk, proxy))
                for future in as_completed(futures):
                    try:
                        chunk_results, chunk_failed = future.result()
                        results.update(chunk_results)
                        global_failed_tickers.extend(chunk_failed)
                    except Exception as e:
                        logger.error("Error scraping a chunk: %s", e)
            if global_failed_tickers:
                logger.info("Failed tickers (parallel): %s", global_failed_tickers)
        return results

# Example usage:
if __name__ == "__main__":
    # Provide a list of proxies in the form "http://username:password@ip:port".
    proxy_list = [
        "http://your_username:your_password@35.239.40.187:3128",
        # Add more proxies if needed.
    ]
    # List of stock symbols to scrape.
    stock_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "FB", "NFLX", "NVDA"]

    scraper = YahooFinanceHistoricalScraper(headless=True, timeout=10, proxies=proxy_list)
    
    # Sequential scraping (one worker):
    data_dict_seq = scraper.scrape_multiple_stocks(stock_symbols, workers=1)
    for ticker, df in data_dict_seq.items():
        logger.info("Sequential - Data for %s:\n%s", ticker, df.head())
    
    # Parallel scraping (e.g., using 3 workers):
    data_dict_par = scraper.scrape_multiple_stocks(stock_symbols, workers=3)
    for ticker, df in data_dict_par.items():
        logger.info("Parallel - Data for %s:\n%s", ticker, df.head())