import os
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service

TWITTER_URL = "https://x.com/"
DRIVER_PATH = "./msedgedriver.exe"
SESSION_FOLDER = os.path.join(os.getcwd(), 'selenium_session')
DEFAULT_TIMEOUT = 600
PAGE_LOAD_WAIT_TIME = 7
TWITTER_SCROLL_COUNT = 15
HASHES_FILE = "loaded_tweet_hashes.json"
CSV_FILE = "tweets.csv"
TWEET_XPATH = '//article[@role="article" and @data-testid="tweet"]'


def create_driver(headless = False):
    options = Options()
    options.add_argument(f"--user-data-dir={SESSION_FOLDER}")
    service = Service(DRIVER_PATH)
    if headless:
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-extensions")

    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--disable-web-security')
    options.add_argument('--allow-insecure-localhost')
    options.add_argument('--ignore-ssl-errors=yes')
    options.add_argument("--log-level=3")

    return webdriver.Edge(service=service, options=options)