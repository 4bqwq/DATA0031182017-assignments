import time
import hashlib
from selenium.webdriver.common.by import By
from utils import TWITTER_URL, TWEET_XPATH, PAGE_LOAD_WAIT_TIME, TWITTER_SCROLL_COUNT

def generate_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def extract_tweet_data(tweet_element):
    tweet_data = {
        "text": "",
        "replies": "0",
        "retweets": "0",
        "likes": "0",
        "views": "0",
        "author_name": "",
        "author_handle": "",
        "publication_time": "",
    }

    is_data_missing = False

    try:
        tweet_data["text"] = tweet_element.find_element(By.XPATH, './/div[@data-testid="tweetText"]').text
    except Exception:
        print("Error extracting text")
        return None

    try:
        tweet_data["replies"] = tweet_element.find_element(By.XPATH, './/button[@data-testid="reply"]').text
    except Exception:
        print("Error extracting replies")
        is_data_missing = True

    try:
        tweet_data["retweets"] = tweet_element.find_element(By.XPATH, './/button[@data-testid="retweet"]').text
    except Exception:
        print("Error extracting retweets")
        is_data_missing = True

    try:
        tweet_data["likes"] = tweet_element.find_element(By.XPATH, './/button[@data-testid="like"]').text
    except Exception:
        print("Error extracting likes")
        is_data_missing = True

    try:
        tweet_data["views"] = tweet_element.find_element(By.XPATH, './/a[contains(@aria-label, "views")]').text
    except Exception:
        print("Error extracting views")
        is_data_missing = True

    try:
        tweet_data["author_name"] = tweet_element.find_elements(
            By.XPATH, './/div[contains(@dir, "ltr")]//span'
        )[0].text
    except Exception:
        print("Error extracting author name")
        is_data_missing = True

    try:
        tweet_data["author_handle"] = tweet_element.find_elements(
            By.XPATH, './/div[contains(@dir, "ltr")]//span'
        )[3].text
    except Exception:
        print("Error extracting author handle")
        is_data_missing = True

    try:
        tweet_data["publication_time"] = tweet_element.find_element(
            By.XPATH, '//time[@datetime]'
        ).get_attribute("datetime")
    except Exception:
        print("Error extracting publication time")
        is_data_missing = True

    if is_data_missing == False:
        print("Successfully extract data")
        
    return tweet_data


def search_tweets(driver, keyword, loaded_tweet_hashes):
    search_url = f"{TWITTER_URL}search?q={keyword}&src=typed_query"
    driver.get(search_url)
    time.sleep(PAGE_LOAD_WAIT_TIME)

    all_tweets = []

    for _ in range(TWITTER_SCROLL_COUNT):
        tweet_elements = driver.find_elements(By.XPATH, TWEET_XPATH)

        tweet_count = 0
        for tweet_element in tweet_elements:
            tweet_data = extract_tweet_data(tweet_element)
            if not tweet_data:
                continue

            tweet_text = tweet_data.get("text", "")
            tweet_hash = generate_hash(tweet_text)

            if tweet_hash and tweet_hash not in loaded_tweet_hashes:
                tweet_count = tweet_count + 1
                loaded_tweet_hashes.add(tweet_hash)
                all_tweets.append(tweet_data)

        print(f"Scraped {len(loaded_tweet_hashes)} tweets")
        print(f"Page Crawling Progress: {_ + 1} / {TWITTER_SCROLL_COUNT}")

        if (tweet_count == 0):
            break

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(PAGE_LOAD_WAIT_TIME)

    return all_tweets