import time
import csv
import json
import argparse
from selenium.webdriver.common.by import By
from utils import create_driver, HASHES_FILE, CSV_FILE
from search import search_tweets
from login import login

def load_hashes():
    try:
        with open(HASHES_FILE, "r", encoding="utf-8") as file:
            return set(json.load(file))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()

def save_hashes(hashes):
    with open(HASHES_FILE, "w", encoding="utf-8") as file:
        json.dump(list(hashes), file, indent=4)

def append_to_csv(data, filename=CSV_FILE):
    with open(filename, "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["text", "replies", "retweets", "likes", "views", "author_name", "author_handle", "publication_time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerows(data)

    print(f"Appended {len(data)} new tweets to {filename}")

def main(keyword = 'coronavirus'):
    print(keyword)
    parser = argparse.ArgumentParser(description="Twitter Scraper")
    parser.add_argument("--skip", action="store_true", help="Skip login check")
    args = parser.parse_args()

    loaded_tweet_hashes = load_hashes()

    print(f"Load {len(loaded_tweet_hashes)} tweets")
    if not args.skip:
        if login() == False:
            print('Not logged in')
            return

    driver = create_driver(headless=True)    
    print("Start crawling")

    try:
        new_tweets = search_tweets(driver, keyword, loaded_tweet_hashes)
        append_to_csv(new_tweets)
        save_hashes(loaded_tweet_hashes)

    finally:
        driver.quit()

if __name__ == "__main__":
    
    '''
    main(keyword = '(covid OR coronavirus OR COVID OR vaccine OR vaccination OR remote working OR work from home OR WFH) since:2023-01-01 until:2023-06-01 min_faves:1000')
    time.sleep(40)
    main(keyword = '(covid OR coronavirus OR COVID OR vaccine OR vaccination OR remote working OR work from home OR WFH) since:2023-02-01 until:2023-03-01')
    time.sleep(40)
    main(keyword = '(covid OR coronavirus OR COVID OR vaccine OR vaccination OR remote working OR work from home OR WFH) since:2023-03-01 until:2023-04-01')
    time.sleep(40)
    main(keyword = '(covid OR coronavirus OR COVID OR vaccine OR vaccination OR remote working OR work from home OR WFH) since:2023-04-01 until:2023-05-01')
    # time.sleep(40)
    '''

    '''
    main(keyword = '(covid OR coronavirus OR COVID OR vaccine OR vaccination OR remote working OR work from home OR WFH) since:2023-05-01 until:2023-06-01')
    time.sleep(40)
    main(keyword = '(covid OR coronavirus OR COVID OR vaccine OR vaccination OR remote working OR work from home OR WFH) since:2023-06-01 until:2023-07-01')
    time.sleep(40)
    main(keyword = '(covid OR coronavirus OR COVID OR vaccine OR vaccination OR remote working OR work from home OR WFH) since:2023-07-01 until:2023-08-01')
    time.sleep(40)
    main(keyword = '(covid OR coronavirus OR COVID OR vaccine OR vaccination OR remote working OR work from home OR WFH) since:2023-08-01 until:2023-09-01')
    # time.sleep(40)
    '''

    '''
    main(keyword = '(covid OR coronavirus OR COVID OR vaccine OR vaccination OR remote working OR work from home OR WFH) since:2023-09-01 until:2023-10-01')
    time.sleep(40)
    main(keyword = '(covid OR coronavirus OR COVID OR vaccine OR vaccination OR remote working OR work from home OR WFH) since:2023-10-01 until:2023-11-01')
    time.sleep(40)
    main(keyword = '(covid OR coronavirus OR COVID OR vaccine OR vaccination OR remote working OR work from home OR WFH) since:2023-11-01 until:2023-12-01')
    time.sleep(40)
    main(keyword = '(covid OR coronavirus OR COVID OR vaccine OR vaccination OR remote working OR work from home OR WFH) since:2023-12-01 until:2024-01-01')
    '''

    main(keyword = '(covid OR coronavirus OR COVID OR vaccine OR vaccination OR remote working OR work from home OR WFH) since:2020-01-01 until:2020-03-01 min_faves:1000')
    # time.sleep(40)

    '''
    main(keyword = '(covid OR coronavirus OR COVID OR vaccine OR vaccination OR remote working OR work from home OR WFH) since:2020-03-01 until:2020-05-01 min_faves:1000')
    time.sleep(40)
    main(keyword = '(covid OR coronavirus OR COVID OR vaccine OR vaccination OR remote working OR work from home OR WFH) since:2020-05-01 until:2020-07-01 min_faves:1000')
    time.sleep(40)
    main(keyword = '(covid OR coronavirus OR COVID OR vaccine OR vaccination OR remote working OR work from home OR WFH) since:2020-07-01 until:2020-09-01 min_faves:1000')
    time.sleep(40)
    main(keyword = '(covid OR coronavirus OR COVID OR vaccine OR vaccination OR remote working OR work from home OR WFH) since:2020-09-01 until:2020-11-01 min_faves:1000')
    time.sleep(40)
    main(keyword = '(covid OR coronavirus OR COVID OR vaccine OR vaccination OR remote working OR work from home OR WFH) since:2020-11-01 until:2021-01-01 min_faves:1000')
    time.sleep(40)
    '''