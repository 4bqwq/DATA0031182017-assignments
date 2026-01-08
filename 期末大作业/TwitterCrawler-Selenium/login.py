from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from utils import TWITTER_URL, TWEET_XPATH, DEFAULT_TIMEOUT, TWITTER_URL, create_driver

def wait_for_login(driver, timeout = DEFAULT_TIMEOUT):
    driver.get(f"{TWITTER_URL}i/flow/login")
    try:
        print("Waiting for login")
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.XPATH, TWEET_XPATH))
        )
        print("Login")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def is_logged_in(driver):
    try:
        driver.find_element(By.XPATH, TWEET_XPATH)
        return True
    except NoSuchElementException:
        return False
    
def login():
    print("Start")
    driver = create_driver(headless=False)
    print('Loading')
    driver.get(f"{TWITTER_URL}home")
    if not is_logged_in(driver):
        print("Not logged in.")
        logged_in = wait_for_login(driver)
    else:
        logged_in = True
        print("Login")
    driver.quit()
    return logged_in

if __name__ == "__main__":
    login()