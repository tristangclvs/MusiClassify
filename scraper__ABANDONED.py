import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup

#                                                                               #
# #                                                                           # #
# ============================================================================= #
# Aborted code : I have not the rights of the scraped website to use it         #
# ============================================================================= #
# #                                                                           # #
#                                                                               #

# # Ask for Genre and creates directory if not already existing # #
# Genre of the musics to scrape
genre: str = input("Enter the genre of the musics you want to scrape: ")

download_directory = f"downloads/{genre.capitalize()}"

if not os.path.exists(download_directory):
    os.makedirs(download_directory)
    print(f"Directory '{download_directory}' created successfully.")
else:
    print(f"Directory '{download_directory}' already exists.")

# =============================================================================

# the URL of the website you want to scrape
url: str = f"https://freemusicarchive.org/genre/{genre.capitalize()}/"

# create a new Chrome browser instance
driver = webdriver.Chrome()

# implicit wait for 5 seconds
driver.implicitly_wait(5)

# navigate to the URL of the website you want to scrape
driver.get(url)

# click on the Cookies button
cookies_button = driver.find_element(By.ID, 'CybotCookiebotDialogBodyLevelButtonLevelOptinAllowallSelection')
driver.implicitly_wait(5)
ActionChains(driver).move_to_element(cookies_button).click(cookies_button).perform()

# Get all links to the songs, song titles and artists
links = driver.find_elements(By.CSS_SELECTOR, 'a.js-download')
track_titles = driver.find_elements(By.CSS_SELECTOR, 'span.ptxt-track')
track_artists = driver.find_elements(By.CSS_SELECTOR, 'span.ptxt-artist')

# index for easier access to the song title and artist name
index = 0

for link in links:
    try:
        # get the song title and artist name
        song_title: str = "".join(ch for ch in track_titles[index].text if ch.isalnum())
        artist_name: str = "".join(ch for ch in track_artists[index].text.split(' ')[-1] if ch.isalnum())
        print(f"Processing {song_title} by {artist_name}")

        # click on the link
        ActionChains(driver).move_to_element(link).click(link).perform()
    except:
        print("Error")
        driver.execute_script(
            "window.scrollTo(0, window.scrollY + 1000)")  # need to restart the scraping with new links after scrolling
        # close the browser
        # driver.quit()
    finally:
        # find the link to the file you want to download
        file_link: str = driver.find_element(By.CSS_SELECTOR, 'a.download').get_attribute('href')
        response = requests.get(file_link)

        # save the file
        with open(f"{download_directory}/{song_title}-{artist_name}.mp3", "wb") as file:
            file.write(response.content)
        index += 1

        # close the download overlay
        close_button = driver.find_element(By.ID, 'downloadOverlayClose')
        ActionChains(driver).move_to_element(close_button).click(close_button).perform()

        # print(file_link)

# close the browser
driver.quit()
# response = requests.get(file_link)
print("Process finished")

# download the file
# response = requests.get(file_link)

# save the file to your computer
# with open("SCRAPED_FILE.mp3", "wb") as file:
#     file.write(response.content)
