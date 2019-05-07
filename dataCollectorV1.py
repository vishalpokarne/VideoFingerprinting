import time
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
import subprocess, sys

video_driver = webdriver.Chrome("C:/Users/VISHAL/Downloads/chromedriver_win32/chromedriver.exe")

for video_number in range(1,100):
    with open("D:\\VideoFingerprint\\Dataset\\NewVideoSet.txt ") as file:
        counter = 15

        for line in file:
            file_name = str(counter) + '_' + str(video_number)
            counter += 1
            video_driver.get(''+line)
            print("URL no: ", counter, "\nURL: ", line)

            try:
                end_time = video_driver.find_element_by_class_name("ytp-time-duration")
            except NoSuchElementException:
                print("Video shifted to another link")
                continue

            try:
                run_time = 60 * int(str(end_time.text).split(":")[0]) + int(str(end_time.text).split(":")[1])
                if run_time <= 30:
                    print(run_time)
                    time.sleep(run_time+1)

                    p = subprocess.Popen(["powershell.exe","D:\\VideoFingerprint\\run_tshark.ps1 " + file_name + ".csv"],
                                         stdout=sys.stdout)
                    youtube_player = video_driver.execute_script(
                        "return document.getElementById('movie_player').getPlayerState();")
                    print("Initial Status: ", youtube_player)

                    while youtube_player != 0:
                        youtube_player = video_driver.execute_script(
                            "return document.getElementById('movie_player').getPlayerState();")
                    print("Final Status: ", youtube_player)
                    print("Video ended")
                    p.terminate()
                else:
                    print(run_time)

                    p = subprocess.Popen(["powershell.exe", "D:\\VideoFingerprint\\run_tshark.ps1 " + file_name + ".csv"],
                                         stdout=sys.stdout)
                    youtube_player = video_driver.execute_script(
                        "return document.getElementById('movie_player').getPlayerState();")
                    print("Initial Status: ", youtube_player)

                    while youtube_player != 0:
                        youtube_player = video_driver.execute_script(
                            "return document.getElementById('movie_player').getPlayerState();")
                    print("Final Status: ", youtube_player)
                    print("Video ended")
                    p.terminate()
            except:
                pass