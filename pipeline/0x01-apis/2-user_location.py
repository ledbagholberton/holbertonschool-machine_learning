#!/usr/bin/env python3
"""
The user is passed as first argument of the script with the full API URL,
example: ./2-user_location.py https://api.github.com/users/holbertonschool
If the user doesn’t exist, print Not found
If the status code is 403, print Reset in X min where X is the number of
minutes from now and the value of X-Ratelimit-Reset
Your code should not be executed when the file is imported
(you should use if __name__ == '__main__':)
"""
import requests
import sys
import time


if __name__ == '__main__':
    url = sys.argv[1]
    par = "Accept: application/vnd.github.v3+json"
    r = requests.get(url, par)

    if r.status_code == 200:
        print(r.json()["location"])
    elif r.status_code == 403:
        url = "https://api.github.com?callback=foo"
        r = requests.get(url)
        datos = r.json()["meta"]
        limit = datos["X-RateLimit-Reset"]
        x = limit - time.time()
        print("Reset in {} minuts").format(x)
    else:
        print("Not found")
