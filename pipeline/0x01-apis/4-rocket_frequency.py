#!/usr/bin/env python3
"""
Using SpaceX API, write script that displays number of launches per rocket.
All launches should be taking in consideration
Each line should contain the rocket name and the number of launches
separated by : (format below in the example)
Order the result by the number launches (descending)
If multiple rockets have the same amount of launches, order them by
alphabetic order (A to Z)
"""
import requests


if __name__ == '__main__':
    rockets = {}
    list_r = []
    url = "https://api.spacexdata.com/v4/rockets"
    r = requests.get(url)
    all_rockets = r.json()
    for iter in all_rockets:
        rockets[iter["name"]] = 0
        list_r.append(iter["name"])
    url = "https://api.spacexdata.com/v4/launches"
    r = requests.get(url)
    all_launches = r.json()
    for iter in all_launches:
        if iter["success"] is "true":
            if iter["name"] in list_r:
                rockets[iter["name"]] += 1
        else:
            pass
    print(rockets)
