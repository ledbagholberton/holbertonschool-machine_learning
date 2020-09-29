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
    url_r = "https://api.spacexdata.com/v4/rockets"
    r = requests.get(url_r)
    all_rockets = r.json()
    for iter in all_rockets:
        rockets[iter["name"]] = 0
        list_r.append(iter["name"])
    url_l = "https://api.spacexdata.com/v4/launches"
    lx = requests.get(url_l)
    all_launches = lx.json()
    for iter in all_launches:
        id_rocket = iter["rocket"]
        rk = requests.get(url_r + '/' + id_rocket).json()["name"]
        if rk in list_r:
                rockets[rk] += 1
    sort_orders = sorted(rockets.items(), key=lambda x: (x[1], x[0]), reverse=True)
    for i in sort_orders:
        if i[1] > 0:
            print(i[0], ':', i[1])
