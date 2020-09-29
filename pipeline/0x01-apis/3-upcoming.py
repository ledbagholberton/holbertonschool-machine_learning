#!/usr/bin/env python3
"""
By using the (unofficial) SpaceX API, write a script that displays the
upcoming launch with these information:
Name of the launch
The date (in local time)
The rocket name
The name (with the locality) of the launchpad
Format:
<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)
"""
import requests
import sys
import time


if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches/next"
    r = requests.get(url)
    launch_name = r.json()["name"]
    date = r.json()["static_fire_date_utc"]
    rocket = r.json()["rocket"]
    url_rocket = "https://api.spacexdata.com/v4/rockets/" + rocket
    rocket_name = requests.get(url_rocket).json()["name"]
    launchpad = r.json()["launchpad"]
    url_launchpad = "https://api.spacexdata.com/v4/launchpads/" + launchpad
    launchpad_name = requests.get(url_launchpad).json()["name"]
    locality = requests.get(url_launchpad).json()["locality"]
    print("{} ({}) {} - {} ({})".format(launch_name, date, rocket_name,
          launchpad_name, locality))
