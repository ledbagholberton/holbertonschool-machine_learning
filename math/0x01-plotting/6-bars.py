#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))
apples = fruit[3]
bananas = fruit[2]
oranges = fruit[1]
peaches = fruit[0]
names = np.array(["Farrah", "Fred", "Felicia"])
offset = np.array([0, 0, 0])
bar_width = 0.5

app = plt.bar(names, apples, bar_width, color="red", label="apples")
offset = offset + apples
ban = plt.bar(names, bananas, bar_width, bottom=offset,
              color="yellow", label="bananas")
offset = offset + bananas
ora = plt.bar(names, oranges, bar_width, bottom=offset,
              color="#ff8000", label="oranges")
offset = offset + oranges
pea = plt.bar(names, peaches, bar_width, bottom=offset,
              color="#ffe5b4", label="peaches")
plt.ylabel('Quantity of fruit')
plt.title('Number of Fruit per Person')
plt.yticks(np.arange(0, 81, 10))
plt.legend(handles=[app, ban, ora, pea])
plt.show()
