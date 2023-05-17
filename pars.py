import requests
import csv
from bs4 import BeautifulSoup

colors = ["красный", "зелёный", "синий", "жёлтый", "чёрный", "белый"]
vectors = [
    '1,0,0,0,0,0',
    '0,1,0,0,0,0',
    '0,0,1,0,0,0',
    '0,0,0,1,0,0',
    '0,0,0,0,1,0',
    '0,0,0,0,0,1'
]

data = []

for color, vector in zip(colors, vectors):
    url = f"https://wordassociation.ru/{color}"
    data.append((url, vector))

with open('color_data.csv', 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)

    for url, vector in data:
        response = requests.get(url)
        html_code = response.text
        soup = BeautifulSoup(html_code, "html.parser")
        word_lists = soup.find_all('ol', class_='list-associations col-md-4')

        for word_list in word_lists:
            items = word_list.find_all('li')

            for item in items:
                word = item.find('span').text

                row = [word] + vector.split(",")
                writer.writerow(row)

print("Данные успешно добавлены в файл.")
