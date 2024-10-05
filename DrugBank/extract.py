import requests
from bs4 import BeautifulSoup as bs 
import pandas as pd

def get_drugbank_info(uniprot_id):
    url = f'https://go.drugbank.com/unearth/q?searcher=bio_entities&query={uniprot_id}'
    page = requests.get(url)
    soup = bs(page.text, 'html.parser')

    link = soup.find('h2', class_='hit-link')
    link = link.find('a')
    link = link.get('href')

    new_url = 'https://go.drugbank.com' + link
    print(new_url)

    page2 = requests.get(new_url)
    soup2 = bs(page2.text, 'html.parser')
    table = soup2.find('table', id='target-relations')
    table = table.find('tbody')
    rows = table.find_all('tr')
    drugs = {}
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols[:-1]]
        drugs[cols[0]] = [ele for ele in cols[1:] if ele]
    return drugs

if __name__ == '__main__':
    uniprot_id = 'P05067'
    drugs = get_drugbank_info(uniprot_id)
    print(drugs)
    