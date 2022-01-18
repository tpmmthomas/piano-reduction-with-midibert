import requests
from bs4 import BeautifulSoup

base_url = "http://www.piano-midi.de/"
mainpage = "midicoll.htm"

page = requests.get(base_url+mainpage)
soup = BeautifulSoup(page.content,"html.parser")

links = []
for link in soup.findAll('a'):
    links.append(link.get('href'))

for link in links:
    if link == "midi_files.htm":
        continue
    if link == "other.htm":
        break
    print(link)
    page = requests.get(base_url+link)
    soup = BeautifulSoup(page.content,"html.parser")
    link2 = []
    for linkk in soup.findAll('a'):
        link2.append(linkk.get('href'))
    for mid in link2:
        if mid is None:
            continue
        if mid.find(".mid") != -1 and mid.find("format") == -1:
            print(mid)
            midifile = requests.get(base_url+mid)
            filepath = mid.split("/")[-1]
            with open(filepath,'wb') as f:
                f.write(midifile.content)