import argparse
import pandas as pd
import re
import reverse_geocoder as rg


parser = argparse.ArgumentParser()
parser.add_argument('--input', dest="fin", required=True,
                    help="path to x-plane apt.dat")
args = parser.parse_args()

fin = args.fin

curr_ap = None
prev_ap = None
skip = False

aps = []

with open(fin, 'rb') as f:
    for line in f:
        try:
            line = line.strip().decode()
        except UnicodeDecodeError:
            continue
        items = re.split('\s+', line)


        if items[0] == '1':
            icao = items[4]
            name = ' '.join(items[5:]).title()
            alt = items[1]

            prev_ap = curr_ap
            curr_ap = icao

            if (not icao.isalpha()) or len(icao)!=4:
                skip = True
            elif 'closed' in name.lower() or '[x]' in name.lower():
                skip = True
            else:
                skip = False

        if items[0] == '100':
            if skip:
                continue

            if curr_ap == prev_ap:
                continue

            lat = round(float(items[9]), 5)
            lon = round(float(items[10]), 5)

            # print(icao, lat, lon, alt, name)

            ap = {
                'icao': icao,
                'lat': lat,
                'lon': lon,
                'alt': alt,
                'name': name,
            }

            aps.append(ap)
            prev_ap = curr_ap

df = pd.DataFrame(aps)
latlons = df[['lat','lon']].values.tolist()
latlons = [tuple(l) for l in latlons]
geo = rg.search(latlons)
dfgeo = pd.DataFrame(geo)
df['country'] = dfgeo['cc']
df['location'] = dfgeo['name']

df = df[['icao', 'lat', 'lon', 'alt', 'country', 'name', 'location']]
df = df.sort_values('icao')

df.to_csv('db/airports.csv', index=False)
