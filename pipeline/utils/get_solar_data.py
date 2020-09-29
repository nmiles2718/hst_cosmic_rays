import datetime as dt
import json
from urllib import request

import pandas as pd

NOAA_URL = 'https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json'

def get_solar_data(url):
    """Download the most recent solar data
    """
    response = request.urlopen(url)
    if response.status == 200:
        data = json.loads(response.read())
    else:
        print(f'Invalid response! HTTP Status Code: {response.status}')
    df = pd.DataFrame(data)
    dates = [dt.datetime.strptime(val, '%Y-%m') for val in df['time-tag']]
    df.index = pd.DatetimeIndex(dates)
    return df

if __name__ == "__main__":
    df = get_solar_data(NOAA_URL)
    todays_date = dt.datetime.today().strftime('%b%d_%Y')
    fout = f'noaa_solar_indices_{todays_date}.txt'
    print(f"Saving the results...\nFilename: {fout}")
    df.to_csv(fout, header=True, index=True)