from src.mensa_ml import datetime_to_string, string_to_datetime
from datetime import timedelta
import pandas as pd
import numpy as np

import requests
from sklearn.preprocessing import OneHotEncoder
from src.utils import is_christmas, is_covid, is_summer_semester, is_winter_semester, is_in_lecture_free
from tqdm import tqdm
from time import sleep

# use one hot encoder for day of week and month
day_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
month_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# Fit the data
unique_days = np.array(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).reshape(-1, 1)
unique_months = np.array([
    "January", 
    "February", 
    "March", 
    "April", 
    "May", 
    "June", 
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]).reshape(-1, 1)
day_encoder.fit(unique_days) # beverages_df[['Day']]
month_encoder.fit(unique_months) # beverages_df[['Month']]


# define borders
date_from_str = "2024-10-18"
date_to_str = "2024-12-31"

# get convert to datetime format
date_from = string_to_datetime(date_string=date_from_str)
date_to = string_to_datetime(date_string=date_to_str)

# get the full timeline
full_timeline = [date_from + timedelta(n) for n in range(int((date_to - date_from).days) + 1)]
full_timeline_str = list(map(lambda elem: datetime_to_string(elem), full_timeline))

# read the old DataFrame
old_df = pd.read_csv("data/extended_data.csv", index_col=0)
last_date_str = old_df.index.max()
holidays = pd.read_csv("data/holidays.csv", index_col=0)

# create a dataframe for new data
# df = pd.DataFrame(columns=old_df.columns)
df = pd.read_csv("data/future_data.csv", index_col=0)

# function to adjust week
MAX_WEEK = 53
def adjust_week(week: float) -> float:
    return week / MAX_WEEK

def get_weather_for_date(date_str: str) -> list[float]:

    lat = 49.94564
    lon = 11.571335
    api_key = "0439182e552334891ac9fde1ae3ba9fa" #os.getenv(key="API_KEY_WEATHER")
    units = "metric"

    call_string = f"https://api.openweathermap.org/data/3.0/onecall/day_summary?lat={lat}&lon={lon}&date={date_str}&appid={api_key}&units={units}"

    response = requests.get(url=call_string)

    # Check the status code
    if response.status_code != 200:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        # get data
        response_dict = response.json()

        # extract important features
        temp_min = response_dict["temperature"]["min"]
        temp_max = response_dict["temperature"]["max"]
        temp = (response_dict["temperature"]["morning"] + response_dict["temperature"]["afternoon"] + response_dict["temperature"]["evening"]) / 3

        pressure = response_dict["pressure"]["afternoon"]

        humidity = response_dict["humidity"]["afternoon"]

        precipitation = response_dict["precipitation"]["total"]

        wind_speed = response_dict["wind"]["max"]["speed"]
        wind_deg = response_dict["wind"]["max"]["direction"]

        # return features as array
        return [temp, temp_min, temp_max, pressure, humidity, wind_speed, wind_deg, precipitation]

# define prices
## coffee
student_price_coffee = old_df.loc[last_date_str, "Student Price Coffee"]
worker_price_coffee = old_df.loc[last_date_str, "Worker Price Coffee"]
guest_price_coffee = old_df.loc[last_date_str, "Guest Price Coffee"]
## mil coffee
student_price_milk_coffee = old_df.loc[last_date_str, "Student Price Milk Coffee"]
worker_price_milk_coffee = old_df.loc[last_date_str, "Worker Price Milk Coffee"]
guest_price_milk_coffee = old_df.loc[last_date_str, "Guest Price Milk Coffee"]
## cocoa
student_price_cocoa = old_df.loc[last_date_str, "Student Price Cocoa"]
worker_price_cocoa = old_df.loc[last_date_str, "Worker Price Milk Cocoa"]
guest_price_cocoa = old_df.loc[last_date_str, "Guest Price Milk Cocoa"]
## tee
student_price_tee = old_df.loc[last_date_str, "Student Price Tee"]
worker_price_tee = old_df.loc[last_date_str, "Worker Price Tee"]
guest_price_tee = old_df.loc[last_date_str, "Guest Price Tee"]
## coffee time
student_price_coffee_time = old_df.loc[last_date_str, "Student Price Coffee Time"]
worker_price_coffee_time = old_df.loc[last_date_str, "Worker Price Coffee Time"]
guest_price_coffee_time = old_df.loc[last_date_str, "Guest Price Coffee Time"]
## taxes
student_tax = old_df.loc[last_date_str, "Student Tax"]
worker_tax = old_df.loc[last_date_str, "Worker Tax"]
guest_tax = old_df.loc[last_date_str, "Guest Tax"]

# cycle to get data
try:
    for date_str, date in tqdm(list(zip(full_timeline_str, full_timeline, strict=True)), desc="Processing dates"):

        if date_str in df.index:
            continue

        # Get day name
        day = date.strftime('%A')

        # Get month name
        month = date.strftime('%B')

        # use encoders
        _month = list(month_encoder.transform([[month]])[0])
        _day = list(day_encoder.transform([[day]])[0])

        # Get week and year
        week = date.isocalendar().week
        year = date.year

        # Get weather data
        wheather_data = get_weather_for_date(date_str=date_str)

        # get data about holidays
        is_holiday_mark = float(is_christmas(date_str) or date_str in holidays)
        is_lecture_free = is_in_lecture_free(date_str)
        is_covid_mark = is_covid(date_str)
        is_winter = is_winter_semester(date_str)
        is_summer = is_summer_semester(date_str)

        # save all the data
        data = [
            # targets
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            # prices 
            ## coffee
            student_price_coffee,
            worker_price_coffee,
            guest_price_coffee,
            # milk coffee
            student_price_milk_coffee,
            worker_price_milk_coffee,
            guest_price_milk_coffee,
            # cocoa
            student_price_cocoa,
            worker_price_cocoa,
            guest_price_cocoa,
            # tee
            student_price_tee,
            worker_price_tee,
            guest_price_tee,
            # coffee time
            student_price_coffee_time,
            worker_price_coffee_time,
            guest_price_coffee_time,
            # taxes
            student_tax, 
            worker_tax, 
            guest_tax, 
            # time
            week, year
        ] + _day + _month + wheather_data + [is_holiday_mark, is_lecture_free, is_covid_mark, is_winter, is_summer]
        
        df.loc[date_str] = data

        # sleep for some time - to not get banned
        sleep(1.5)
except Exception:
    pass


df.sort_index(inplace=True)
df.to_csv("data/future_data.csv")