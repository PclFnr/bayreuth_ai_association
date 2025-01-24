# functions for the whole project
from datetime import datetime


def string_to_datetime(date_string: str) -> datetime:
    """
    Converts a date string to a datetime object.

    Parameters:
    -----------
        date_string: str
            The date string in the format '%Y-%m-%d'.

    Returns:
    --------
        datetime: The corresponding datetime object.
    """
    return datetime.strptime(date_string, "%Y-%m-%d")


def datetime_to_string(date: datetime) -> str:
    """
    Converts a datetime object to a string.

    Parameters:
    -----------
        date (datetime): The datetime object to convert.

    Returns:
    --------
        str: The date string in the format '%Y-%m-%d'.
    """
    return datetime.strftime(date, "%Y-%m-%d")


MAX_WEEK = 53
def adjust_week(week: float) -> float:
    return week / MAX_WEEK


# scaler year and week
MAX_YEAR = 2050
MIN_YEAR = 2017
def adjust_year(year: float, min_year: int, max_year: int) -> float:
    return (year - min_year)/ (max_year - min_year)


BAVARIA = "BY"
LOCATIONS = "All"
def get_holidays(data: list[dict]) -> list[str]:
    """Get all holidays from this list."""
    dates = list()
    
    for date in data:
        date_str = date["date"]["iso"]
        
        # if for any location then save it
        if date["locations"] == LOCATIONS:
            dates.append(date_str)
        else:
            for state in date["states"]:
                # if state is Bayern and no exceptions, then save it
                if state["abbrev"] == BAVARIA and state["exception"] is None:
                    dates.append(date_str)
                    break
                    
    return dates


def is_in_lecture_free(date_str: str) -> float:
    """Check if date is in lecture free."""
    # define borders of semesters
    winter_start, winter_end = "10-01", "02-10"
    summer_start, summer_end = "04-01", "07-31"
    
    # split date and create a key
    _, month, day = tuple(date_str.split("-"))
    key = month + "-" + day
    
    # check key
    return float((winter_end <= key <= summer_start) or (summer_end <= key <= winter_start))


def is_winter_semester(date_str: str) -> float:
    """Check if date is in winter semester."""
    # define borders of semesters
    winter_start, winter_end = "10-01", "03-31"
    
    # split date and create a key
    _, month, day = tuple(date_str.split("-"))
    key = month + "-" + day
    
    # check key
    return float((winter_start <= key) or (key <= winter_end))


def is_summer_semester(date_str: str) -> float:
    """Check if date is in summer semester."""
    # define borders of semesters
    summer_start, summer_end = "04-01", "09-31"
    
    # split date and create a key
    _, month, day = tuple(date_str.split("-"))
    key = month + "-" + day
    
    # check key
    return float(summer_start <= key <= summer_end)


def is_covid(date_str: str) -> float:
    """Check if date is in COVID-Period."""
    start = "2020-03-01"
    end = "2022-10-01"
    
    return float(start <= date_str <= end)


def is_christmas(date_str: str) -> float:
    """Check if university is closed this day."""
    # define borders
    christmas_pause_start, christams_pause_end  = "12-20", "01-06"
    
    # split date and create a key
    _, month, day = tuple(date_str.split("-"))
    key = month + "-" + day 
    
    return float((christmas_pause_start <= key) or (key <= christams_pause_end))


