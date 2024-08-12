from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone
from datetime import timedelta, date
import time
from scipy.stats import norm
from dateutil.relativedelta import relativedelta
from requests.models import Response


def get_workdays(start_date: date, end_date: date) -> int:
    """
    Assuming people only code on weekdays -- Which isn't a great assumption...
    But anyway, all _per_day stats are actually "per workday"
    """
    weekdays = 0
    delta = timedelta(days=1)
    start_date_loop: date = start_date

    while start_date_loop <= end_date:
        if start_date_loop.weekday() < 5:
            weekdays += 1
        start_date_loop += delta

    return weekdays


def convert_to_letter_grade(score):
    """
    Not used. It's... meh
    """
    grades = ['F', 'D', 'C', 'B', 'A']
    modifiers = ['-', '', '+']

    grade_idx = int(score) if score == 4 else int(score // 1)
    modifier_idx = 0

    if (score - grade_idx) >= 0.666:
        modifier_idx = 2
    elif (score - grade_idx) >= 0.333:
        modifier_idx = 1
    elif score == 4:
        modifier_idx = 1

    letter_grade = grades[grade_idx] + modifiers[modifier_idx]

    return letter_grade


def truncate_filename(repos):
    """
    Shorten filename since the list of repos can be yuge.
    """
    max_length = 230
    if len(repos) > max_length:
        repos = repos[:max_length]
        # remove any illegal characters
        # filename = ''.join(c for c in filename if c.isalnum() or c in ['_', '.', ' '])
    return repos

# get_first_day_months_ago gets the first day of the month, n months ago


def get_first_day_months_ago(months_ago) -> date:
    current_date = date.today()
    date_months_ago = current_date - relativedelta(months=months_ago)
    first_day_of_month = date(date_months_ago.year, date_months_ago.month, 1)
    return first_day_of_month


def get_first_day_of_month(date_in_month: date) -> date:
    return date_in_month.replace(day=1)


def get_last_day_months_ago(months_ago) -> date:
    current_date = date.today()
    date_months_ago = current_date - relativedelta(months=months_ago)
    first_day_of_month = date(date_months_ago.year, date_months_ago.month, 1)
    last_day_of_month = first_day_of_month + \
        relativedelta(months=1) - timedelta(days=1)
    return last_day_of_month


def sleep_until_ratelimit_reset_time(reset_epoch_time: int):
    """
    Sleep seconds from now to the future time passed in.
    This is necessary when we [frequently!] overrun our Github API rate limit.
    """
    # Convert the reset time from Unix epoch time to a datetime object in UTC
    reset_time = datetime.fromtimestamp(reset_epoch_time, tz=timezone.utc)

    # Get the current time in UTC
    now = datetime.now(tz=timezone.utc)

    # Calculate the time difference
    time_diff = reset_time - now

    # Check if the sleep time is negative, which can happen if the reset time has passed
    if time_diff.total_seconds() < 0:
        print("\tNo sleep required. The rate limit reset time has already passed.")
    else:
        print(f"\tSleeping until rate limit reset: {time_diff}")
        time.sleep(time_diff.total_seconds())
    return


def get_duration_in_days(open_date: str, close_date: str) -> float:
    """
    Returns the duration in fractions of days
    Used to calculate how long a PR is open
    We don't store the datetime object, but we do use it for partial date math
    """
    opened = datetime.strptime(open_date, '%Y-%m-%dT%H:%M:%SZ')
    closed = datetime.strptime(close_date, '%Y-%m-%dT%H:%M:%SZ')
    duration_seconds = (closed - opened).total_seconds()
    # Convert seconds to days as a float
    return round(duration_seconds / 86400, 3)


def get_end_of_last_complete_month() -> date:
    today: date = date.today()
    first_of_this_month: date = today.replace(day=1)
    last_day_of_last_month: date = first_of_this_month - timedelta(days=1)
    return last_day_of_last_month
