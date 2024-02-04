from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone
from datetime import timedelta, date
import time
from scipy.stats import norm
from dateutil.relativedelta import relativedelta
from requests.models import Response


def get_workdays(start_date: datetime, end_date: datetime):
    """
    Assuming people only code on weekdays -- Which isn't a great assumption...
    But anyway, all _per_day stats are actually "per workday"
    """
    weekdays = 0
    delta = timedelta(days=1)

    while start_date <= end_date:
        if start_date.weekday() < 5:
            weekdays += 1
        start_date += delta

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


def get_date_months_ago(months_ago) -> datetime:
    current_date = datetime.now()
    date_months_ago = current_date - relativedelta(months=months_ago)
    return date_months_ago


def sleep_until_ratelimit_reset_time(reset_epoch_time):
    """
    Sleep seconds from now to the future time passed in. 
    This is necessary when we [frequently!] overrun our Github API rate limit.

    """
    # Convert the reset time from Unix epoch time to a datetime object
    reset_time = datetime.utcfromtimestamp(reset_epoch_time)

    # Get the current time
    now = datetime.utcnow()

    # Calculate the time difference
    time_diff = reset_time - now

    # Check if the sleep time is negative, which can happen if the reset time has passed
    if time_diff.total_seconds() < 0:
        print("\tNo sleep required. The rate limit reset time has already passed.")
    else:
        time_diff = timedelta(seconds=int(time_diff.total_seconds()))
        # Print the sleep time using timedelta's string representation
        print(f"\tSleeping until rate limit reset: {time_diff}")
        time.sleep(time_diff.total_seconds())
    return


def get_duration_in_days(open_date: str, close_date: str) -> float:
    """
    Returns the duration in fractions of days
    Used to calculate how long a PR is open
    """
    opened = datetime.strptime(open_date, '%Y-%m-%dT%H:%M:%SZ')
    closed = datetime.strptime(close_date, '%Y-%m-%dT%H:%M:%SZ')
    duration_seconds = (closed - opened).total_seconds()
    return duration_seconds / 86400  # Convert seconds to days as a float
