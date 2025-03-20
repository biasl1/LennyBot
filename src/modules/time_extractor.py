import re
import datetime
import logging
import time

def extract_time(text, base_timestamp=None):
    """Extract time from text message."""
    if not base_timestamp:
        base_timestamp = time.time()
    
    text = text.lower()
    extracted_time = None
    time_description = None
    
    # Relative time patterns (most common in chat)
    rel_patterns = [
        (r'in (\d+) (minute|minutes|min|mins)', lambda m: int(m.group(1)) * 60),
        (r'in (\d+) (hour|hours|hr|hrs)', lambda m: int(m.group(1)) * 3600),
        (r'in (\d+) (second|seconds|sec|secs)', lambda m: int(m.group(1))),
        (r'in (a|one) (minute|min)', lambda m: 60),
        (r'in (a|one) (hour|hr)', lambda m: 3600)
    ]
    
    # Absolute time patterns
    abs_patterns = [
        (r'at (\d{1,2})(?::(\d{2}))?(?:\s*(am|pm))?', parse_clock_time),
        (r'(\d{1,2})(?::(\d{2}))?(?:\s*(am|pm))', parse_clock_time)
    ]
    
    # Try relative patterns first (most common in chat)
    for pattern, time_func in rel_patterns:
        match = re.search(pattern, text)
        if match:
            seconds_to_add = time_func(match)
            extracted_time = base_timestamp + seconds_to_add
            time_description = match.group(0)
            break
    
    # Try absolute time patterns if no relative match
    if not extracted_time:
        for pattern, time_func in abs_patterns:
            match = re.search(pattern, text)
            if match:
                extracted_time = time_func(match, base_timestamp)
                time_description = match.group(0)
                break
    
    # Default to 1 minute if no time found
    if not extracted_time:
        extracted_time = base_timestamp + 60
        time_description = "in 1 minute (default)"
    
    return extracted_time, time_description

def parse_clock_time(match, base_timestamp):
    """Parse clock time formats."""
    base_dt = datetime.datetime.fromtimestamp(base_timestamp)
    hour = int(match.group(1))
    minute = int(match.group(2)) if match.group(2) else 0
    period = match.group(3).lower() if match.group(3) else None
    
    # Handle 12-hour clock
    if period == 'pm' and hour < 12:
        hour += 12
    elif period == 'am' and hour == 12:
        hour = 0
    
    # Create target datetime
    target_dt = base_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
    
    # If time is in the past, assume tomorrow
    if target_dt.timestamp() < base_timestamp:
        target_dt += datetime.timedelta(days=1)
    
    return target_dt.timestamp()