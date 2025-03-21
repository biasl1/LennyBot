from typing import Tuple, Dict, List, Any, Optional
import re
import datetime
import logging
import time

def extract_time(text: str, reference_time: float = None) -> Tuple[float, str]:
    """
    Extract time information from natural language text with enhanced accuracy.
    Returns (timestamp, human-readable string)
    """
    if reference_time is None:
        reference_time = time.time()
    
    # Make sure text is lowercase for consistent matching
    text = text.lower()
    
    # Define patterns with more precise recognition
    # Minutes pattern (e.g., "in 5 minutes", "in a minute")
    min_pattern = r'in\s+(\d+|a|an|one|two|three|five|ten|fifteen|twenty|thirty|forty|fifty|sixty|half)\s*(minute|min|minutes|mins)'
    min_match = re.search(min_pattern, text)
    
    if min_match:
        # Convert text numbers to digits
        num_map = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "five": 5, 
                   "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, 
                   "forty": 40, "fifty": 50, "sixty": 60, "half": 0.5}
        
        minutes_str = min_match.group(1)
        minutes = num_map.get(minutes_str, None)
        
        if minutes is None:
            try:
                minutes = int(minutes_str)
            except ValueError:
                minutes = 1  # Default to 1 minute if parsing fails
        
        future_time = reference_time + (minutes * 60)
        return future_time, f"in {minutes} minute{'s' if minutes != 1 else ''}"
    
    # Hours pattern (e.g., "in 2 hours", "in an hour")
    hour_pattern = r'in\s+(\d+|a|an|one|two|three|four|five|six|twelve|half)\s*(hour|hr|hours|hrs)'
    hour_match = re.search(hour_pattern, text)
    
    if hour_match:
        num_map = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, 
                   "five": 5, "six": 6, "twelve": 12, "half": 0.5}
        
        hours_str = hour_match.group(1)
        hours = num_map.get(hours_str, None)
        
        if hours is None:
            try:
                hours = int(hours_str)
            except ValueError:
                hours = 1  # Default to 1 hour if parsing fails
        
        future_time = reference_time + (hours * 3600)
        return future_time, f"in {hours} hour{'s' if hours != 1 else ''}"
    
    # Time of day pattern (e.g., "at 3pm", "at 15:30")
    time_of_day_pattern = r'at\s+(\d+)(?::(\d+))?\s*(am|pm)?'
    time_match = re.search(time_of_day_pattern, text)
    
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2)) if time_match.group(2) else 0
        ampm = time_match.group(3)
        
        # Convert to 24-hour format if needed
        if ampm:
            if ampm.lower() == 'pm' and hour < 12:
                hour += 12
            elif ampm.lower() == 'am' and hour == 12:
                hour = 0
        elif hour < 7:  # Assume PM for times like "at 5" (5pm, not 5am)
            hour += 12
            
        # Get current time components
        now = datetime.datetime.fromtimestamp(reference_time)
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # If the time has already passed today, assume tomorrow
        if target.timestamp() <= reference_time:
            target = target + datetime.timedelta(days=1)
            
        return target.timestamp(), f"at {hour}:{minute:02d}{'am' if hour < 12 else 'pm'}"
    
    # Default: if no clear time, set reminder for 1 minute from now
    future_time = reference_time + 60
    return future_time, f"in 1 minute (default)"

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

def get_current_time_formatted():
    """Return the current time in a nicely formatted way."""
    now = datetime.datetime.now()
    
    # Format: "3:45 PM, Tuesday, March 21, 2025"
    time_str = now.strftime("%I:%M %p, %A, %B %d, %Y")
    
    # Remove leading zero from hour if present
    if time_str.startswith("0"):
        time_str = time_str[1:]
        
    return time_str

