import os
from datetime import datetime

DATE_TIME_FORMAT = "%Y-%m-%d_%H-%M-%S"

def add_date_time_to_path(path, extension):
    current_date_time = datetime.now().strftime(DATE_TIME_FORMAT)
    # Insert the timestamp immediately after the folder name in the path
    new_name = f"{path}/{current_date_time}{extension}"
    # Create the folder if it doesn't exist
    os.makedirs(os.path.dirname(new_name), exist_ok=True)
    return new_name