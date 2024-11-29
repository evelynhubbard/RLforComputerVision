import os
from datetime import datetime

DATE_TIME_FORMAT = "%Y-%m-%d_%H-%M-%S"

def add_date_time_to_path(path):
    current_date_time = datetime.now().strftime(DATE_TIME_FORMAT)
    directory, original_name = os.path.split(path)
    file_name, file_extension = os.path.splitext(original_name)
    new_name = f"{file_name}_{current_date_time}{file_extension}"
    return os.path.join(directory, new_name)