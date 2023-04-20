from datetime import datetime
import os

class Logger:
    logger_path: str
    
    @classmethod
    def set_logger_path(cls, path: str) -> None:
        try:
            with open(path, "w") as log_file:
                log_file.write(f"Logger created at -> {path}\n")
        except FileNotFoundError:
            directory_path = "".join(path.rpartition("/")[:-1])
            os.makedirs(directory_path)
            
            with open(path, "w") as log_file:
                log_file.write(f"Logger created at -> {path}\n")
            
        cls.logger_path = path
    
    @classmethod
    def info(cls, message):
        log_time = datetime.now().strftime("%Y/%m/%d/ %H.%M.%S")
        with open(cls.logger_path, "a") as log_file:
            log_file.write(f"[{log_time}][INFO] {message}\n")    

    @classmethod
    def warn(cls, message):
        log_time = datetime.now().strftime("%Y/%m/%d/ %H.%M.%S")
        with open(cls.logger_path, "a") as log_file:
            log_file.write(f"[{log_time}][WARN] {message}\n")    

    @classmethod
    def error(cls, message):
        log_time = datetime.now().strftime("%Y/%m/%d/ %H.%M.%S")
        with open(cls.logger_path, "a") as log_file:
            log_file.write(f"[{log_time}][ERROR] {message}\n")    

