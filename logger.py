import logging

# --- Logging Setup ---
class LogColors:
    DEBUG = "\033[94m"    # Blue
    INFO = "\033[92m"     # Green
    WARNING = "\033[93m"  # Yellow
    ERROR = "\033[91m"    # Red
    CRITICAL = "\033[95m" # Magenta
    RESET = "\033[0m"     # Reset

class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = {
            logging.DEBUG: LogColors.DEBUG,
            logging.INFO: LogColors.INFO,
            logging.WARNING: LogColors.WARNING,
            logging.ERROR: LogColors.ERROR,
            logging.CRITICAL: LogColors.CRITICAL
        }.get(record.levelno, "")
        record.msg = f"{color}{record.msg}{LogColors.RESET}"
        return super().format(record)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(ColorFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)
