import logging
import sys

# ANSI 색상 코드
RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
GRAY   = "\033[90m"

TAG_COLORS = {
    "WEB":   BLUE,
    "LLM":   CYAN,
    "NAV":   GREEN,
    "ARUCO": YELLOW,
}


class PinkyFormatter(logging.Formatter):
    def format(self, record):
        tag = getattr(record, 'tag', 'SYS')
        color = TAG_COLORS.get(tag, WHITE)
        time  = self.formatTime(record, "%H:%M:%S")
        level_color = RED if record.levelno >= logging.WARNING else GRAY
        level = f"{level_color}{record.levelname}{RESET}"
        tag_str = f"{BOLD}{color}[{tag}]{RESET}"
        msg = record.getMessage()
        return f"{GRAY}{time}{RESET} {tag_str} {level} {msg}"


def _make_logger(name: str, tag: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(PinkyFormatter())
        logger.addHandler(handler)
    logger.propagate = False

    # tag를 기본 extra로 주입하는 어댑터 반환
    return logging.LoggerAdapter(logger, {'tag': tag})


web_log   = _make_logger("pinky.web",   "WEB")
llm_log   = _make_logger("pinky.llm",   "LLM")
nav_log   = _make_logger("pinky.nav",   "NAV")
aruco_log = _make_logger("pinky.aruco", "ARUCO")
