import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
import os

def setup_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)

    today_str = datetime.now().strftime("%Y%m%d")
    log_file_path = os.path.join(log_dir, f"{today_str}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 避免重复添加 Handler
    if not any(isinstance(h, TimedRotatingFileHandler) for h in logger.handlers):
        file_handler = TimedRotatingFileHandler(
            filename=log_file_path,
            when='midnight',
            interval=1,
            backupCount=0,
            encoding='utf-8',
            utc=False
        )
        file_handler.suffix = "%Y%m%d.log"

        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 控制台输出（可选）
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger
