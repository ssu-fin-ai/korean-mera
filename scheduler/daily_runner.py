"""일별 자동 실행 스케줄러"""

import schedule
import time
from datetime import datetime

from loguru import logger

from config import SETTINGS
from scheduler.pipeline import MERAPipeline


def job():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"=== 일별 파이프라인 시작: {now} ===")
    try:
        pipeline = MERAPipeline()
        pipeline.update_today()
        report = pipeline.run_daily()
        print(report)
    except Exception as e:
        logger.error(f"파이프라인 실패: {e}")


def start_scheduler():
    run_time = SETTINGS["data"]["update_time"]
    schedule.every().monday.at(run_time).do(job)
    schedule.every().tuesday.at(run_time).do(job)
    schedule.every().wednesday.at(run_time).do(job)
    schedule.every().thursday.at(run_time).do(job)
    schedule.every().friday.at(run_time).do(job)

    logger.info(f"스케줄러 시작: 평일 {run_time} 실행")
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    start_scheduler()
