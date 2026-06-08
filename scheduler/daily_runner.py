"""일별 자동 실행 스케줄러

schedule 라이브러리를 사용해 평일 지정 시각에 파이프라인을 자동 실행한다.
settings.yaml의 data.update_time에 설정된 시각(예: "09:05")에 실행.

실행 흐름:
  1. update_today(): 오늘 종목 패턴을 ChromaDB에 추가 (RAG DB 최신화)
  2. run_daily():    LangGraph MERA 그래프 실행 → 포트폴리오 리포트 출력

--mode schedule로 실행하면 이 스케줄러가 무한 루프로 대기한다.
"""

import schedule
import time
from datetime import datetime

from loguru import logger

from config import SETTINGS
from scheduler.pipeline import MERAPipeline


def job():
    """매일 지정 시각에 실행되는 파이프라인 작업"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"=== 일별 파이프라인 시작: {now} ===")
    try:
        pipeline = MERAPipeline()
        # 1단계: 오늘 패턴을 ChromaDB에 추가 (최신 RAG 데이터 확보)
        pipeline.update_today()
        # 2단계: MERA 그래프 실행 → 포트폴리오 생성 + 리포트 저장
        report = pipeline.run_daily()
        print(report)
    except Exception as e:
        logger.error(f"파이프라인 실패: {e}")


def start_scheduler():
    """평일(월~금) 지정 시각에 job을 등록하고 스케줄러를 시작

    settings.yaml data.update_time: 실행 시각 (예: "09:05")
    60초마다 schedule.run_pending()으로 예약된 작업 확인.
    Ctrl+C로 중단할 수 있다.
    """
    run_time = SETTINGS["data"]["update_time"]

    # 평일 5일 모두 동일 시각에 등록
    schedule.every().monday.at(run_time).do(job)
    schedule.every().tuesday.at(run_time).do(job)
    schedule.every().wednesday.at(run_time).do(job)
    schedule.every().thursday.at(run_time).do(job)
    schedule.every().friday.at(run_time).do(job)

    logger.info(f"스케줄러 시작: 평일 {run_time} 실행")
    while True:
        schedule.run_pending()
        time.sleep(60)  # 1분마다 예약 작업 확인


if __name__ == "__main__":
    start_scheduler()
