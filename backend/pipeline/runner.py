import schedule
import time
import logging
from app.update_predictions import run_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def schedule_pipeline():
    """
    Schedule the pipeline to run on Monday nights at 11 PM.
    """
    schedule.every().monday.at("23:00").do(run_pipeline)  # 11 PM Monday

    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute for scheduled tasks

if __name__ == "__main__":
    logger.info("Scheduler started. The pipeline will run every Monday night at 11 PM.")
    try:
        schedule_pipeline()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped.")
