from pipeline.models import FantasyFootballPredictor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Running predictions only...")

    # Create the predictor instance
    predictor = FantasyFootballPredictor(output_dir="predictions")

    # Ensure data is prepared before predicting
    predictor.prepare_data()

    # Run predictions
    prediction_files = predictor.predict_all()

    logger.info(f"Predictions saved at: {prediction_files}")


