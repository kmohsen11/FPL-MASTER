import os
import sys
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline():
    """
    Main pipeline function to prepare data, train models, and generate predictions.
    """
    try:
        from pipeline.models import FantasyFootballPredictor, PredictionExporter  # Import your pipeline classes

        logger.info("Pipeline execution started.")
        
        # Initialize the predictor
        predictor = FantasyFootballPredictor(output_dir="predictions")
        logger.info("Preparing data...")
        predictor.prepare_data()
        
        logger.info("Training models...")
        predictor.train_models()
        
        logger.info("Generating predictions...")
        prediction_files = predictor.predict_all()
        logger.info(f"Prediction files generated: {prediction_files}")
        
        backend_output_dir = "predictions"
        os.makedirs(backend_output_dir, exist_ok=True)
        
        exporter = PredictionExporter(prediction_files, output_dir=backend_output_dir)
        backend_data = exporter.prepare_backend_predictions()
        
        # Save backend predictions to JSON
        backend_json_path = Path(backend_output_dir) / "predictions.json"
        with open(backend_json_path, "w") as f:
            json.dump(backend_data, f, indent=4)
        
        logger.info(f"Backend predictions saved to: {backend_json_path}")
        logger.info("Pipeline execution completed successfully.")
        return {"message": "Pipeline executed successfully.", "files": prediction_files}
    
    except Exception as e:
        logger.error(f"An error occurred during the pipeline execution: {str(e)}", exc_info=True)
        return {"error": str(e)}

if __name__ == "__main__":
    logger.info("Pipeline runner started.")
    result = run_pipeline()
    logger.info(f"Pipeline runner result: {result}")
