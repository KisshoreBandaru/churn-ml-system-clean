import joblib
import logging
from collections import Counter
from app.config import MODEL_PATH

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self, model_path=MODEL_PATH):
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)

        logger.info(f"Model expects {self.model.n_features_in_} features")

        # --- MONITORING STATE ---
        self.model_path = model_path
        self.request_count = 0
        self.prediction_counter = Counter()

        logger.info("Model loaded successfully")

    def predict(self, data):
        """
        data = ChurnInput (validated by schema)
        """

        # --- TRANSLATION (THIS WAS MISSING) ---
        features = [[
            data.tenure,
            data.monthly_charges,
            data.total_charges,
            data.support_calls
        ]]

        logger.info(f"Input features received: {features}")

        # --- PREDICTION ---
        prediction = int(self.model.predict(features)[0])

        # --- MONITORING ---
        self.request_count += 1
        self.prediction_counter[prediction] += 1

        logger.info(
            f"Model={self.model_path} | "
            f"TotalRequests={self.request_count} | "
            f"Prediction={prediction} | "
            f"Distribution={dict(self.prediction_counter)}"
        )

        return prediction
