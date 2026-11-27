# The prediction service in a container (week 45).
#   python src/train_final_model.py            # the image copies models/rf_final.joblib
#   docker build -t no1-load-forecast:1.0 .
#   docker run --rm -p 8000:8000 no1-load-forecast:1.0
# The Python version and the pins in service/requirements.txt match the
# environment that trained the model, because a saved scikit-learn model is
# tied to the library version that wrote it.
FROM python:3.14-slim
WORKDIR /app
COPY service/requirements.txt service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt
COPY src/config.py src/features.py src/
COPY service/ service/
COPY models/rf_final.joblib models/
EXPOSE 8000
# 0.0.0.0, not 127.0.0.1: otherwise nothing outside the container can reach it.
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
