FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK corpora at build time so the container starts instantly
RUN python -c "\
import nltk; \
nltk.download('wordnet'); \
nltk.download('averaged_perceptron_tagger_eng'); \
nltk.download('punkt_tab'); \
nltk.download('omw-1.4'); \
nltk.download('stopwords'); \
nltk.download('brown'); \
"

COPY app.py paraphraser.py ./

EXPOSE 8080

# Cloud Run injects $PORT (always 8080); pass it through to Streamlit
CMD ["sh", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true"]
