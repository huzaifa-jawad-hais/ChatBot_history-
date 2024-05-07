# 
FROM python:3.10.10

ENV OPENAI_API_KEY=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
ENV PINECONE_API_KEY=ed3c3f10-6a2e-4091-bd36-3f0fbc645f72
ENV APP_HOME /app
WORKDIR $APP_HOME

COPY . ./

# 
RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN pip install --upgrade pip

RUN pip freeze
# 

# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]