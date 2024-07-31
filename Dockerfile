FROM python:3.9 

ADD main.py .
ADD action_extractor.py .

RUN pip install fastapi uvicorn langchain langchain-community langchain-experimental text_generation
CMD ["python", "main.py"] 