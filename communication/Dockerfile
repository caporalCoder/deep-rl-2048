FROM python:3

# Install project dependencies
ADD requirements.txt /app
RUN pip install -r requirements.txt

# Run project
ADD /app
CMD python3 main.py