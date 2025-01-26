# ## Group Name: MLOPS Group 63

# ## Group Member Names:
#     1.   Sivakumar G - 2023aa05486
#     2.   Pabbisetty Jayakrishna - 2023aa05487
#     3.   Ravi shankar S - 2023aa05488
#     4.   Srivatsan V R - 2023aa05962


# Use the official Python base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask app and model file into the container
COPY app.py .
COPY best_rf_model.joblib .

# Expose the port Flask will run on
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app.py"]
