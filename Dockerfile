FROM python:3.10-slim

# Working directory
WORKDIR /app/src

# Install the Python dependencies
COPY ./requirements.txt /app/requirements.txt

RUN echo "Installing Python dependencies" && \
    pip install --no-cache-dir -r /app/requirements.txt --quiet


COPY ./src /app/src
RUN mkdir /app/docs

# Keep the container running
CMD ["bash"]
