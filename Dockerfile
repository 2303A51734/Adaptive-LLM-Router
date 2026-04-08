FROM python:3.10-slim

WORKDIR /app

# Copy all project files into the container
COPY . /app/

# Install the project and all dependencies using pyproject.toml
RUN pip install --no-cache-dir .

# Hugging Face Spaces port requirement
EXPOSE 7860

# Start the server using the shortcut we made in pyproject.toml
CMD ["server"]