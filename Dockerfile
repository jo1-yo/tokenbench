# Container for the routing proxy (tokenbench router). Build: docker build -t tokenbench .
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY tokenbench ./tokenbench
COPY configs ./configs
COPY data ./data
RUN pip install --no-cache-dir .
EXPOSE 8080
# Pass the key at runtime: docker run -e NEBIUS_API_KEY=... -p 8080:8080 tokenbench
CMD ["tokenbench", "router", "--config", "configs/nebius.yaml", "--host", "0.0.0.0", "--port", "8080"]
