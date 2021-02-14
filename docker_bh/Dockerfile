FROM julia:latest
WORKDIR /app
COPY . .
RUN julia requirements.jl
RUN julia BNN.jl
RUN mkdir -p /app

CMD julia main.jl
