FROM julia:latest

USER root

#Julia Dependencies
ENV JULIA_PKGDIR /opt/julia
ENV JULIA_DEPOT_PATH /opt/julia

RUN mkdir -p /opt/julia && chmod 777 /opt/julia
RUN mkdir -p /app && chmod 777 /app

COPY . /app/
RUN julia /app/requirements.jl
RUN julia /app/BNN.jl

CMD julia /app/main.jl $ITERS $M $K $L
