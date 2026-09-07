# Historical reproduction environment. Julia 1.6.1 is the version the thesis
# experiments ran under; the pinned Manifest resolves against it.
FROM julia:1.6.1

WORKDIR /app

# Dependencies first, so source edits do not invalidate the resolved environment.
COPY Project.toml Manifest.toml ./
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

COPY src ./src
COPY scripts ./scripts
COPY experiments ./experiments
COPY test ./test
COPY data ./data

CMD ["julia", "--project=.", "-e", "using Pkg; Pkg.test()"]
