#---------------------------
#Load Boston Housing Dataset
#---------------------------
data = load("boston.jld")["boston"]

# Generating test/training sets:
nrow, ncol = size(data)
nrow_test  = div(nrow, 3)
nrow_train = nrow - nrow_test

x = data[:,1:13]
y = data[:,14]
y_raw = y

dx = fit(ZScoreTransform, x, dims=1)
StatsBase.transform!(dx, x)
dy = fit(ZScoreTransform, y; dims=1)
StatsBase.transform!(dy, y);

x_raw = x
x = transpose(x);