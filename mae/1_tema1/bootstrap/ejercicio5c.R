set.seed(100)

# Parámetros
R <- 2000
m <- 100
alfa <- 0.05
n <- length(muestra_original)

muestras_bootstrap <- sample(muestra_original, n*R, rep = TRUE)
muestras_bootstrap <- matrix(muestras_bootstrap, nrow = n)
varianza_original <-  var(muestra_original)
varianzas_bootstrap <- apply(muestras_bootstrap, 2, var)
T_bootstrap <- sqrt(n) * (varianzas_bootstrap - varianza_original)
ic_min <- varianza_original -  quantile(T_bootstrap, 1-alfa/2)/sqrt(n)
ic_max  <- varianza_original -  quantile(T_bootstrap, alfa/2)/sqrt(n)
print(ic_max)
print(ic_min)
