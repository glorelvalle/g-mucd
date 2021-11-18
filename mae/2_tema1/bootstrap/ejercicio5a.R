h <- c() 
for(R in c(1000,2000,3000,4000)){
  
  muestra_original = c(1,2,3.5,4,7,7.3,8.6,12.4,13.8,18.1)
  n <- length(muestra_original)
  vars <- c()
  #mediana_recortada_original <- media_recortada(muestra_original)
  varianza_muestra_original <- var(muestra_original)
  for(i in 1:10){
    muestras_bootstrap <- sample(muestra_original, n*R, rep = TRUE)
    muestras_bootstrap <- matrix(muestras_bootstrap, nrow = n)
    
    varianzas_bootstrap <- apply(muestras_bootstrap, 2, var)
    
    vars<- append(vars,sd(varianzas_bootstrap))
  }
  
  h <- append(h,mean(vars))
}
print(h)