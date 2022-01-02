media_recortada <- function(v) {
  
  for(i in c(1,2)){
    retirar = c(which.max(v),which.max(v))
    v <- v[! v %in% retirar]
  }
  return(mean(v))
}

h <- c()

for(R in c(10,100,1000,2000)){
  
  muestra_original = c(1,2,3.5,4,7,7.3,8.6,12.4,13.8,18.1)
  n <- length(muestra_original)
  sds <- c()
  mediana_recortada_original <- media_recortada(muestra_original)
  for(i in 1:10){
    muestras_bootstrap <- sample(muestra_original, n*R, rep = TRUE)
    muestras_bootstrap <- matrix(muestras_bootstrap, nrow = n)
    
    medias_bootstrap <- apply(muestras_bootstrap, 2, media_recortada)
    
    sds<- append(sds,sd(medias_bootstrap))
  }
  
  h <- append(h,mean(sds))
}
print(h)

