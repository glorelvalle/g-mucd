set.seed(123)
n<-20
m<-200
alpha<-3
beta<-6

error1<-NULL
error2<-NULL

for (i in 1:m){
  muestra<-rbeta(n,alpha,beta)
  
  ks_F<-ks.test(muestra,"pbeta",alpha,beta)
  
  error1<-c(error1,ks_F$statistic)
  
  estimador<-dbeta(muestra,alpha,beta)
  nucleo<-density(muestra,n=20)$y
  
  ks_f<-ks.test(nucleo,estimador)
  
  error2<-c(error2,ks_f$statistic)
  
}

mean(error1)
mean(error2)

