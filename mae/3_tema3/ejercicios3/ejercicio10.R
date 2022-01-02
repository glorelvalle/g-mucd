load(url('http://verso.mat.uam.es/~joser.berrendero/datos/combustible.RData'))
head(fuel2001,10)
p_comp <- prcomp(fuel2001[,-c(2)], scale=TRUE)
data <- data.frame(
  c1=p_comp$x[ ,1], 
  c2=p_comp$x[ ,2])

ggplot(data, aes(x=c1, y=c2)) +
  geom_point(colour='darkblue') +
  ggtitle('Componentes principales') +
  xlab('Primera componente') +
  ylab('Segunda componente') +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank())

# Excluimos la variable a predecir y creamos un df con las dos componentes
p_comp <- prcomp(fuel2001[,-c(2)], scale=TRUE)
data <- data.frame(
  c1=p_comp$x[ ,1], 
  c2=p_comp$x[ ,2])

# Plot de las componentes principales
ggplot(data, aes(x=c1, y=c2)) +
  geom_point(colour='darkblue') +
  ggtitle('Componentes principales') +
  xlab('Primera componente') +
  ylab('Segunda componente') +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank())

# Cálculo de la varianza acumulada
var_acumulada <- cumsum(p_comp$sdev^2)/sum(p_comp$sdev^2)
#sdev <- data.frame(indice = 1:6, var_acumulada)

modelo_completo <- lm(FuelC ~ Drivers + Income + Miles + MPC + Pop + Tax,
                      data=fuel2001)
summary(modelo_completo)
modelo_reducido <- lm(FuelC ~ Drivers + Miles + Pop,
                      data=fuel2001)
anova(modelo_reducido)
anova(modelo_reducido, modelo_completo) 


data <- data.frame(y=fuel2001$FuelC, 
                    x1=fuel2001$Drivers, 
                    x2=fuel2001$Income, 
                    x3=fuel2001$Miles, 
                    x4=fuel2001$MPC, 
                    x5=fuel2001$Pop, 
                    x6=fuel2001$Tax)

modelo_forward <- leaps::regsubsets(y ~ ., data=data, method='forward')
s_forward <- summary(modelo_forward)
s_forward

plot(s_forward$bic ,xlab ='Número de variables', 
     ylab="BIC", type="l", col='blue', main='Modelos con criterio BIC')
which.min(s_forward$bic)

s_forward$outmat[which.min(s_forward$bic), ]

library(glmnet)
x <- as.matrix(datos[, -1])
y <- datos[, 1]

modelo_lasso <- glmnet(x, y, alpha=1)
lasso <- cv.glmnet (x, y, alpha=1)
lambda.lasso = lasso$lambda.1se

plot(modelo_lasso, xvar='lambda', label=TRUE)
abline(v = log(lambda.lasso), col='magenta', lty=2, lwd=2)


plot(lasso)
i <- which(min(lasso$cvm) == lasso$cvm)
abline(h = lasso$cvm[i] + lasso$cvsd[i], lty=2)

lambda.lasso = lasso$lambda.1se
final_lasso <- glmnet(x, y, alpha=1, lambda=lambda.lasso)
final_lasso
coef(final_lasso)
c <- modelo_lasso$beta[1:6, modelo_lasso$lambda == lambda.lasso]
c[c != 0]

x <- as.matrix(datos[, -1])
y <- datos[, 1]

modelo_ridge <- glmnet(x, y, alpha=0)
ridge <- cv.glmnet (x, y, alpha=0)
lambda.ridge = ridge$lambda.1se

plot(modelo_ridge, xvar='lambda', label=TRUE)
abline(v = log(lambda.ridge), col='green', lty=2, lwd=2)

plot(ridge)
i <- which(min(ridge$cvm) == ridge$cvm)
abline(h = ridge$cvm[i] + ridge$cvsd[i], lty=2)

lambda.ridge = ridge$lambda.1se
final_ridge <- glmnet(x, y, alpha=1, lambda=lambda.ridge)
final_ridge
coef(final_ridge)
c <- modelo_ridge$beta[1:6, modelo_ridge$lambda == lambda.ridge]
c[c != 0]