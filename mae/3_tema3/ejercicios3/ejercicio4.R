# Guardado de los datos
scrs <- c(42644.00, 8352.28, 36253.69, 36606.19, 7713.13, 762.55, 32700.17, 761.41)
vars <- c('Indep.', 'x1', 'x2', 'x3', 'x1 y x2', 'x1 y x3', 'x2 y x3', 'x1, x2 y x3')
datos <- data.frame(scrs=scrs, vars=vars)
# Datos completos
n <- 20
p <- 3
scr <- tail(scrs, n=1)
sce <- scrs[1]-scr
df <- c(p, n-p-1)
sum_sq <- c(sce, scr)
mean_sq <- sum_sq/df
f_value <- c(mean_sq[1]/mean_sq[2], NA)
pr <- c(pf(f_value[1], df1=p, df2=n-p-1, lower.tail=FALSE), NA)

anova <- data.frame("anova data"=c('scrs', 'Residuals'), 'Df'=df, 'Sum Sq'=sum_sq, 'Mean Sq'=mean_sq, 'F value'=f_value, 'Pr(>F)'=pr)
anova

k <- 1
f_value <- ((scrs[6]-scr)/k / (scr/(n-p-1)))
pr <- pf(f_value, df1=k, df2=n-p-1, lower.tail=FALSE)
pr
k <- 2
f_value <- ((scrs[3]-scr)/k / (scr/(n-p-1)))
pr <- pf(f_value, df1=k, df2=n-p-1, lower.tail=FALSE)
pr

#grafico<- ggplot(data = anova, aes(x=vars, y=scrs, color=scrs)) +
#  geom_boxplot() +
#  theme_bw()
#grafico

# Análisis de la varianza
#anova <- aov (lm(datos$scrs ~ datos$vars))
#summary(anova)
#names(anova)
#fm$residuals