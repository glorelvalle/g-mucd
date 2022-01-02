set.seed(123)
n <- 20
alpha <- 3
beta <- 6
muestra <- rbeta(n,alpha,beta)

# Estimación de pdf con estimador del núcleo
e_nucleo <- density(muestra)
plot(e_nucleo, lwd = 1.5, font.main = 1,
     main = "Compararativa función de densidad vs. estimador del núcleo",
     xlab = "Muestra", ylab = "Densidad")
curve(dbeta(x, alpha, beta), from = 0, to = 1, lty = 2, add = TRUE, col = 'darkblue', lwd = 2)  
leg <- c('f(x)', TeX(r'($\hat{f}$)'))
legend("right", legend = leg, lty=c(1,1), col=c('darkblue','black'))

# Estimación de cdf con la función empírica
df <- data.frame(muestra)
ggplot(df, aes(muestra))+
  ggtitle('Comparativa función de distribución vs. función empírica') +
  theme(plot.title = element_text(hjust = 0.5)) +
  stat_ecdf(aes(muestra, linetype='F. empírica')) +
  stat_ecdf(aes(muestra), geom = 'point') +
  geom_function(fun = pbeta, args = list(alpha, beta), size = 1, 
                col = 'magenta', aes(linetype = 'F(x)')) + 
  labs(x = 'Muestra', y = 'Distribución', linetype = '') +
  xlim(-0.01, 1.01) +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank())