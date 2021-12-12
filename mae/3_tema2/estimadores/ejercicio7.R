alpha <- 3
beta <- 6

# Función de densidad
graf1 <- ggplot()+
  ggtitle(TeX(r'(Función de densidad $Beta(\alpha = 3, \beta = 6)$)')) +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_function(fun = dbeta, args = list(alf,bet),size = 1.1, col = 'darkblue')+
  labs(x="x",y="Densidad")+
  xlim(-0.1, 1.1)+
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank())

# Función de distribución
graf2 <- ggplot()+
  ggtitle(TeX(r'(Función de distribución $Beta(\alpha = 3, \beta = 6)$)')) +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_function(fun = pbeta, args = list(alf,bet),size = 1.1, col = 'magenta') +
  labs(x="x",y="Distribución")+
  xlim(-0.1, 1.1)+
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank())


graf1+graf2