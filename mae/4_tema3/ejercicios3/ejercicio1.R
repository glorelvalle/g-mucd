library(ggplot2)
library(dplyr)
library(KernSmooth)

# lectura del fichero

df <- read.table("https://matematicas.uam.es/~joser.berrendero/datos/Datos-geyser.txt",
                 header=FALSE, skip=1, sep=' ', col.names=c(' ', 'D', 'Y', 'X'),
                 colClasses=c('numeric', 'numeric', 'numeric', 'numeric'))
# guardado en dataframe
df <- df[,c('D', 'Y', 'X')]

# ajuste localmente lineal
ajuste <- with(df, locpoly(X, Y, degree=1, bandwidth=dpill(X, Y), gridsize=107))
ajuste

# plot de ambas
df %>%
  mutate(curva=ajuste$y) %>%
  ggplot() +
  ggtitle('Estimador localmente lineal vs. Nadaraya-Watson') +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_point(aes(X, Y)) +
  geom_line(aes(ajuste$x, curva), color='cadetblue', size=1.1) +
  geom_label(aes(x=2, y=80, label='Localmente lineal'), color='cadetblue') +
  geom_smooth(formula = y ~ x, mapping=aes(X, Y), method='loess', se=FALSE, span=0.25, method.args=list(degree=0), col='darkorange1') +
  geom_label(aes(x=2, y=85, label='Nadayara-Watson'), color='darkorange2')+
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank())

