# Load data 
data <- cbind(c(1.6907, 59, 6),
              c(1.7242, 60, 13),
              c(1.7552, 62, 18),
              c(1.7842, 56, 28),
              c(1.8113, 63, 52),
              c(1.8369, 59, 53),
              c(1.8610, 62, 61),
              c(1.8839, 60, 60))

# Create dataframe
df <- as.data.frame(t(data))
colnames(df) <- c("Dose", "Insects", "Deaths")
df

# Get dose values
value <- df$Dose

# Fit model (deaths vs survivors)
model <- glm(cbind(df$Deaths, df$Insects-df$Deaths) ~ value, family=binomial)

# Predict
predicted <- predict(model, data.frame(value=1.8), type='response')

# Print
cat("Probabilidad de que el escarabajo muera: ", predicted)

summary(model)

# Plot
new_data <- data.frame(Dose = seq(min(df$Dose), max(df$Dose), len=8))
pred_value <- predict(model, new_data, type='response')
plot(value, df$Deaths, pch=16)
lines(new_data$Dose, pred_value, col='red', lwd=2)