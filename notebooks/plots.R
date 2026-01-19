library(tidyverse)
library(ggplot2)

data <- read_csv('model-stats_loss_accuracy.csv')

data <- data |>
  rename(`Model Type` = Model,
          Balance =  `Balance Type`,
          Epochs = epochs,
          `Accuracy Rate` = accuracy,
          Loss = loss) |>
  mutate(`Model Type` = str_to_title(`Model Type`))

p <- ggplot(data, aes(x=Epochs, y=`Accuracy Rate`, shape=`Model Type`, color=Balance)) +
  geom_point() +
  theme_bw() +
  ggtitle('Model accuracy by type and data balance') +
  scale_y_continuous(labels = scales::percent) +
  geom_line(aes(y=0.5), linetype='dotted')

ggsave('charts/accuracy-plot.png', width=1600, height=900, units='px')
