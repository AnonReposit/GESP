# Function that returns an array that is the cummulative sum of a gaussian.
get_sample_sequence_of_gaussians <- function(mu, sigma, length)
{
  res <- c(0)
  for (i in 1:length) {
    res <- c(res, res[i]+rnorm(1, mu, sigma))
  }
  res <- res[2:(length+1)]
  return(res)
}


# Define parameters
mu <- 0
sigma <- 1
length <- 100
num_rows <- 30


# Here we store the proportion of falsely rejected tests for each seed
proportion_falsely_rejected_0_05 <- c()
proportion_falsely_rejected_0_01 <- c()

pb = txtProgressBar(min = 2, max = 3000, initial = 2)
for(seed in 2:3000){
  setTxtProgressBar(pb,seed)

  # Set the seed
  set.seed(seed)

  # Generate data for each variable
  a <- matrix(NA, nrow = num_rows, ncol = length)
  b <- matrix(NA, nrow = num_rows, ncol = length)
  for (i in 1:num_rows) {
    a[i,] <- get_sample_sequence_of_gaussians(mu, sigma, length)
    b[i,] <- get_sample_sequence_of_gaussians(mu, sigma, length)
  }

  # Initialize a vector to store the p-values
  p_values <- numeric(length)

  # Perform Mann-Whitney test column-wise
  for (i in 1:length) {
    result <- wilcox.test(a[, i], b[, i], alternative = "two.sided")
    p_values[i] <- result$p.value
  }

  # Compute percentage of points in which test was falsely rejected
  alpha <- 0.05
  proportion_falsely_reject <- mean(p_values < alpha)*100
  proportion_falsely_rejected_0_05 <- c(proportion_falsely_rejected_0_05, proportion_falsely_reject)

  alpha <- 0.01
  proportion_falsely_reject <- mean(p_values < alpha)*100
  proportion_falsely_rejected_0_01 <- c(proportion_falsely_rejected_0_01, proportion_falsely_reject)
}
close(pb)

cat(paste("On average, the percentage of falsely rejected pointwise tests was ", mean(proportion_falsely_rejected_0_05), "% when alpha=0.05"))
cat(paste("On average, the percentage of falsely rejected pointwise tests was ", mean(proportion_falsely_rejected_0_01), "% when alpha=0.01"))

# >> On average, the percentage of falsely rejected pointwise tests was 5.05435145048349% when alpha=0.05
# >> On average, the percentage of falsely rejected pointwise tests was 1.05835278426142% when alpha=0.01
