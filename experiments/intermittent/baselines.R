# fable.intermittent's methods on Car Parts (CPU). Two protocols:
#  "w1", "w2": the benchmark's two expanding windows (h = 12) on the series complete over 51 months;
#  "monash":   the Monash archive's zero-filled version, all 2,674 series, last 12 months as test.
# Writes the forecast mean and the quantiles at 0.5, 0.75, 0.835, 0.975, 0.995 per series and step.
args <- commandArgs(trailingOnly = TRUE)
methods <- if (length(args)) args else c("TWEES", "NEGBINES", "PARAMSD")
suppressMessages({library(fable.intermittent); library(fable); library(tsibble); library(dplyr); library(distributional)})
h <- 12; taus <- c(0.5, 0.75, 0.835, 0.975, 0.995)
x <- read.csv("data/carparts.csv", check.names = FALSE)
complete <- x[, colSums(is.na(x)) == 0]
filled <- x; filled[is.na(filled)] <- 0
spec <- list(TWEES = function() TWEES(value), NEGBINES = function() NEGBINES(value),
             HSPES = function() HSPES(value), PARAMSD = function() PARAMSD(value))
run <- function(m, data, T, label) {
  d <- do.call(rbind, lapply(names(data), function(k)
    data.frame(id = k, t = yearmonth("1998 Jan") + 0:(T - 1), value = data[[k]][1:T])))
  fc <- as_tsibble(d, key = id, index = t) |> model(f = spec[[m]]()) |> forecast(h = h)
  q <- sapply(taus, function(a) quantile(fc$value, a))
  res <- data.frame(id = fc$id, window = label, step = rep(1:h, length.out = nrow(fc)), mean = mean(fc$value), q)
  names(res)[5:9] <- paste0("q", taus); res
}
for (m in methods) {
  t0 <- Sys.time()
  res <- rbind(run(m, complete, nrow(x) - h, "w1"), run(m, complete, nrow(x) - 2 * h, "w2"),
               run(m, filled, nrow(x) - h, "monash"))
  write.csv(res, paste0("results/carparts_", m, ".csv"), row.names = FALSE)
  cat(m, "done in", format(Sys.time() - t0), "\n")
}
