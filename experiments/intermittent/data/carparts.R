# Car parts (Hyndman et al. 2008): monthly sales of 2,674 car parts, Jan 1998 - Mar 2002 (51 months).
if (!requireNamespace("expsmooth", quietly = TRUE)) install.packages("expsmooth", repos = "https://cloud.r-project.org")
data("carparts", package = "expsmooth")
x <- as.data.frame(carparts)
cat(dim(x), sum(is.na(x)), "\n")
write.csv(x, "carparts.csv", row.names = FALSE)
