for (f in list.files("fable.intermittent/data", full.names = TRUE)) {
  e <- new.env(); load(f, envir = e)
  for (nm in ls(e)) {
    x <- as.data.frame(get(nm, envir = e))
    cat(nm, nrow(x), paste(names(x), collapse = ","), paste(sapply(x, function(c) class(c)[1]), collapse = ","), "\n")
    for (j in seq_along(x)) if (!is.numeric(x[[j]]) && !is.character(x[[j]])) x[[j]] <- as.character(x[[j]])
    write.csv(x, paste0(nm, ".csv"), row.names = FALSE)
  }
}
