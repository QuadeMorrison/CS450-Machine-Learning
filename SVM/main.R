library (e1071)

formula <- function(symbol) {
  as.formula(paste(symbol, "~ ."))
}

get.data <- function(fileName, split.percent) {
  data <- read.csv(fileName, head=TRUE, sep=",")
  rows <- 1:nrow(data)
  test.rows <- sample(rows, trunc(length(rows) * split.percent))
  return(list(data[test.rows,], data[-test.rows,]))
}

init.svm <- function(train, test, header.index, g, c, k) {
  names <- names(train)
  header <- names[[header.index]]
  model <- svm(formula(header), data = train, kernel= k, gamma = g, cost = c, type="C")
  prediction <- predict(model, test[-header.index])
  agreement <- prediction == test[[header]]
  print(paste("Kernel:", k, " ", "Gamma:", g, " ", "Cost:", c))
  accuracy = prop.table(table(agreement))
  print(accuracy)
}

run.svm <- function(fileName, split.percent, header.index, g, c, k) {
  data <- get.data(fileName, split.percent)
  init.svm(data[[1]], data[[2]], header.index, g, c, k)
}

svm.10.times <- function(fileName, header.index) {
  run.svm(fileName, 0.7, header.index, 0.01, 10, "radial")
  run.svm(fileName, 0.7, header.index, 0.5, 10, "radial")
  run.svm(fileName, 0.7, header.index, 1, 10, "radial")
  run.svm(fileName, 0.7, header.index, 0.01, 100, "radial")
  run.svm(fileName, 0.7, header.index, 0.01, 10, "radial")
  run.svm(fileName, 0.7, header.index, 0.01, 50, "radial")
  run.svm(fileName, 0.7, header.index, 0.01, 1000, "radial")
  run.svm(fileName, 0.7, header.index, 0.01, 100, "linear")
  run.svm(fileName, 0.7, header.index, 0.01, 100, "polynomial")
  run.svm(fileName, 0.7, header.index, 0.01, 100, "sigmoid")
}

svm.10.times("vowel.csv", 13)
svm.10.times("letters.csv", 1)
svm.10.times("abalone.csv", 9)
