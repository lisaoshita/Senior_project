# ========================================================================================
# Normalized Cumulative Discounted Gain at k = 5 (metric used in competition)
# ========================================================================================

ndcg5 <- function(preds, dtrain) {
  
  labels <- getinfo(dtrain,"label")
  num.class = length(unique(labels))
  pred <- matrix(preds, nrow = num.class)
  top <- t(apply(pred, 2, function(y) order(y)[num.class:(num.class-4)]-1))
  
  x <- ifelse(top == labels,1,0)
  dcg <- function(y) sum((2 ^ y - 1) / log(2 : (length(y) + 1), base = 2))
  ndcg <- mean(apply(x,1,dcg))
  return(list(metric = "ndcg5", value = ndcg))
  
}

dcg_at_k <- function (r, k=min(5, length(r)) ) {
  #only coded alternative formulation of DCG (used by kaggle)
  r <- as.vector(r)[1:k]
  sum(( 2^r - 1 )/ log2( 2:(length(r)+1)) )
} 

ndcg_at_k <- function(r, k=min(5, length(r)) ) {
  r <- as.vector(r)[1:k]
  if (sum(r) <= 0) return (0)     # no hits (dcg_max = 0)
  dcg_max = dcg_at_k(sort(r, decreasing=TRUE)[1:k], k)
  return ( dcg_at_k(r, k) / dcg_max )
}


score_predictions <- function(preds, truth) {
  # preds: matrix or data.frame
  # one row for each observation, one column for each prediction.
  # Columns are sorted from left to right descending in order of likelihood.
  # truth: vector
  # one row for each observation.
  preds <- as.matrix(preds)
  truth <- as.vector(truth)
  
  stopifnot( length(truth) == nrow(preds))
  r <- apply( cbind( truth, preds), 1
              , function(x) ifelse( x == x[1], 1, 0))[ -1, ]
  if ( ncol(preds) == 1) r <-  rbind( r, r)  #workaround for 1d matrices
  as.vector( apply(r, 2, ndcg_at_k) )
}

