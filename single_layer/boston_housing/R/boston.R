rm(list=ls())
boston.data = read.csv("Boston.csv")
samp<-floor(0.8 * nrow(boston.data))
set.seed(12399645)
index.train <- sample(seq_len(nrow(boston.data)), size = samp)
boston.train <- boston.data[index.train, ]
boston.test <- boston.data[-index.train, ]
names(boston.train)
str(boston.data)

#1- GLM
#Step-wise Model building and assessment- in sample
nullmodel<- lm(medv~1, data=boston.train)
fullmodel<- lm(medv~., data=boston.train)
#Step-wise
model.step.s<- step(nullmodel, scope=list(lower=nullmodel, upper=fullmodel), direction='both')
(summary(model.step.s)$sigma)^2
summary(model.step.s)$r.squared
summary(model.step.s)$adj.r.squared
AIC(model.step.s)
BIC(model.step.s)
pred.mod.step<- predict(model.step.s, newdata = boston.train)
mean((boston.train$medv-pred.mod.step)^2)
#Out-of-sample prediction error 
test.pred.mod.step<-predict(model.step.s, newdata=boston.test) 
mean((boston.test$medv-test.pred.mod.step)^2)

#2 CART
# library(rpart)
# library(ROCR)
# boston.rpart <- rpart(formula = medv ~ ., data = boston.train)
# boston.rpart
# par(mar=c(1,1,1,1))
# plot(boston.rpart)
# plotcp(boston.rpart)
# text(boston.rpart)
# boston.train.pred.tree = predict(boston.rpart) #in-sample
# boston.test.pred.tree = predict(boston.rpart, boston.test) #outofsample
# mean((boston.train.pred.tree - boston.train$medv)^2)
# mean((boston.test.pred.tree - boston.test$medv)^2)


#3 GAM
library(mgcv)
boston.gam <- gam(medv ~ s(crim)+s(zn)+s(indus)+chas+s(nox)+s(rm)+s(age)
                  +s(dis)+rad+s(tax)+s(ptratio)+s(black)+s(lstat), data=boston.train)
summary(boston.gam)
par(mar=c(0.6,0.6,0.6,0.6))
plot(boston.gam, pages=1)
vis.gam(boston.gam, view = c("lstat", "black")) 
plot(fitted(boston.gam), residuals(boston.gam), 
     xlab = 'fitted', ylab = 'residuals', main = 'Residuals by Fitted from GAM') 

boston.gam.mse.train <- boston.gam$deviance/boston.gam$df.residual
AIC(boston.gam)
BIC(boston.gam)

#insample
boston.gam.predict.train<- predict(boston.gam, boston.train) 
boston.gam.mse.train<- mean((boston.gam.predict.train- boston.train[, "medv"])^2)

#outofsample
boston.gam.predict.test <- predict(boston.gam, boston.test) 
boston.gam.mse.test<- mean((boston.gam.predict.test - boston.test[, "medv"])^2)

#Neural Nets
#install.packages("neuralnet")
library(neuralnet)
f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))

nn1 <- neuralnet(f,data=boston.train,hidden=c(4,2),linear.output=T)
plot(nn1)
pr.nn_train <- compute(nn1, boston.train[,1:13])
sum((boston.train$medv - pr.nn_train$net.result)^2)/nrow(boston.train)
#outofsample
pr.nn_test <- compute(nn1, boston.test[,1:13])
sum((boston.test$medv - pr.nn_test$net.result)^2)/nrow(boston.test)

#NNet2
# Train networks with sizes of hidden units ranging from 0 to 20, then calculate the average SSE for training and testing datasets 
library(nnet)
library(ggplot2)
averaged.SSE.train <- vector() ; 
averaged.SSE.test <- vector() 
for (n in 1:20){ 
  # size of hidden units ranging from 0 to 20   
  train.predict <- 0; test.predict <- 0   
  for(i in 1:10){ 
    # for each size, train 10 networks with different random starting points, averaging the ten results to make the prediction more accurate     
    set.seed(i)     
    Boston.nnet <- nnet(medv ~ ., size = n, data = boston.train, maxit=1000, decay=0, linout = TRUE)      
    train.predict<-train.predict+predict(Boston.nnet, boston.train)     
    test.predict<-test.predict+predict(Boston.nnet, boston.test) 
    }   
  # average outcomes of 10 networks 
  train.predict.avg <-train.predict/10 ;test.predict.avg <-test.predict/10 
  train.predict.origin <- train.predict.avg ;   test.predict.origin <- test.predict.avg 
  averaged.SSE.train[n] <- sum( (train.predict.origin -  boston.train$medv)^2 ) / nrow(boston.train) 
  averaged.SSE.test[n] <- sum( (test.predict.origin -  boston.test$medv)^2 ) / nrow(boston.test) 
}

legend <- c(rep("train", length(averaged.SSE.train)), rep("test", length(averaged.SSE.test)) ) 
result <- data.frame(size=rep(1:20, 2), average.SSE = c(averaged.SSE.train, averaged.SSE.test), legend = legend ) 
ggplot(data=result, aes(x=size, y=average.SSE, group = legend, colour = legend)) +geom_line(aes(x=as.integer(size))) +geom_point(size=1) +ggtitle("Plot of Averaged SSE vs. Hidden Layers") +xlab("Number of Hidden Units") 

#Checking best decay
k <- 1; Decay <- seq(0, 0.01,length.out=11) 
averaged.SSE.test <- vector() 

for (n in 7:10){
  ## number of hidden unites ranges from 7 to 10  
  for (d in Decay){ # Weight Decay ranges from 0 to 0.01   
    test.predict <- 0   
    for(i in 1:10){ # for each Decay, train 10 networks with different random starting points     
      set.seed(i) 
      Boston.nnet <- nnet(medv ~ ., size = n, data = boston.train, maxit=1000, decay=d, linout = TRUE) 
      test.predict<-test.predict+predict(Boston.nnet, boston.test)   
    }
    # average outcomes of 10 networks
    test.predict.avg <-test.predict/10 
    test.predict.origin <- test.predict.avg 
    #### need to calculate test.predict.origin based on different case 
    averaged.SSE.test[k] <- sum( (test.predict.origin -  boston.test$medv)^2 ) / nrow(boston.test)   
    k <- k + 1  
    } 
} 
size <- c(rep(7,11),rep(8,11),rep(9,11),rep(10,11)) 
result <- data.frame(decay=rep(Decay, 4), average.SSE = averaged.SSE.test, size=size) 

# plot 
ggplot(data=result, aes(x=decay, y=average.SSE, group = size, colour = as.character(size) ) ) + geom_line(aes(x=decay)) + geom_point(size=1) +ggtitle("Average SSE vs. Weight Decays ") + xlab("Weight Decays") 


#### Final model:0.008 decay and 8 hidden units
n<-8; boston.predict<-0.008
for(i in 1:10){ 
  set.seed(i)   
  Boston.nnet <- nnet(medv ~ ., size = n, data = boston.data, maxit=1000, decay=0.008, linout = TRUE)   
  boston.predict<-boston.predict+predict(Boston.nnet, boston.data) 
}
predict.avg<-boston.predict/10
predict.origin <- predict.avg 
averaged.SSE <- sum( (predict.origin -  boston.data$medv)^2 ) / nrow(boston.data)

#insample
n<-8; boston.predict<-0.008
Boston.nnet.tr <- nnet(medv ~ ., size = n, data = boston.train, maxit=5000, decay=0.008, linout = TRUE)   
boston.predict.tr<-predict(Boston.nnet.tr, boston.train) 
averaged.SSE.tr <- sum((boston.predict.tr -  boston.train$medv)^2 ) / nrow(boston.train)

#outsample
n<-8; boston.predict<-0.008
Boston.nnet.test <- nnet(medv ~ ., size = n, data = boston.test, maxit=5000, decay=0.008, linout = TRUE)   
boston.predict.test<-predict(Boston.nnet.test, boston.test) 
averaged.SSE.test <- sum((boston.predict.test -  boston.test$medv)^2 ) / nrow(boston.test)

#nnet-single layer
library(neuralnet); 
train <- as.data.frame(boston.data); 
n <- names(train) 
f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))
average.SSE.test <- matrix(NA, nrow=6, ncol=6)
for (i in 3:6){   ## first hidden layer   
  for (j in 2:i){  ## second hidden layer     
    predict.test <- 0       
    for(k in 1:10){       
      set.seed(k)       
      nn <- neuralnet(f,data=train,hidden=c(i,j),linear.output=T)       
      predict.test <- predict.test + compute(nn,test[,1:13])$net.result
      }     
    predict.avg.test<-predict.test/10     
    predict.origin.test <- predict.avg.test * 1
    average.SSE.test[i,j] <- sum( (predict.origin.test -  test.origin[,14])^2 ) / nrow(test.origin)
    }
  } 
nn <- neuralnet(f,data=train,hidden=c(8,1),linear.output=T) 
plot(nn)