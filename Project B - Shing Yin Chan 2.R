# Project B - Shing Yin Chan

cat("\014") # clear console
rm(list=ls()) # clear memory

######################## Question 1 ########################

data = read.table("Project_B_data.csv", header=TRUE, sep=",")
output_file <- "c:/Users/sycha/Downloads/weights2019.txt"

library(lubridate)
library(tidyverse)
library(dplyr)
library(zoo)

date_first <- "1990-01-02"
date_last  <- "2001-12-31"
pid <- 925144829
vid <- 0
ht  <- function(X) X[c(1:4,nrow(X)-1,nrow(X)),] 

# check missing data

str(data)
missing <- complete.cases(data)
sum(!missing)

# rename column names

names(data) <- c("Date","ID","Log Return")
data$Date <- as.Date(data$Date, format="%m/%d/%Y")
str(data)
summary(data$`Log Return`)

U  <- sort(unique(data$ID))
Nu <- length(U)

# create a data.frame to store mean market return

Dateo <- data$Date
Return <- exp(data$`Log Return`) - 1
Return <- data.frame(Dateo, Return)

aggregate_data <- aggregate(Return$Return, by=list(Return$Date), mean)
aggregate_data_sd <- aggregate(Return$Return, by=list(Return$Date), sd)
names(aggregate_data) <- c("Date","Mean Market Return")
names(aggregate_data_sd) <- c("Date","Sd")
Return.market <- aggregate_data$`Mean Market Return`
return.market <- log(1 + Return.market)
Sd.market <- aggregate_data_sd$Sd

plot(aggregate_data$Date, return.market, type="l",main="Daily Market Returns", xlab="Date", ylab="Market Return", col.lab="black");grid()
plot(aggregate_data$Date, return.market, type="p",main="Daily Market Returns", xlab="Date", ylab="Market Return", col.lab="black");grid()

h <- hist(return.market, breaks = "FD", density = 10,
          col = "blue", xlab = "Market daily return", main = "Distribution: Martket daily return", col.lab="black") 
xfit <- seq(min(return.market), max(return.market), length = 40) 
yfit <- dnorm(xfit, mean = mean(return.market), sd = sd(return.market)) 
yfit <- yfit * diff(h$mids[1:2]) * length(return.market) 

lines(xfit, yfit, col = "black", lwd = 2)

# calculate portfolio return

returns <- spread(data,ID,"Log Return")
returns <- as.matrix(returns[,c(-1)]) #remove the Date column 
returns[is.na(returns)] <- 0 #assume 0 return for dates with NA 

Returns <- exp(returns) -1 	#Convert log return into simple returns for cross-sectional use
Returns.mean <- apply(Returns,1,mean) #calculate each row's mean
Returns.excess <- Returns - Returns.mean 

D  <- sort(unique(data$Date))
begind <- which(D==date_first)
endd <- which(D==date_last)
tradedates <- begind:endd

Date <- D[tradedates]
R    <- Returns[tradedates,]
k = 1 #lag 1 = no lag
Ntd = endd

w <- -Returns.excess[tradedates,]

for (i in tradedates) {
  
  for (j in c(seq(1:690))) {
    
    if ((w[i,j]) < 0) {
      
      w[i,j] = 0
      
    }  else {w[i,j] = w[i,j]}
  }
}

wp <- w # store the positive signal in a new matrix

w <- -Returns.excess[tradedates,]

for (i in tradedates) {
  
  for (j in c(seq(1:690))) {
    
    if ((w[i,j]) > 0) {
      
      w[i,j] = 0
      
    }  else {w[i,j] = w[i,j]}
  }
}

wn <- w # store the negative signal in a new matrix

wp <- wp / apply(wp, 1, sum)
wn <- -wn / apply(wn, 1, sum)
w <- wp+wn
sum(w) 

#w <- w * 2 / apply(abs(w),1,sum) # Normalize weights to full exposure
dummym <- w * 0 # create a dummy matrix to store data
dummym[2:Ntd, ] <- w[1:(Ntd-1),] # Exposure that earns return is equal to weight at previous close
Return.portfolio <- apply(dummym * R, 1, sum) # Portfolio simple return
return.portfolio <- log(1 + Return.portfolio) # Portfolio log return

port.long  <- dummym * (dummym>0)
port.short <- dummym * (dummym<0)
return.portfolio.long  <- log(1 + apply(port.long * R,1,sum))
return.portfolio.short <- log(1 + apply(port.short * R,1,sum))

stats.corlongshort <- cor(return.portfolio.long,return.portfolio.short)

plot(Date,return.portfolio,type="l",main="Daily Portfolio Returns",ylab="Portfolio Return");grid()
hist(return.portfolio, breaks="FD", main="Distribution: Portfolio daily return", xlab="Portfolio daily return")

# Correlation of long vs. short sides of the portfolio:
  
plot(return.portfolio.long,return.portfolio.short,pch=16,xlab="Daily Return of Long Positions",ylab="Daily Return of Short Positions",main="Long vs. Short  Strategy Returns");grid()

plot(return.market, return.portfolio.long,pch=16,xlab="Daily Market Returns",ylab="Daily Return of Long Positions",main="Market Returns vs. Long Strategy Returns");grid()
plot(return.market, return.portfolio.short,pch=16,xlab="Daily Market Returns",ylab="Daily Return of Short Positions",main="Market Returns vs. Short  Strategy Returns");grid()
stats.cormarketlong <- cor(return.portfolio.long,return.market)
stats.cormarketshort <- cor(return.portfolio.short,return.market)

# calculation of performance statistics

stats.market <- data.frame(Lag=0,Return=252*mean(return.market), StdDev=sqrt(252)*sd(return.market), "Sharpe"=sqrt(252)*mean(return.market)/sd(return.market))
stats.portfolio <- data.frame(Lag=k,Return=252*mean(return.portfolio), StdDev=sqrt(252)*sd(return.portfolio), "Sharpe"=sqrt(252)*mean(return.portfolio)/sd(return.portfolio))

print(stats.market)
print(stats.portfolio)
print(stats.corlongshort)

market.portfolio.cor = cor(return.portfolio,return.market)
plot(return.market,return.portfolio, pch= 16 ,xlab="Market daily return",ylab="Portfolio daily return",main="Portfolio return vs. Market return");grid()

# moving sd of the portfolio return

rollingsd63 <- data.frame("Date" = Date[63:3028], 
                          "Rolling SD" = rollapply(data = return.portfolio,width=63,FUN=sd))

rollingm63 <- data.frame("Date" = Date[63:3028], 
                          "Rolling Mean" = rollapply(data = return.portfolio,width=63,FUN=mean))

return.portfolio <- data.frame(return.portfolio)
return.portfolio$Date <- D

return.portfolio <- merge(return.portfolio, rollingm63, by="Date", all=T)
return.portfolio <- merge(return.portfolio, rollingsd63, by="Date", all=T)

return.portfolio$rollingsharpe <- return.portfolio$Rolling.Mean/return.portfolio$Rolling.SD
plot(return.portfolio$Date, return.portfolio$Rolling.SD, type="l",main="3-month rolling standard deviation",xlab="Date", ylab="Standard deviation");grid()
plot(return.portfolio$Date, return.portfolio$rollingsharpe, type="l",main="3-month rolling sharpe ratio",xlab="Date", ylab="Sharpe ratio");grid()

# individual stock returns

Returns.stockmean <- apply(Returns,2,mean)
Returns.stocksd <- apply(Returns, 2, sd)

hist(Returns.stockmean, breaks="FD", main="Distribution: Individual stock mean return", xlab="Stock mean return")

# Format weight data for flat file output_file

weights <- gather(cbind(d=Date,as.data.frame(w)),id,w,2:(Nu+1))
weights <- cbind(pid,weights[,1:2],k,w=weights[,3],vid)

write.table(weights,file=output_file,sep="\t",eol="\r\n",quote=FALSE,row.names=FALSE,col.names=TRUE,append=FALSE)

ht(weights)

# calculation of the lag return 2

k = 2

w <- -Returns.excess[tradedates,]
w <- w * 2 / apply(abs(w),1,sum)      # Normalize weights to full exposure
w2 <- w * 0
w2[2:Ntd, ] <- w[1:(Ntd-1),] 
e <- w * 0 # create a matrix to store data
e[3:Ntd, ] <- w[1:(Ntd-2),] # Exposure that earns return is equal to weight at previous close
Return.portfolio2 <- apply(e * R, 1, sum) # Portfolio simple return
return.portfolio2 <- log(1 + Return.portfolio2) # Portfolio log return

plot(Date,return.portfolio2,type="l",main="Daily Strategy Returns, Lag = 2",ylab="Portfolio Return");grid()

stats.portfolio2 <- data.frame(Lag=k,Return=252*mean(return.portfolio2), StdDev=sqrt(252)*sd(return.portfolio2), "Sharpe"=sqrt(252)*mean(return.portfolio2)/sd(return.portfolio2))

print(stats.portfolio2)

market.portfolio.cor2 =cor(return.portfolio2,return.market)
plot(return.market,return.portfolio2, pch= 16 ,xlab="Market daily return",ylab="Portfolio daily return",main="Portfolio return vs. Market return");grid()

weights <- gather(cbind(d=Date,as.data.frame(w2)),id,w2,2:(Nu+1))
weights <- cbind(pid,weights[,1:2],k,w=weights[,3],vid)

write.table(weights,file="c:/Users/sycha/Downloads/weight2019k2.txt",sep="\t",eol="\r\n",quote=FALSE,row.names=FALSE,col.names=TRUE,append=FALSE)

ht(weights)

# calculation of the lag return 3

k = 3

w <- -Returns.excess[tradedates,]
w <- w * 2 / apply(abs(w),1,sum)      # Normalize weights to full exposure
w3 <- w * 0
w3[3:Ntd, ] <- w[1:(Ntd-2),] 
e <- w * 0 # create a matrix to store data
e[(k+1):Ntd, ] <- w[1:(Ntd-k),] # Exposure that earns return is equal to weight at previous close
Return.portfolio3 <- apply(e * R, 1, sum) # Portfolio simple return
return.portfolio3 <- log(1 + Return.portfolio3) # Portfolio log return

plot(Date,return.portfolio3,type="l",main="Daily Strategy Returns, Lag = 3",ylab="Portfolio Return");grid()

stats.portfolio3 <- data.frame(Lag=k,Return=252*mean(return.portfolio3), StdDev=sqrt(252)*sd(return.portfolio3), "Sharpe"=sqrt(252)*mean(return.portfolio3)/sd(return.portfolio3))

print(stats.portfolio3)

market.portfolio.cor3 =cor(return.portfolio3,return.market)
plot(return.market,return.portfolio3, pch= 16 ,xlab="Market daily return",ylab="Portfolio daily return",main="Portfolio return vs. Market return");grid()

weights <- gather(cbind(d=Date,as.data.frame(w3)),id,w3,2:(Nu+1))
weights <- cbind(pid,weights[,1:2],k,w=weights[,3],vid)

write.table(weights,file="c:/Users/sycha/Downloads/weight2019k3.txt",sep="\t",eol="\r\n",quote=FALSE,row.names=FALSE,col.names=TRUE,append=FALSE)

ht(weights)

# calculation of the lag return 4

k = 4

w <- -Returns.excess[tradedates,]
w <- w * 2 / apply(abs(w),1,sum)      # Normalize weights to full exposure
w4 <- w * 0
w4[4:Ntd, ] <- w[1:(Ntd-3),] 
e <- w * 0 # create a matrix to store data
e[(k+1):Ntd, ] <- w[1:(Ntd-k),] # Exposure that earns return is equal to weight at previous close
Return.portfolio4 <- apply(e * R, 1, sum) # Portfolio simple return
return.portfolio4 <- log(1 + Return.portfolio4) # Portfolio log return

plot(Date,return.portfolio4,type="l",main="Daily Strategy Returns, Lag = 4",ylab="Portfolio Return");grid()

stats.portfolio4 <- data.frame(Lag=k,Return=252*mean(return.portfolio4), StdDev=sqrt(252)*sd(return.portfolio4), "Sharpe"=sqrt(252)*mean(return.portfolio4)/sd(return.portfolio4))

print(stats.portfolio4)

market.portfolio.cor4 =cor(return.portfolio4,return.market)
plot(return.market,return.portfolio4, pch= 16 ,xlab="Market daily return",ylab="Portfolio daily return",main="Portfolio return vs. Market return");grid()

weights <- gather(cbind(d=Date,as.data.frame(w4)),id,w4,2:(Nu+1))
weights <- cbind(pid,weights[,1:2],k,w=weights[,3],vid)

write.table(weights,file="c:/Users/sycha/Downloads/weight2019k4.txt",sep="\t",eol="\r\n",quote=FALSE,row.names=FALSE,col.names=TRUE,append=FALSE)

ht(weights)

# calculation of the lag return 5

k = 5

w <- -Returns.excess[tradedates,]
w <- w * 2 / apply(abs(w),1,sum)      # Normalize weights to full exposure
w5 <- w * 0
w5[5:Ntd, ] <- w[1:(Ntd-4),] 
e <- w * 0 # create a matrix to store data
e[(k+1):Ntd, ] <- w[1:(Ntd-k),] # Exposure that earns return is equal to weight at previous close
Return.portfolio5 <- apply(e * R, 1, sum) # Portfolio simple return
return.portfolio5 <- log(1 + Return.portfolio5) # Portfolio log return

plot(Date,return.portfolio5,type="l",main="Daily Strategy Returns, Lag = 5",ylab="Portfolio Return");grid()

stats.portfolio5 <- data.frame(Lag=k,Return=252*mean(return.portfolio5), StdDev=sqrt(252)*sd(return.portfolio5), "Sharpe"=sqrt(252)*mean(return.portfolio5)/sd(return.portfolio5))

print(stats.portfolio5)

market.portfolio.cor5 =cor(return.portfolio5,return.market)
plot(return.market,return.portfolio5, pch= 16 ,xlab="Market daily return",ylab="Portfolio daily return",main="Portfolio return vs. Market return");grid()

weights <- gather(cbind(d=Date,as.data.frame(w5)),id,w5,2:(Nu+1))
weights <- cbind(pid,weights[,1:2],k,w=weights[,3],vid)

write.table(weights,file="c:/Users/sycha/Downloads/weight2019k5.txt",sep="\t",eol="\r\n",quote=FALSE,row.names=FALSE,col.names=TRUE,append=FALSE)

ht(weights)

# my strategy

k = 1

ws <- -Returns.excess[tradedates,]

# filter out those signals not exceeding one standard deviation from the market return  

for (i in tradedates) {
  
  for (j in c(seq(1:690))) {
  
    if (abs(ws[i,j]) > 2*Sd.market[i]) {
      
      ws[i,j] = 0
      
    }  else {ws[i,j] = ws[i,j]}
  }
}

ws <- ws * 2 / apply(abs(ws),1,sum)   # Normalize weights to full exposure
sum(ws)

e <- ws * 0 # create a matrix to store data
e[(k+1):Ntd, ] <- ws[1:(Ntd-k),] # Exposure that earns return is equal to weight at previous close
Return.portfolio.sy <- apply(e * R, 1, sum) # Portfolio simple return
return.portfolio.sy <- log(1 + Return.portfolio.sy) # Portfolio log return

plot(Date,return.portfolio.sy,type="l",main="Daily (Sy Chan) Strategy Returns",ylab="Portfolio Daily Return");grid()

stats.portfolio.sy <- data.frame(Lag=k,Return=252*mean(return.portfolio.sy), StdDev=sqrt(252)*sd(return.portfolio.sy), "Sharpe"=sqrt(252)*mean(return.portfolio.sy)/sd(return.portfolio.sy))

print(stats.market)
print(stats.portfolio.sy)

market.portfolio.cor.sy =cor(return.portfolio.sy,return.market)
plot(return.market,return.portfolio.sy, pch= 16 ,xlab="Market daily return",ylab="Portfolio daily return",main="Portfolio return (Sy Chan Strategy) vs. Market return");grid()
hist(return.portfolio.sy, breaks="FD", main="Distribution: Portfolio (Sy Chan Strategy) daily return", xlab="Portfolio daily return")

port.long.sy  <- e * (e>0)
port.short.sy <- e * (e<0)
return.portfolio.long.sy  <- log(1 + apply(port.long.sy * R,1,sum))
return.portfolio.short.sy <- log(1 + apply(port.short.sy * R,1,sum))

stats.corlongshort.sy <- cor(return.portfolio.long.sy,return.portfolio.short.sy)

plot(return.portfolio.long,return.portfolio.short,pch=16,xlab="Daily Return of Long Positions",ylab="Daily Return of Short Positions",main="Long vs. Short (Sy Chan) Strategy Returns");grid()

weights <- gather(cbind(d=Date,as.data.frame(ws)),id,ws,2:(Nu+1))
weights <- cbind(pid,weights[,1:2],"99",w=weights[,3],vid) # set k = 99 to differentiate this model result than the others

write.table(weights,file="c:/Users/sycha/Downloads/weight2019sy.txt",sep="\t",eol="\r\n",quote=FALSE,row.names=FALSE,col.names=TRUE,append=FALSE)

ht(weights)

