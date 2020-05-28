library("xts")
library("openair")
library("tidyr")
library("tidyverse")
library("forecast")
library("reshape")
library("seastests")
library("IDPmisc")
library("quantmod")
library("PerformanceAnalytics")
library("bsts")
library("dplyr")
library("roll")
library("corrplot")
library("dtwclust")
library("TSclust")
library("lattice")



#Getting data
options(scipen = 999)
setwd("C:/Users/sursa/Documents/analytics_fin/")
 
#bring in data from Factors
Factors = read.csv("F-F_Research_Data_Factors.csv") 
Factors$Date <- paste0("01", Factors$Date)
Factors$Date <- as.Date(Factors$Date,format="%d%Y%m",length=4,by="months")
Factors <- na.omit(Factors)
Factors$Date <- LastDayInMonth(Factors$Date)
Factors <- as.xts(Factors[,2:5]/100,order.by=Factors$Date)
Factors_roll <- roll_prod(1+Factors,width=3)-1


#bringing individual stock data
Data = read.csv("Project_Data.csv")
Data$dates <- as.Date.character(strptime(Data$dates, "%m/%d/%y"))
Data$dates <- LastDayInMonth(Data$dates)
Data <- Data[Data$TICKER != "VLO",]
Data <- Data[Data$PE > -1000000,]
Data <- Data[Data$TICKER != "",]
Data1 <- selectByDate(Data, month =c(4,7,10,1))
Data1$EPS <- Data1$Price/Data1$PE

#Doing earnings seasonality study
#Data2 <- Data1 %>% select(2,6,9)
Data2 <- subset(Data1,select=c(2,6,9))
Data2 <- cast(Data2,dates~TICKER,mean,value="EPS")

b <- ncol(Data2)
t <- 20
seasonality_scores <- vector()

for (i in 2:b){
  col_data <- if(length(NaRV.omit(Data2[,i]))<t) 1 else wo(NaRV.omit(Data2[,i]),freq = 4)
  seasonality_scores <- cbind(seasonality_scores,if(length(NaRV.omit(Data2[,i]))<t) 1 else as.numeric(col_data$Pval[3]))
}


seasonality_results <- cbind.data.frame(colnames(Data2),c(1,seasonality_scores))
colnames(seasonality_results) <- c("TICKER","SEA_SCORE")
#Picking seasonality scores with p values less than 0.01
sample_chosen <- seasonality_results[seasonality_results$SEA_SCORE < 0.002,]

#Now finding earnings that besides being seasonal, move in similar patterns
Data2 <- subset(Data1,select=c(2,6,8,9))
Data2 <- Data2 %>% filter(TICKER %in% sample_chosen[,1])
Data3_EPS <- cast(Data2,dates~TICKER,mean,value="EPS")
Data3_EPS <- Data3_EPS[ , colSums(is.na(Data3_EPS)) == 0]
Data4_EPS <- as.xts(Data3_EPS[,2:ncol(Data3_EPS)],order.by=Data3_EPS$dates)
colnames(Data4_EPS) <- colnames(Data3_EPS[2:ncol(Data3_EPS)])
Data4_EPS <- CalculateReturns(Data4_EPS, method = "arithmetic")
Data4_EPS <- Data4_EPS[2:nrow(Data4_EPS),]

Diss_matrix <- as.matrix(diss(t(as.matrix(Data4_EPS)), "COR"))
Diss_list <- as.data.frame(rowMeans(Diss_matrix))
colnames(Diss_list) <- c("Dissimilarty")

#selecting a cluster of companies that have similar patterns
sample_chosen_2 <- subset(Diss_list,Dissimilarty<quantile(Diss_matrix,0.25))







#Now building portfolio index

Data5 <- subset(Data,select=c(2,6,8))
Data5 <- Data5 %>% filter(TICKER %in% rownames(sample_chosen_2))
Data5 <- cast(Data5,dates~TICKER,mean,value="Price")
Data6 <- as.xts(Data5[,2:ncol(Data5)],order.by=Data5$dates)
Data6 <- CalculateReturns(Data6, method = "arithmetic")
colnames(Data6) <- colnames(Data5[2:ncol(Data5)])
Data6 <- Data6[2:nrow(Data6),]
Data6 <- as.xts(rowMeans(Data6[,]),order.by=Data5$dates[2:length(Data5$dates)])
colnames(Data6) <- c("Seasonality_index")
Data6 <- merge(Data6,Factors)
Data6 <- na.omit(Data6)

model_seasonal <- lm(Seasonality_index-RF~.-RF,data=Data6)
summary(model_seasonal)
residuals <- as.xts((model_seasonal$residuals+model_seasonal$coefficient[1]),order.by=Data5$dates[2:length(Data5$dates)])

Data6 <- merge(Data6,residuals)



Data7 <-  data.frame(date=index(Data6), coredata(Data6))
#Data7$Month <- months.Date(Data7$date)
#Data7$Year <- months.Date(Data7$date)
Data7$Month <- month <- as.numeric(format(Data7$date,'%m'))
Data7$Year <- year <- as.numeric(format(Data7$date,'%Y'))
xyplot(residuals~ Month , groups=Year, data = Data7,auto.key = TRUE,main="Monthly Residuals",type="p",ylim=c(-0.05,0.05), cex = 0.8)
abline(h=0,col="blue")


Data8 <- cast(Data7, Month~Year,mean,value="residuals")

Data7$Weight[Data7$Month == 1] <- 0
Data7$Weight[Data7$Month == 2 ] <- 1
Data7$Weight[Data7$Month == 3 ] <- 0
Data7$Weight[Data7$Month == 4 ] <- 1
Data7$Weight[Data7$Month == 5] <- 0
Data7$Weight[Data7$Month == 6] <- 0
Data7$Weight[Data7$Month == 7] <- 0
Data7$Weight[Data7$Month == 8] <- 0
Data7$Weight[Data7$Month == 9] <- 0
Data7$Weight[Data7$Month == 10] <- 1
Data7$Weight[Data7$Month == 11] <- 1
Data7$Weight[Data7$Month == 12] <- 0

Data7$Weight2[Data7$Month == 1] <- 1
Data7$Weight2[Data7$Month == 2 ] <- 0
Data7$Weight2[Data7$Month == 3 ] <- 1
Data7$Weight2[Data7$Month == 4 ] <- 0
Data7$Weight2[Data7$Month == 5] <- 1
Data7$Weight2[Data7$Month == 6] <- 1
Data7$Weight2[Data7$Month == 7] <- 1
Data7$Weight2[Data7$Month == 8] <- 1
Data7$Weight2[Data7$Month == 9] <- 1
Data7$Weight2[Data7$Month == 10] <- 0
Data7$Weight2[Data7$Month == 11] <- 0 
Data7$Weight2[Data7$Month == 12] <- 1

Data7$Weight3[Data7$Month == 1] <- -1
Data7$Weight3[Data7$Month == 2 ] <- 1
Data7$Weight3[Data7$Month == 3 ] <- 0
Data7$Weight3[Data7$Month == 4 ] <- 1
Data7$Weight3[Data7$Month == 5] <- 0
Data7$Weight3[Data7$Month == 6] <- -1
Data7$Weight3[Data7$Month == 7] <- -1
Data7$Weight3[Data7$Month == 8] <- -1
Data7$Weight3[Data7$Month == 9] <- -1
Data7$Weight3[Data7$Month == 10] <- 1
Data7$Weight3[Data7$Month == 11] <- 1
Data7$Weight3[Data7$Month == 12] <- 0

Data8 <- aggregate(residuals~ Month, Data7, mean)
Data8$Roll_return <- roll_prod(1+Data8$residuals,width=3)-1


plot(Data8$Month,Data8$residuals,type="l",xlab="Month of Year",ylab="Monthly Residuals from Linear Model",main="",ylim=c(-0.025,0.025))
abline(h=0, col="blue")
abline(h=-0.0050, col="red")
title("Seasonal Excess Return")

Data9 <- merge(Data6$Seasonality_index,Data7$Weight)
Data9 <- merge(Data9,Data7$Weight2)
Data9 <- merge(Data9,Data7$Weight3)
Data9$Weighted_Returns <- Data9$Seasonality_index*Data9$Data7.Weight
Data9$Weighted_Returns2 <- Data9$Seasonality_index*Data9$Data7.Weight2
Data9$Weighted_Returns3 <- Data9$Seasonality_index*Data9$Data7.Weight3

Event_dates <- Return.portfolio(Data9[,5], weights = 1)
Non_event_dates <- Return.portfolio(Data9[,6], weights = 1)

Event_dates <- Event_dates[Event_dates$portfolio.returns != 0,]
Non_event_dates <- Non_event_dates[Non_event_dates$portfolio.returns != 0,]
plot(density(na.omit(Event_dates)),main="")
lines(density(na.omit(Non_event_dates)),col="blue")
title("Kernel Density Plot")
legend("topleft",legend = c("Event Dates", "Non-Event Dates"),col=c("Black","Blue"),lty=1, cex=0.9)


Dynamic_port <- Return.portfolio(Data9[,7], weights = 1)
Long_port <- Return.portfolio(Data9[,1], weights = 1)
Market_port <- Return.portfolio((Data6[,2]+Data6[,5]), weights = 1)


Port_analysis <- merge(Dynamic_port,Long_port,Market_port)
colnames(Port_analysis) <- c("Dynamic Strategy on Seasonal Index", "Long Seasonal Index","Market")


charts.PerformanceSummary(Port_analysis, Rf = Data6[,5], main = "Investment Performance", geometric = TRUE)
table.AnnualizedReturns(Port_analysis,scale=12,Rf = Data6[,5])*100

chart.RollingPerformance(Port_analysis, Rf = Data6[,5], width= 12,main = "Investment Performance", geometric = TRUE)

table.CalendarReturns(Port_analysis)
