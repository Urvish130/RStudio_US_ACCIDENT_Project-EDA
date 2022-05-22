getwd()                                        #getting current directory
setwd("F://Statistics for BA//Final Project")   #setting the directory
df = read.csv("US_Accidents.csv") 
table(is.na(df))
df = na.omit(df)
table(is.na(df))
df = unique(df)
library(psych) 
install.packages("ISLR")
library(ISLR)
str(df)

summary(df)
describe(df)
install.packages("dlookr")
library(dlookr)
normality(df)    # hypothesis test of normality

jpeg("my_plot.png")          # Paper size

plot_normality(df)
dev.off()
correlate(df)
plot_correlate(df)

df$Amenity = as.integer(df$Amenity == "TRUE")
df$Bump  = as.integer(df$Bump  == "TRUE")
df$Crossing = as.integer(df$Crossing == "TRUE")
df$Give_Way  = as.integer(df$Give_Way  == "TRUE")
df$Junction  = as.integer(df$Junction  == "TRUE")
df$No_Exit  = as.integer(df$No_Exit  == "TRUE")
df$Railway  = as.integer(df$Railway  == "TRUE")
df$Roundabout  = as.integer(df$Roundabout  == "TRUE")
df$Station    = as.integer(df$Station    == "TRUE")
df$Stop   = as.integer(df$Stop   == "TRUE")
df$Traffic_Calming  = as.integer(df$Traffic_Calming  == "TRUE")
df$Traffic_Signal   = as.integer(df$Traffic_Signal   == "TRUE")
df$Turning_Loop   = as.integer(df$Turning_Loop   == "TRUE")
df$Sunrise_Sunset   = as.integer(df$Sunrise_Sunset   == "Day")
df$Civil_Twilight    = as.integer(df$Civil_Twilight   == "Day")
df$Nautical_Twilight   = as.integer(df$Nautical_Twilight   == "Day")
df$Astronomical_Twilight   = as.integer(df$Astronomical_Twilight   == "Day")

#ML Model Severity vs all
library(MASS)
library(car)
library(dplyr)
df = df[vapply(df, function(x) length(unique(x)) > 1, logical(1L))] #Removes columns with same value.
df$Side_isRight = as.integer(df$Side == 'R')
#Multiple linear regression
model=lm(Severity~Distance.mi.+Temperature.F.+	Humidity...	+Pressure.in.	+Visibility.mi.+Wind_Speed.mph.+	Precipitation.in.,data=df)
stepAIC(model,direction = "both",trace = FALSE)
summary(model)

install.packages("leaps")
library(leaps)
regsubsets.out <-regsubsets(Severity~.,data =df,nbest = 1,      # 1 best model for each number of predictors
                            nvmax = NULL,    # NULL for no limit on number of variables
                            force.in = NULL, force.out = NULL,
                            method = "exhaustive")
summary_best_subset <- summary(regsubsets.out)
as.data.frame(summary_best_subset$outmat)


df_ML = lm(log(Severity) ~ Temperature.F. + Wind_Chill.F. + log(Humidity...) + Pressure.in. + Visibility.mi. + Wind_Speed.mph. 
           + Precipitation.in., data = df_for_ml )
x=stepAIC(df_ML,direction = "both",trace = FALSE)
summary(x)
df_best_ml_model = lm(formula = log(Severity) ~ Temperature.F. + Wind_Chill.F. + 
                        log(Humidity...) + Pressure.in. + Wind_Speed.mph., data = df)
summary(df_best_ml_model)
par(mfrow = c(2, 2))
plot(df_best_ml_model)
vif(df_best_ml_model)                 #for checking multicollinearity 
anova(df_best_ml_model)

library(leaps)
volumes = c(1:7)
for (i in volumes) {
  models = regsubsets(log(Severity)~Temperature.F. + Wind_Chill.F. + log(Humidity...) + Pressure.in. + Visibility.mi. + Wind_Speed.mph. 
                      + Precipitation.in., data = df, nvmax = i,
                      method = "seqrep")
  print(summary(models))
}



#Fit the model Logistic one
model = glm(as.factor(Severity) ~ Temperature.F.+ Sunrise_Sunset + Civil_Twilight  + Nautical_Twilight + Astronomical_Twilight +
              Side_isRight + Humidity... + Pressure.in. + Visibility.mi.+ Wind_Speed.mph.+ Precipitation.in., data = df, family = binomial)
              
stepAIC(model,direction = "both",trace = FALSE)

#best model 
best_logistic_model = glm(formula = as.factor(Severity) ~ Temperature.F. + Civil_Twilight + 
      Nautical_Twilight + Astronomical_Twilight + Humidity... + 
      Pressure.in. + Wind_Speed.mph. + Precipitation.in., family = binomial, 
    data = df)
summary(best_logistic_model)
par(mfrow=c(2,2))
plot(best_logistic_model)
car::vif(best_logistic_model)
anova(best_logistic_model,test = "Chisq")
library(car)
durbinWatsonTest(best_logistic_model)

library(tidyverse)
library(broom)
theme_set(theme_classic())

probabilities = predict(best_logistic_model, type = "response")
predicted.classes = ifelse(probabilities > 0.5, "pos", "neg")
head(predicted.classes)
par(mfrow=c(2,2))
plot(best_logistic_model, which = 4, id.n = 3)

#
install.packages("broom")
library(broom)
library(ggplot2) 
model.data = augment(best_logistic_model) %>% 
  mutate(index = 1:n()) 
model.data %>% top_n(3, .cooksd)
ggplot(model.data, aes(index, .std.resid)) + 
  geom_point(aes(color = `as.factor(Severity)`), alpha = .5) +
  theme_bw()

car::vif(best_logistic_model)

df$Severity1   = as.integer(df$Severity >2)
model_isSevere = glm(as.factor(Severity1) ~ Temperature.F.+ Sunrise_Sunset + Civil_Twilight  + Nautical_Twilight + Astronomical_Twilight +
                       Side_isRight + Humidity... + Pressure.in. + Visibility.mi.+ Wind_Speed.mph.+ Precipitation.in., data = df, family = binomial)

stepAIC(model_isSevere,direction = "both",trace = FALSE)

df2=df

library(tidyverse)
library(caret)
set.seed(12367)
training.samples = as.factor(df2$Severity1) %>% 
  createDataPartition(p = 0.8, list = FALSE)
train.data  = df2[training.samples, ]
test.data = df2[-training.samples, ]

model_for_prediction = glm(formula = as.factor(Severity1) ~ Temperature.F. + Sunrise_Sunset + 
                             Civil_Twilight + Astronomical_Twilight + Side_isRight + Humidity... + 
                             Pressure.in. + Visibility.mi. + Wind_Speed.mph. + Precipitation.in., 
                           family = binomial, data = df2)
summary(model_for_prediction)

probabilities = model_for_prediction %>% predict(test.data, type = "response")
predicted.classes = ifelse(probabilities > 0.5, "1", "0")
# Model accuracy
mean(predicted.classes==as.factor(test.data$Severity1))

my_func <- function(df, group){
  df %>%
    group_by(!!group) %>%
    summarise(my_count = n()) %>%
    arrange(desc(my_count))
}
my_group = quo(State)
highest_accident_state=my_func(df, my_group)

ggplot(data=highest_accident_state, aes(x=State, y=my_count)) +
  geom_bar(stat="identity", fill="steelblue", width = 1)+
  geom_text(aes(label=my_count), vjust=1.6, color="white", size=1.5)+
  theme_minimal()

install.packages("lubridate")
library(lubridate)
library(readr)
accidents_new = df %>%
  mutate(startHr=hour(Start_Time))
head(accidents_new)
accidents_count = accidents_new %>%
  count(startHr)
head(accidents_count)

ggplot(accidents_count, aes(startHr, n)) + geom_point() +geom_path()

#before Cleaning
df_actual = read.csv("US_Accidents.csv")
accidents_new_actual = df_actual %>%
  mutate(startHr=hour(Start_Time))
head(accidents_new_actual)
accidents_count_actual = accidents_new_actual %>%
  count(startHr)
head(accidents_count_actual)

ggplot(accidents_count_actual, aes(startHr, n)) + geom_point() +geom_path()
cor(df[,sapply(df,is.numeric)],use="complete.obs",method="pearson")

#Count of severity 
Count_severity = my_func(df, quo(Severity))
ggplot(data=Count_severity, aes(x=Severity, y=my_count)) +
  geom_bar(stat="identity", fill="steelblue", width = 1)+
  geom_text(aes(label=my_count), vjust=1.6, color="white", size=3.5)+
  theme_minimal()

bp= ggplot(Count_severity, aes(x="", y=my_count,fill=Severity))+
  geom_bar(width = 1, stat = "identity")
pie = bp + coord_polar("y", start=0)
pie
library(ggpubr)
ggboxplot(df, x = "Severity", y = "Wind_Chill.F.", width = 0.8)

Count_severity_city = my_func(df, quo(City))
Count_severity_city = head(Count_severity_city, 20)
ggplot(data=Count_severity_city, aes(x=City, y=my_count)) +
  geom_bar(stat="identity", fill="steelblue", width = 1)+
  geom_text(aes(label=my_count), vjust=2.6, color="white", size=2.5)+
  theme_minimal()

#Time Series Fore Casting
df_for_TSF = read.csv("US_Accidents.csv")
head(df_for_TSF)
df_for_TSF$MY = format(as.Date(df_for_TSF$Start_Time), "%m-%y")
count_accidents_MY = my_func(df_for_TSF,quo(MY))
install.packages("zoo")
library(zoo)
count_accidents_MY <- count_accidents_MY[order(as.yearmon(count_accidents_MY$MY, "%m-%Y")),]
head(count_accidents_MY,20)

summary(count_accidents_MY)
accidents.ts = ts(count_accidents_MY$my_count, start = 2016, end = 2020,frequency = 12)
plot(accidents.ts, xlab = "Time", ylab = "Number of accidents", )
plot(decompose(accidents.ts))
acf(accidents.ts)
pacf(accidents.ts)

install.packages("tseries")
library(tseries)
adf.test(accidents.ts)

seasonplot(accidents.ts, xlab = "Month", ylab = "Number of accidents", main="Seasonal plot", year.labels.left = TRUE, col =1:20, pch = 19)

monthplot(accidents.ts, xlab = "Month", ylab = "Number of accidents", main="Seasonal Standard deviation",xaxt = "n")
axis(1, at = 1:12, labels = month.abb, cex = 0.8)

ntrain = length(accidents.ts) - 12
train.ts = window(accidents.ts, start =c(2016,1), end = c(2016,ntrain))
valid.ts = window(accidents.ts, start =c(2016, ntrain +1 ), end = c(2016,ntrain+ 12))

install.packages("MLmetrics")
library(MLmetrics)
library(forecast)

#naive model
model_naive = snaive(train.ts, h = 12)
MAPE(model_naive$mean, 12)*100
accuracy(model_naive, valid.ts)
plot(accidents.ts, xlab = "Time", ylab = "Number of accidents", )
lines(model_naive$mean, col="red", lwd=2)

#ARIMA model
model_arima = auto.arima(train.ts)
summary(model_arima)
model_arima_pred = forecast(model_arima, h= 12, level = 0)
accuracy(model_arima_pred, valid.ts)
Box.test(model_arima$residuals)
plot(model_arima_pred)

pred = predict(auto.arima(train.ts), n.ahead = 10*12)
plot(accidents.ts, xlab = "Time", ylab = "Number of accidents", xlim = c(2016,2027))
lines(pred$pred, col="red", lwd=2)

Forecast_val = ((forecast(accidents.ts, h=30)))
plot(accidents.ts, xlab = "Time", ylab = "Number of accidents", xlim = c(2016,2027))
lines(Forecast_val$mean, col="red", lwd=2)

Forecast_val2 = (forecast(auto.arima(train.ts), h=30))
plot(accidents.ts, xlab = "Time", ylab = "Number of accidents", xlim = c(2016,2027))
lines(Forecast_val2$mean, col="red", lwd=2)

highest_accident_zipcode=my_func(df, quo(Zipcode))
head(highest_accident_zipcode, 20)
ggplot(data=head(highest_accident_zipcode, 20), aes(x=Zipcode, y=my_count)) +
  geom_bar(stat="identity", fill="steelblue", width = 1)+
  geom_text(aes(label=my_count), vjust=1.6, color="white", size=3.5)+
  theme_minimal()

highest_accident_area = my_func(df, quo(Street))
head(highest_accident_area, 20)
ggplot(data=head(highest_accident_area, 20), aes(x=Street, y=my_count)) +
  geom_bar(stat="identity", fill="steelblue", width = 1)+
  geom_text(aes(label=my_count), vjust=1.6, color="white", size=3.5)+
  theme_minimal()



install.packages("usmap")
install.packages("rgdal")
library(usmap)
library(ggplot2)
library(usmap)
library(ggplot2)

# Lat/Lon of Sioux Falls, SD
test_data = data.frame(df$Start_Lng,df$Start_Lat)

transformed_data = usmap_transform(test_data)
plot_usmap("states",labels = TRUE) + 
  geom_point(data = transformed_data, 
             aes(x = df.Start_Lng.1, y = df.Start_Lat.1), 
             color = "red",
             size = .5)

#Data Kriging model
#For Colorado 
library(gstat)
install.packages("raster")
library(sp)
library(rgdal)
library(raster)
library(gstat)
#set.seed(100)
df_for_Kriging_model = data.frame(df[df$State == "CO",])
#df_for_Kriging_model = sample_n(df_for_Kriging_model, 100)
df_for_Kriging_model = head(df_for_Kriging_model, 15)
class(df_for_Kriging_model)
df_for_Kriging_model = data.frame(df_for_Kriging_model$Start_Lng,df_for_Kriging_model$Start_Lat,df_for_Kriging_model$Severity,df_for_Kriging_model$ID)
random_generator = sample(1:nrow(df_for_Kriging_model), round(nrow(df_for_Kriging_model)*.3), replace=F)
train_df = df_for_Kriging_model %>%
  filter(!df_for_Kriging_model.ID %in% random_generator) %>% 
  slice(1:200)
coordinates(train_df) = ~ df_for_Kriging_model.Start_Lng + df_for_Kriging_model.Start_Lat
class(train_df)
proj4string(train_df) = CRS("+proj=longlat +datum=WGS84")
lzn.vgm = variogram(log(df_for_Kriging_model.Severity) ~  1, data = train_df, width=0.1)
lzn.fit = fit.variogram(lzn.vgm, vgm("Gau", "Sph", "Mat", "Exp"))
plot(lzn.vgm,lzn.fit)


us = getData('GADM', country = 'US', level = 1)
us$NAME_1
colorado = us[us$NAME_1 == "Colorado",]
# check the CRS to know which map units are used
proj4string(colorado)
# "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"
# Create a grid of points within the bbox of the SpatialPolygonsDataFrame 
# colorado with decimal degrees as map units
grid = makegrid(colorado, cellsize = 0.1) # cellsize in map units!

# grid is a data.frame. To change it to a spatial data set we have to
grid = SpatialPoints(grid, proj4string = CRS(proj4string(colorado)))

plot(colorado)
plot(grid, pch = ".", add = T)

plot1 = df_for_Kriging_model %>% as.data.frame %>%
  ggplot(aes(df_for_Kriging_model.Start_Lng, df_for_Kriging_model.Start_Lat)) + geom_point(size=1) + coord_equal() + 
  ggtitle("Points with measurements")

# this is clearly gridded over the region of interest
plot2 = grid %>% as.data.frame %>%
  ggplot(aes(x1, x2)) + geom_point(size=1) + coord_equal() + 
  ggtitle("Points at which to estimate")

library(gridExtra)
grid.arrange(plot1, plot2, ncol = 2)

coordinates(grid) = ~ x1 + x2 
lzn.kriged = krige(log(df_for_Kriging_model.Severity) ~ 1, train_df, grid, model=lzn.fit)
lzn.kriged %>% as.data.frame %>%
  ggplot(aes(x=x1, y=x2)) + geom_tile(aes(fill=var1.pred)) + coord_equal() +
  scale_fill_gradient(low = "yellow", high="red") +
  theme_bw()
