########################## Encode Time Variables  ###########################################################3
source("utils/visulizations.R")
library(plotly)

# year sold
price_per_factor_box(train$YrSold,"Year Sold") 
unique(X_com$YrSold)
# minor difference in year, encode as numerical: YrSinceSold

# Month Sold
price_per_factor_box(train$MoSold,"Month Sold")
price_per_factor_plot(train$MoSold,"Month Sold") 
unique(X_com$MoSold)
# minor difference in month sold --> encode as quarter dummies

price_per_factor_box(train$YearRemodAdd,"Year Rennovated")
price_per_factor_plot(train$YearRemodAdd,"Year Rennovated")
ggplotly()
unique(X_com$YearRemodAdd)
# create numeric variable: YrSinceRemodAdd


price_per_factor_plot(train$YearBuilt,"YrBuild") 
unique(train$YearBuilt)
# create numeric variable: YrSinceBuild

price_per_factor_plot(train$GarageYrBlt,"Garage Build")
# create numeric variable: YrSince

include_quarter_dummies <- function(X_com){
  X_com$SoldSecondQuartal <- ifelse(X_com$MoSold %in% 4:6  , 1, 0)  
  X_com$SoldThirdQuartal <- ifelse(X_com$MoSold %in% 7:9  , 1, 0)  
  X_com$SoldFourthQuartal <- ifelse(X_com$MoSold %in% 10:12, 1, 0)  
  X_com$MoSold <- NULL
  return(X_com)
}

