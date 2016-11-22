### this is the file to look at the variables of the data set:

setwd('F:/PHD/IRTG/courses/SPL_statistic_programming_language/Data')
rm(list=ls())

library(readr)

## import data description
datades1 = read.table('data_description.txt',skip = 2, sep = ':')
datades = read.table('data_description.txt',skip = 2, sep = ':')    ## tow variables loss.............
names(datades) = c('varname','vardes')

## import data
train = read_csv('ames_train.csv')

numNa = sapply(train,function(x) sum(is.na(x)))
numNa = data.frame(varname = names(numNa), numNa = unname(numNa))

datades = merge(datades,numNa,by = 'varname')

datades$varcat = character(nrow(datades))
datades$NaImSug = character(nrow(datades))

### can be modified for later use, a good way to look at the data
for (i in 1:nrow(datades)) {
    print(datades[i,])
    datades$varcat[i] = scan(what = character())
    print(datades[i,])
    datades$NaImSug[i] = scan(what = character())
    print(datades[i,])
}    
    
datades[3,'NaImSug'] = 'no need'
datades[4,'NaImSug'] = 'factor, can be ordered or not, NA means no Alley, can be changed to 0'
datades[5,'NaImSug'] = 'factor, can be ordered or not'
datades[6,'NaImSug'] = 'ordered factor, NA means that no basement'
datades[7,'NaImSug'] = 'factor, can be ordered or not, NA may means no basement'
datades[10,'NaImSug'] = 'ordered factor, NA may means no basement'
datades[11,'NaImSug'] = 'ordered factor, NA may means no basement'
datades[14,'NaImSug'] = 'ordered factor, NA may means no basement'


datades[1:7,'varcat'] = c('size_total','size_total','comfort_porch','comfort_alley','comfort_dwelling','comfort_basement','comfort_basement')

write_csv(datades,'data_description.csv')



