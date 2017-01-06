### this is file is used to format the code with 'formatR' package, 
### and also check the 'Mateinfo.txt' file using 'yamldebugger' package
rm(list = ls())
wd = 'F:/PHD/IRTG/courses/SPL/Quantlet'
setwd(wd)

############################ check the metainfo file #####################
#install.packages('devtools')

library(devtools)
#devtools::install_github("lborke/yamldebugger")

library(yamldebugger)
View(data.frame(allKeywords, stringsAsFactors = F))

### trial on the first file
check_metainfo = function(workdir) {
    setwd(workdir)
    d_init = yaml.debugger.init(workdir, show_keywords = TRUE)
    qnames = yaml.debugger.get.qnames(RootPath = d_init$RootPath)
    d_results = yaml.debugger.run(qnames, d_init)
    OverView = yaml.debugger.summary(qnames, d_results, summaryType = "mini")
}
check_metainfo(wd)

####################### get all the code formated #####################
library(formatR)

filename = dir()
wds = paste(wd, '/', filename, sep = '')

code_format = function(wd) {
    setwd(wd)  # set working dictionary to current file
    aa = dir(pattern = '\\.R$')
    aa1 = sapply(aa, function(x) paste('Import_', x, sep = ''))
    file.rename(aa, aa1)  # rename the input codes
    
    mapply(FUN = tidy_source, source = aa1, file = aa, arrow = F)
    file.remove(aa1)
}
lapply(wds, code_format)  ## format the code using 'formatR' package






