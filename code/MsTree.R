#install.packages('reticulate',repos='http://cran.us.r-project.org')
library('reticulate')
#BiocManager::install("HGC")
library(HGC)



##Multi-scale spatial tree
#embedding
use_condaenv("ENV")
source_python("/home/RUN_GNMF.py")


sp_sim_count <- read.csv('/data/simulation/gene10000/sp_sim_count_ieffect5.csv',header = TRUE,row.names=1)
sp_sim_count <- as.matrix(sp_sim_count)

ReduceDim <- read.csv('/data/simulation/gene10000/sp_sim_ieffect5_embedding.csv',header = TRUE,row.names=1)
location <- read.csv('/data/simulation/gene10000/sp_sim_location.csv',header = TRUE,row.names=1)

#Hierarchical clustering
MOB.SNN <- SNN.Construction(ReduceDim, k = 25, threshold = 1/15)  
MOB.ClusteringTree <- HGC.dendrogram(G = MOB.SNN)

hier_clus2 <- cutree(MOB.ClusteringTree, k = 2)
hier_clus3 <- cutree(MOB.ClusteringTree, k = 3)
hier_clus4 <- cutree(MOB.ClusteringTree, k = 4)
hier_clus5 <- cutree(MOB.ClusteringTree, k = 5)

hier_clus <- cbind(hier_clus2,hier_clus3)
hier_clus <- cbind(hier_clus,hier_clus4)
hier_clus <- cbind(hier_clus,hier_clus5)

layer <- dim(hier_clus)[2]



AllEqual <- function(x) {
  all(x == x[1])
}


table(hier_clus5)
hier_select <- which(table(hier_clus5) > 10)#remove categories with too few elements
num_cluster <- length(hier_select)


##Multi-scale tree testing
p_all <- c()
num_all1 <- c()
Cauchy1 <- c()
num_layer1 <- c()


for (k in 1:dim(sp_sim_count)[1]) {
  
  count <- sp_sim_count[k,]
  
  p <- matrix(NA,num_cluster,layer)
  
  for (i in 1:num_cluster) {
    
    temp <- which(hier_clus[,layer]==hier_select[i])
    count_cluster <- count[temp]
    i_neigh <- which(hier_clus[,layer]!=hier_select[i])
    count_neigh <- count[i_neigh]
    
    if(AllEqual(c(count_cluster,count_neigh))){
      p[i,1] <- 1
    }else if(AllEqual(count_cluster) & AllEqual(count_neigh)){
      p[i,1] <- 0
    } else {
      wilcox <- wilcox.test(count_cluster, count_neigh, alternative = "two.sided")
      p[i,1] <- wilcox[["p.value"]]
    }
    
    temp_layer <- layer
    n <- 1
    
    while (temp_layer > 1 ) {
      temp_layer <- temp_layer-1
      n <- n+1
      
      #print(temp_layer)
      #print(n)
      
      length(unique(hier_clus[,temp_layer]))
      
      for (a in 1:length(unique(hier_clus[,temp_layer]))) {
        
        if (any(temp %in% which(hier_clus[,temp_layer]==a))) {
          
          temp_contain <- which(hier_clus[,temp_layer]==a)
          count_contain <- count[temp_contain]
          
          contain_neigh <- which(hier_clus[,temp_layer]!=a)
          count_neigh <- count[contain_neigh]
          
          
          if(AllEqual(c(count_neigh,count_contain))){
            p[i,n] <- 1
          }else if(AllEqual(count_neigh) & AllEqual(count_contain)){
            p[i,n] <- 0
          } else {
            wilcox <- wilcox.test(count_contain, count_neigh, alternative = "two.sided")
            p[i,n] <- wilcox[["p.value"]]
          }
          
        } 
      }
      
      #print(p)
    }
    
  }   
  
  
  p_all[k] <- list(p)
  
  
  
  temp <-c()
  for (i in 1:num_cluster) {
    tempp <- p[i,1:layer-1]
    #temp[i] <- 1 - pcauchy(sum((1/(layer-1))*tan((0.5-tempp)*pi)))
    temp[i] <- pcauchy(mean(tan((0.5-tempp)*pi)), lower.tail = F)
  }
  Cauchy1[k] <- min(temp)
  num_all1[k] <- which.min(temp)
  num_layer1[k] <- which.min(p[which.min(temp),])
  
  
}


Cauchy_adjust <- p.adjust(Cauchy1, method = "holm", n = length(Cauchy1))

