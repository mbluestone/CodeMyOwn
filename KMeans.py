import numpy as np

class KMeans():
    def __init__(self,k):
        self.k = k
        
    def fit(self,data):
        
        self.initialize_centroids(data)

        self.done = False
        while not self.done:
        
            self.calculate_clusters(data)

            self.update_centroids()
        
        
    def initialize_centroids(self,data):
        
        limits = [(min(data[:,i]),max(data[:,i])) for i in range(data.shape[1])]
        
        self.centroids = dict()
        
        for k in range(self.k):
            
            self.centroids[k]=np.array([np.random.randint(lim[0],lim[1]) for lim in limits])
            
    def calculate_clusters(self,data):
        
        self.clusters = dict()
        
        for sample in data:
            
            cluster = np.argmin([np.linalg.norm(sample-c) for c in self.centroids.values()])
            if cluster in self.clusters:
                self.clusters[cluster].append(sample)
            else:
                self.clusters[cluster] = [sample]
            
    def update_centroids(self):
        
        new_centroids = {k:np.mean(v,axis=0) for k,v in self.clusters.items()}
        
        if all([all(v==self.centroids[k]) for k,v in new_centroids.items()]):
            self.done = True
        
        else: 
            self.centroids = new_centroids
            

        