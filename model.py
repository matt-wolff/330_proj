import torch
import torch.nn as nn
from torch_geometric.nn.models import GAT
from torch_geometric.data import Data
from formContactGraph import buildContactGraph
from embeddings import Embeddings1D, Embeddings2D
import csv

def parseNode(cn, cl, dx):
    cl_index = cn.index(dx[0])
    return sum(cl[:cl_index]) + int(dx[1].strip())-1

# No Batching currently
class GatFCM(torch.nn.Module): # GAT Form Contact Map
    def __init__(self):

        super().__init__()


        self.gat = GAT(
            in_channels = 1028,
            hidden_channels = 512,
            num_layers = 1,
            out_channels = 256,
            v2 = True,
            dropout = 0.0,
            act = 'relu',
            norm = None
        )

    def forward(self, x, protFileCodes, chainLengths):
        chainIDs = ["A"] # Fix for multichain TODO
        #print(x.shape)
        #print(self.cm.edge_index.shape)
        #print(self.cm.shape)
        CMs = list()
        for protFileCode in protFileCodes:
            builtContactGraph = append(buildContactGraph(protFileCode))
            edgeList = []
            for con in cg:
                edgeList.append(torch.tensor([parseNode(chainIDs, chainLength, con[0]), parseNode(chainIDs, chainLength, con[1])]))
            
            CM = torch.stack(edgeList, dim=1) # 
            CMs.append(CM)
        CMs = torch.stack(CMs, dim=0)
        #self.cm = Data(CM.contiguous()).edge_index
        #self.cm = CM

        return self.gat(x, CMs)

class mainModel(nn.Module):
    def __init__(self):

        proteinSeqReader = csv.DictReader(csvFileName)
        proteinSeqs = dict()
        proteinSeqReader = proteinSeqReader[1:]
        for row in proteinSeqReader:
            proteinSeqs[row[0]] = row[1]
        self.proteinSeqs = proteinSeqs
        
        self.gat = GatFCM()
        self.preReadout = nn.Linear(256, 256) 

        self.preTransform = nn.Linear(1024, 512)
        self.seqTransform = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                512, 8, 512, dropout=0.0, batch_first = True
            ), num_layer = 6, norm = nn.LayerNorm(512))
        self.preCombination = nn.linear(512, 256)

        self.postMean = nn.Linear(1024, 256)

        self.postCombination = nn.Sequential(
            nn.Linear(768, 768),
            nn.LeakyReLU(),
            nn.LayerNorm(768),
            nn.Linear(768, 768),
            nn.LeakyReLU(),
            nn.LayerNorm(768),
            nn.Linear(768, 768),
            nn.LeakyReLU(),
            nn.LayerNorm(768),
            nn.Linear(768, 768),
            nn.LeakyReLU(),
            nn.LayerNorm(768),
        )
        self.projection = nn.Linear(768, 32) # Assumes projection space is 32d



    def forward(self, protFileCodes): 
        # Embed Part
        seqs = [self.proteinSeqs[protFileCode] for protFileCode in protFileCodes]
        embed2D = Embeddings2D(seqs)
        embed1D = Embeddings1D(seqs)

        # GAT part
        gatOut = self.gat(embed2D, protFileCodes, [[len(seq)] for seq in seqs])
        gatReadable = self.preReadout(gatOut)
        gatRead = torch.mean(gatReadable, dim=1)

        # Seq Transform Part
        transformIn = self.preTransform(embed2D)
        transformOut = self.seqTransform(transformIn)
        transformRead = self.preCombinaiton(transformOut)

        # Seq Mean Part
        meanOut = self.postMean(embed1D)

        # Combination and Projection Part
        combined = torch.cat((meanOut, transformRead, gatRead), dim=1)
        combinedOut = self.postCombination(combined)
        projected = self.projection(combinedOut)

        return projected

class knnClassifier (nn.Module):
    def __init__ (self, batchSize, k):
        super().__init__()
        self.batchSize = batchSize
        self.k = k
        self.model = mainModel()

    def forward (self, posUniProtIDs, negUniProtIDs, queryUniProtIDs):
        posEmbeds = []
        for batch_start in range(0, len(posUniProtIDs), self.batchSize):
            batchIDs = posUniProtIDs[batch_start:batch_start+self.batchSize]
            posEmbeds.append(self.model(batchIDs))
        posEmbeddings = torch.cat(posEmbeds, dim=0)
        posPrototype = torch.mean(posEmbeddings, dim=1)

        negEmbeds = []
        for batch_start in range(0, len(negUniProtIDs), self.batchSize):
            batchIDs = negUniProtIDs[batch_start:batch_start+self.batchSize]
            negEmbeds.append(self.model(batchIDs))
        negEmbeddings = torch.cat(negEmbeds, dim=0)
        negPrototype = torch.mean(negEmbeddings, dim=1)

        queryEmbeds = []
        for batch_start in range(0, len(queryUniProtIDs), self.batchSize):
            batchIDs = queryUniProtIDs[batch_start:batch_start+self.batchSize]
            queryEmbeds.append(self.model(batchIDs))
        queryEmbeddings = torch.cat(queryEmbeds, dim=0)

        queryDists = torch.stack((queryEmbeddings, queryEmbedding), dim=0)
        queryDists[0,:,:] = queryDists[0,:,:] - posPrototype.reshape((1,-1))
        queryDists[1,:,:] = queryDists[1,:,:] - negPrototype.reshape((1,-1))
        queryDists = torch.pow(queryDists, 2)
        queryDists = torch.sum(queryDists, dim=2)
        # Now QueryDists is [2, numExamples], treat logits
        
        return queryDists.T # [numExamples, 2]

        


#model = GatFCM()
#X = torch.zeros(())
#print(model(X).shape)
#chainIDs = ["A"]
#chainLengths = [1000]
#cg = buildContactGraph("AF-L0R819-F1-model_v4")
#edgeList = []
#CM = torch.stack(edgeList, dim=1) # 

#gat = GatFCM(chainIDs, chainLengths, cg)
#gat([,]) # TODO test codes
#gat(torch.zeros(1000,1028))
#print(CM.contiguous())

if "__main__" == __name__:
    pass 
