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

    def forward(self, protFileCode, chainLengths):
        chainIDs = ["A"] # Fix for multichain TODO
        #print(x.shape)
        #print(self.cm.edge_index.shape)
        #print(self.cm.shape)
        builtContactGraph = buildContactGraph(protFileCode)

        
        edgeList = []
        for con in cg:
            edgeList.append(torch.tensor([parseNode(chainIDs, chainLengths, con[0]), parseNode(chainIDs, chainLengths, con[1])]))
        CM = torch.stack(edgeList, dim=1) # 
        #self.cm = Data(CM.contiguous()).edge_index
        #self.cm = CM

        print(self.gat(x, CM).shape)

class mainModel(nn.Module):
    def __init__(self, batchSize):
        self.batchSize = batchSize

        proteinSeqReader = csv.DictReader(csvFileName)
        proteinSeqs = dict()
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



    def forward(self, protFileCode): # TODO batch support inEmbed and GAT Parts; the other parts should be compatible
        # Embed Part
        seq = self.proteinSeqs(protFileCode)
        embed2D = Embeddings2D(seq, self.batchSize)
        embed1D = Embeddings1D(seq, self.batchSize)

        # GAT part
        gatOut = self.gat(protFileCode, len(seq))
        gatReadable = self.preReadout(gatOut)
        gatRead = torch.mean(gatReadable, dim=2)

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

        #TODO some manner of saving this shit for KNN & classification

model = GatFCM()
#X = torch.zeros(())
#print(model(X).shape)
chainIDs = ["A"]
chainLengths = [1000]
cg = buildContactGraph("AF-L0R819-F1-model_v4")
#edgeList = []
#CM = torch.stack(edgeList, dim=1) # 

gat = GatFCM(chainIDs, chainLengths, cg)

gat(torch.zeros(1000,1028))
#print(CM.contiguous())
