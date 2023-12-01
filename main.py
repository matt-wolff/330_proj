import argparse
import sys
sys.path.append('..')

import pandas as pd
import torch.optim as optim
from model import ballClassifier,ProteinEmbedder, PEwoPostCombination,PEwoDirectEmbedding,PEwoGAT
import ast
import random
import torch.nn.functional as F
from torch.utils import tensorboard
import torch_geometric as pyg
import torch
from embeddings import ESMEmbedder
from tqdm import tqdm
from datetime import datetime
import copy
import json

VAL_INTERVAL = 50
DEVICE = 'cuda'
# TRAIN_PATH = 'data/go_tasks.csv'
# VAL_PATH = 'data/go_tasks.csv'
PATH = 'data/go_tasks.csv'
# learning_rates = [1e-6, 5e-6, 1e-5]
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)


def get_support_and_query_ids(row):
    pos_ids = ast.literal_eval(row["Positive IDs"])
    rand_pos_ids = random.sample(pos_ids, k=8)
    support_ids = rand_pos_ids[:5]
    query_pos_ids = rand_pos_ids[5:]

    neg_ids = ast.literal_eval(row["Negative IDs"])
    if len(neg_ids) < 3:
        all_other = ast.literal_eval(row["All Other IDs"])
        query_neg_ids = random.sample(neg_ids, k=len(neg_ids))
        query_neg_ids += random.sample(all_other, k=3 - len(neg_ids))
    else:
        query_neg_ids = random.sample(neg_ids, k=3)

    return support_ids, query_pos_ids, query_neg_ids

def getRangeCombos(space):
    hypers = list()
    hypers.append(dict())
    for hname,hrange in space.items():
        nhypers = list()
        for hyper in hypers:
            for hval in hrange:
                nhyper = copy.deepcopy(hyper)
                nhyper[hname] = hval
                nhypers.append(nhyper)
        hypers = nhypers
    return hypers
            

def defaultHypers(learning_rate):#, run_name):
    hyper = dict()
    hyper["learning_rate"] = learning_rate
    hyper["num_epochs"] = 5
    #hyper["run_name"] = run_name
    hyper["inmodel_type"] = ProteinEmbedder
    hyper["ball_radius"] = 1
    hyper["projection_space_dims"] = 32
    hyper["gat_layers"] = 2
    hyper["gat_hidden_size"] = 512
    hyper["gat_dropout"] = 0.0
    hyper["postcomb_dim"] = 512
    hyper["postcomb_layers"] = 4
    return hyper

def main(args):

    if args.device == "cuda" and torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    train_df = pd.read_csv('data/train_go_tasks.csv', encoding='utf-8')
    val_df = pd.read_csv('data/val_go_tasks.csv', encoding='utf-8')
    test_df = pd.read_csv('data/test_go_tasks.csv', encoding='utf-8')

    # Allows us to not touch the hyperparameters we are not interested in modifying

    def hyperrangeWrap(hyper):
        hyperranges = dict()
        for key,item in hyper.items():
            hyperranges[key] = [item]
        return hyperranges

    if args.mode=="train":
        hyper = defaultHypers(args.learning_rate)#, args.run_name)
        train(hyper,train_df,val_df,DEVICE)
    elif args.mode=="hp_search":
        hyperranges=hyperrangeWrap(defaultHypers(args.learning_rate))#, args.run_name))
        hyperranges["ball_radius"] = [4] #[0.5,1,4]
        hyperranges["projection_space_dims"] = [128]#[8,32,128]
        #hyperranges["run_name"] = [args.run_name] # I think this is a bad idea with hps NOTE 
        hypers = getRangeCombos(hyperranges)
        for hyper in hypers:
            train(hyper,train_df,val_df,DEVICE)
    elif args.mode=="ablate": # NOTE set these to ideal arguments except for inmodel_type
        hyperranges=hyperrangeWrap(defaultHypers(args.learning_rate))#, args.run_name))
        #hyperranges["run_name"] = [args.run_name] # I think this is a bad idea with hps NOTE 
        hyperranges["inmodel_type"] = [ProteinEmbedder, PEwoPostCombination,PEwoDirectEmbedding,PEwoGAT]
        hypers = getRangeCombos(hyperranges)
        for hyper in hypers:
            train(hyper,train_df,val_df,DEVICE)
    elif args.mode=="test_batch":
        modelpaths = list()
        with open(args.batchfile) as f:
            line = f.readline()
            while line:
                modelpaths.append(line.strip())
                line = f.readline()
 
        modelhist = list()
        with open("modelhistory") as f:
            line = f.readline()
            while line:
                modelhist.append(line)
                line = f.readline()
        histdict = dict()
        for hist in modelhist:
            his = hist.split("  ")
            runname = "ball_" + (":".join(his[-1].split(":")[1:])).strip()
            his = his[:-1]
            his = [attr[:-1] for attr in his]
            his = [attr.split(":") for attr in his]
            hyper = dict()
            lu = {
                "<class 'model.ProteinEmbedder'>":ProteinEmbedder,
                "<class 'model.PEwoPostCombination'>":PEwoPostCombination,
                "<class 'model.PEwoDirectEmbedding'>":PEwoDirectEmbedding,
                "<class 'model.PEwoGAT'>":PEwoGAT,
            }
            for attrname, attrval in his:
                if attrname == "inmodel_type":
                    hyper[attrname] = lu[attrval]
                else:
                    hyper[attrname] = int(attrval) if attrval.isnumeric() else float(attrval)
            histdict[runname]  = hyper
        print("models extracted")
    
        for modelpath in tqdm(modelpaths):
            runname = "_".join((modelpath.split("/")[-1]).split("_")[:-1])
            import pdb
            pdb.set_trace()
            hyper = histdict[runname]
            model = ballClassifier(hyper, batchSize=1)
            emb = ESMEmbedder(DEVICE).to(DEVICE)
            model.model.emb = emb
            model.load_state_dict(torch.load(modelpath))
            model.to(DEVICE)
            test(model,test_df,DEVICE,savestr=modelpath)
       
    
    if args.mode=="validate" or args.mode=="test" or args.mode=="continue_train":
        # The following require model loading
        if (args.model_type == "ball"):
            hyper = defaultHypers(args.learning_rate)#, args.run_name) # These parameters are dummy, they will get replaced upon load
            # TODO manual edit if desired
            hyper["from_epoch"]=4
            hyper["from_training_number"]="2023-11-27_18:08:03.197355"
            hyper["projection_space_dims"]=128
            hyper["ball_radius"]=1
            
            model = ballClassifier(hyper, batchSize=1) # BS is dummy, will be overwritten on load
            emb = ESMEmbedder(DEVICE).to(DEVICE)
            model.model.emb = emb
        else:
            raise Exception("Sought model type not found")

        model.load_state_dict(torch.load(args.model_path))
        model.to(DEVICE)

        if args.mode=="validate": 
            validate(model,val_df,DEVICE)
        elif args.mode=="test":
            test(model,test_df,DEVICE)
        elif args.mode=="continue_train":
            #hyper=defaultHypers(args.learning_rate)
            train(hyper,train_df,val_df,DEVICE,from_epoch=hyper["from_epoch"],trainingNumber=hyper["from_training_number"],use_model=model)

def train(hyper,train_df,val_df,DEVICE,from_epoch=0,trainingNumber=None,use_model=None):
    if trainingNumber is None:
        run_name = "run_"+str(datetime.now()).replace(" ","_")
    else:
        run_name = "run_"+str(trainingNumber)
    hyper["run_name"] = run_name
    if trainingNumber is None:
        with open("modelhistory", 'a') as f:
            f.write(",  ".join([str(a)+":"+str(b) for a,b in hyper.items()]))
            f.write("\n")

    log_dir = f'./logs/{run_name}'
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    lr = hyper["learning_rate"]
    num_epochs = hyper["num_epochs"]

    if use_model is None:
        inmodel = hyper["inmodel_type"](hyper, "data/residues.json")
        inmodel = inmodel.to(DEVICE)
        ball = ballClassifier(hyper,batchSize=8,model=inmodel).to(DEVICE)
    else:
        ball = use_model
    num_train_tasks, _ = train_df.shape

    def initializeParams(module): ## NOTE when you add in the pretrained model; ensure you do not initialize that
        if isinstance(module, torch.nn.Linear):
            module.weight.data = torch.nn.init.xavier_normal_(module.weight.data, gain=torch.nn.init.calculate_gain('relu'))
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.LayerNorm) or isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        if isinstance(module, pyg.nn.models.GAT):
            module.reset_parameters()

    if use_model is None:
        ball.apply(initializeParams)
    
        emb = ESMEmbedder(DEVICE).to(DEVICE)
        ball.model.emb = emb

    optimizer = optim.AdamW(ball.parameters(), lr=lr)
    for epoch in range(from_epoch, num_epochs):
        for index, row in tqdm(train_df.iterrows(), desc=f'Training epoch: {epoch}', total=len(train_df.index)):  # Iterating over each task
            optimizer.zero_grad()

            support_ids, query_pos_ids, query_neg_ids = get_support_and_query_ids(row)
            _, queryDists = ball.get_prototype_and_query_dists(support_ids, query_pos_ids + query_neg_ids)

            # contrastive loss https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1640964
            pos_dists = queryDists[:3]
            neg_dists = queryDists[3:]
            loss = 0.5 * torch.sum(torch.pow(pos_dists, 2))
            loss += 0.5 * torch.sum(torch.pow(torch.max(torch.Tensor([0]).to(DEVICE), ball.radius - neg_dists), 2))  # Doesn't add to loss if dist >= margin
            loss /= len(queryDists)
            loss.backward()
            optimizer.step()

            i_step = num_train_tasks*epoch + index
            writer.add_scalar('loss/train', loss.item(), i_step)

            if index % VAL_INTERVAL == 0:
                validate(ball, val_df, DEVICE, writer, i_step)
                if (index + VAL_INTERVAL) // num_train_tasks > index // num_train_tasks:
                    torch.save(ball.state_dict(), f'ball_{run_name}_epoch{epoch}.pt')


def validate(model,ds,device,writer=None,i_step=None): # Writer requires i_step
    print("Starting Validation...")
    loss,acc,tp_val,fn_val,tn_val,fp_val = testcore(model,ds,"Validating",device)
    if writer is not None:
        writer.add_scalar('loss/val', loss, i_step)
        writer.add_scalar('val_accuracy/', acc, i_step)
        writer.add_scalar('true_positive_val/', tp_val, i_step)
        writer.add_scalar('false_positive/', fp_val, i_step)
        writer.add_scalar('true_negative_val/', tn_val, i_step)
        writer.add_scalar('false_negative_val/', fn_val, i_step)

def test(model,ds,device,savestr=None):
    print("Starting Testing...")
    loss,acc,tp_val,fn_val,tn_val,fp_val = testcore(model,ds,"Testing",device)
    if savestr != None:
        with open("testRes", "a") as f:
            f.write(str({savestr: (loss.item(),acc.item(),tp_val.item(),fn_val.item(),tn_val.item(),fp_val.item())}))
            f.write("\n")

def testcore(model,ds,keyword,device):
    with torch.no_grad():
        val_losses, val_accuracies, tp, fn, tn, fp = [], [], [], [], [], []
        for iter_val, row_val in tqdm(ds.iterrows(), desc=f"{keyword}", total=len(ds.index)):
            support_ids_val, query_pos_ids_val, query_neg_ids_val = get_support_and_query_ids(row_val)
            probs = model(support_ids_val, query_pos_ids_val + query_neg_ids_val)
            targets = torch.Tensor([0,0,0,1,1,1]).to(device).to(torch.int64)
            loss = F.nll_loss(torch.log(probs), targets)
            
            true_positives = torch.sum((torch.argmax(probs,dim=1)==targets)[targets==1]).type(torch.int64).item()
            false_negatives = 3 - true_positives
            true_negatives = torch.sum((torch.argmax(probs,dim=1)==targets)[targets==0]).type(torch.int64).item()
            false_positives = 3 - true_negatives
            accuracy = torch.mean((torch.argmax(probs,dim=1)==targets).float()).item()
            
            val_losses.append(loss.item())
            val_accuracies.append(accuracy)
            tp.append(true_positives)
            fn.append(false_negatives)
            tn.append(true_negatives)
            fp.append(false_positives)

        loss_val = torch.mean(torch.Tensor(val_losses))
        accuracy_val = torch.mean(torch.Tensor(val_accuracies))
        tp_val = torch.mean(torch.Tensor(tp))
        fn_val = torch.mean(torch.Tensor(fn))
        tn_val = torch.mean(torch.Tensor(tn))
        fp_val = torch.mean(torch.Tensor(fp))
    return loss_val,accuracy_val,tp_val,fn_val,tn_val,fp_val


if __name__ == '__main__':
    modes = ['train',"hp_search",'validate','test','ablate',"continue_train","test_batch"]

    parser = argparse.ArgumentParser('Train a Ball Classifier!')
    parser.add_argument('--mode', type=str, help=(' '.join(modes)))
    parser.add_argument("--batchfile", type=str, default="ERROR")
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='learning rate for the network')
    parser.add_argument('--device', type=str, default='cuda')
    #parser.add_argument('--run_name', type=str, default='temp')
    parser.add_argument('--model_type', type=str, default='temp')
    parser.add_argument('--model_path', type=str, default='temp')

    args = parser.parse_args()
    assert(args.mode in modes)

    main(args)
