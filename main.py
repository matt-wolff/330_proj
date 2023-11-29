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
        device = "cuda"
    else:
        device = "cpu"

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
        train(hyper,train_df,val_df,device)
    elif args.mode=="continue_train":
        hyper = defaultHypers(args.learning_rate)
        train(hyper, train_df, val_df, device, (args.model_path, args.model_filename))
    elif args.mode=="hp_search":
        hyperranges=hyperrangeWrap(defaultHypers(args.learning_rate))#, args.run_name))
        hyperranges["ball_radius"] = [4] #[0.5,1,4]
        hyperranges["projection_space_dims"] = [128]#[8,32,128]
        #hyperranges["run_name"] = [args.run_name] # I think this is a bad idea with hps NOTE
        hypers = getRangeCombos(hyperranges)
        for hyper in hypers:
            train(hyper,train_df,val_df,device)
    elif args.mode=="ablate": # NOTE set these to ideal arguments except for inmodel_type
        hyperranges=hyperrangeWrap(defaultHypers(args.learning_rate))#, args.run_name))
        #hyperranges["run_name"] = [args.run_name] # I think this is a bad idea with hps NOTE
        hyperranges["inmodel_type"] = [ProteinEmbedder, PEwoPostCombination,PEwoDirectEmbedding,PEwoGAT]
        hypers = getRangeCombos(hyperranges)
        for hyper in hypers:
            train(hyper,train_df,val_df,device)

    if args.mode=="validate" or args.mode=="test":
        # The following require model loading
        if (args.model_type == "ball"):
            hyper = defaultHypers(args.learning_rate)#, args.run_name) # These parameters are dummy, they will get replaced upon load

            model = ballClassifier(hyper, batchSize=1) # BS is dummy, will be overwritten on load
            emb = ESMEmbedder(device).to(device)
            model.model.emb = emb
        else:
            raise Exception("Sought model type not found")

        if args.model_path != "":
            model.load_state_dict(torch.load(f'{args.model_path}/{args.model_filename}'))
        else:
            model.load_state_dict(torch.load(args.model_filename))
        model.to(device)

        if args.mode=="validate":
            validate(model,val_df,device)
        elif args.mode=="test":
            test(model,test_df,device)


def train(hyper, train_df, val_df, device, model_data=("", "")):
    model_path, model_filename = model_data
    if model_filename:
        from_epoch = int(model_filename.split("_")[4][-4]) + 1  # If saved at epoch X, train from epoch X+1
        run_name = model_filename[5:-7]  # Assuming starts with "ball_" and ends with "_epochX"
    else:
        from_epoch = 0
        run_name = "run_" + str(datetime.now()).replace(" ", "_")
        with open("modelhistory", 'a') as f:
            f.write(",  ".join([str(a)+":"+str(b) for a,b in hyper.items()]))
            f.write("\n")

    hyper["run_name"] = run_name

    log_dir = f'./logs/{run_name}'
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    lr = hyper["learning_rate"]
    num_epochs = hyper["num_epochs"]

    inmodel = hyper["inmodel_type"](hyper, "data/residues.json")
    inmodel = inmodel.to(device)
    ball = ballClassifier(hyper,batchSize=8,model=inmodel).to(device)
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

    if model_filename is not None:
        emb = ESMEmbedder(device).to(device)
        ball.model.emb = emb
        if model_path != "":
            ball.load_state_dict(torch.load(f"{model_path}/{model_filename}"))
        else:
            ball.load_state_dict(torch.load(model_filename))
    else:
        ball.apply(initializeParams)
        emb = ESMEmbedder(device).to(device)
        ball.model.emb = emb
    ball.to(device)

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
            loss += 0.5 * torch.sum(torch.pow(torch.max(torch.Tensor([0]).to(device), ball.radius - neg_dists), 2))  # Doesn't add to loss if dist >= margin
            loss /= len(queryDists)
            loss.backward()
            optimizer.step()

            i_step = num_train_tasks*epoch + index
            writer.add_scalar('loss/train', loss.item(), i_step)

            if index % VAL_INTERVAL == 0:
                validate(ball, val_df, device, writer, i_step)
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

def test(model,ds,device):
    print("Starting Testing...")
    loss,acc,tp_val,fn_val,tn_val,fp_val = testcore(model,ds,"Testing",device)

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
    modes = ['train',"continue_train","hp_search",'validate','test','ablate']

    parser = argparse.ArgumentParser('Train a Ball Classifier!')
    parser.add_argument('--mode', type=str, help=(' '.join(modes)))
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='learning rate for the network')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--run_name', type=str, default='temp')
    parser.add_argument('--model_type', type=str, default='temp')
    parser.add_argument('--model_path', type=str, default='',
                        help="Path to file with saved model parameters. Ex: models/baseline/val_models")
    parser.add_argument('--model_filename', type=str, default='',
                        help="Filename of PyTorch model. Ex: ball_run_2023-11-28_21:26:51.266950_epoch0.pt")

    args = parser.parse_args()
    assert(args.mode in modes)

    main(args)
