import pickle,pandas as pd,torch,argparse,warnings,os
from model import LGS_PPIS
from torch.utils.data import DataLoader
from utils import ProDataset
from metric import analysis
from torch.autograd import Variable
import numpy as np

def train_one_epoch(model, data_loader,args):
    epoch_loss_train = 0.0
    n = 0
    for data in data_loader:
        model.optimizer.zero_grad()
        _, _, labels, node_features, graphs,distance_map,norm = data

        if torch.cuda.is_available():
            node_features = Variable(node_features.cuda(args.device))
            graphs = Variable(graphs.cuda(args.device))
            y_true = Variable(labels.cuda(args.device))
            distance_map = Variable(distance_map.cuda(args.device))
            norm = Variable(norm.cuda(args.device))
        else:
            node_features = Variable(node_features)
            graphs = Variable(graphs)
            y_true = Variable(labels)
            distance_map = Variable(distance_map)
            norm = Variable(norm)

        node_features = torch.squeeze(node_features)
        graphs = torch.squeeze(graphs)
        y_true = torch.squeeze(y_true)
        distance_map = torch.squeeze(distance_map)
        norm = torch.squeeze(norm)

        y_pred = model(node_features, graphs,distance_map,norm)  # y_pred.shape = (L,2)

        # calculate loss
        loss = model.criterion(y_pred, y_true)

        # backward gradient
        loss.backward()

        # update all parameters
        model.optimizer.step()

        epoch_loss_train += loss.item()
        n += 1

    #epoch_loss_train_avg = epoch_loss_train / n



def train_full_model(all_dataframe, args):
    print("\n\nTraining a full model using all training data...\n")
    model = LGS_PPIS(args.LAYER, 54, 256, 2, 0.1, args.LAMBDA, args.ALPHA)
    if torch.cuda.is_available():
        model.cuda(1)

    with open("./Dataset/Test_60.pkl", "rb") as f:
        Test_60 = pickle.load(f)
    IDs, sequences, labels = [], [], []
    for ID in Test_60:
        IDs.append(ID)
        item = Test_60[ID]
        sequences.append(item[0])
        labels.append(item[1])
    test_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    test_dataframe = pd.DataFrame(test_dic)

    train_loader = DataLoader(dataset=ProDataset(all_dataframe,args), batch_size=1, shuffle=True, num_workers=6)

    test_loader = DataLoader(dataset=ProDataset(test_dataframe,args), batch_size=1, shuffle=True, num_workers=6)

    # binary_acc=[]
    # precision=[]
    # recall=[]
    # f1=[]
    # AUC=[]
    # AUPRC=[]
    # MCC=[]
    metric_record = np.zeros((50,7))
    for epoch in range(50):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()

        train_one_epoch(model, train_loader,args)

        model.eval()
        epoch_loss = 0.0
        n = 0
        valid_pred = []
        valid_true = []
        pred_dict = {}
        for data in test_loader:
            with torch.no_grad():
                sequence_names, _, labels, node_features, graphs,distance_map, norm = data
                node_features = Variable(node_features.cuda(args.device))
                graphs = Variable(graphs.cuda(args.device))
                y_true = Variable(labels.cuda(args.device))
                distance_map = Variable(distance_map.cuda(args.device))
                norm = Variable(norm.cuda(args.device))

                node_features = torch.squeeze(node_features)
                graphs = torch.squeeze(graphs)
                y_true = torch.squeeze(y_true)
                distance_map = torch.squeeze(distance_map)

                y_pred = model(node_features, graphs,distance_map,norm)
                loss = model.criterion(y_pred, y_true)
                softmax = torch.nn.Softmax(dim=1)
                y_pred = softmax(y_pred)
                y_pred = y_pred.cpu().detach().numpy()
                y_true = y_true.cpu().detach().numpy()
                valid_pred += [pred[1] for pred in y_pred]
                valid_true += list(y_true)
                pred_dict[sequence_names[0]] = [pred[1] for pred in y_pred]

                epoch_loss += loss.item()
                n += 1
        epoch_loss_avg = epoch_loss / n
        result_test = analysis(valid_true,valid_pred)
        print("========== Evaluate Test set ==========")
        print("Test loss: ", epoch_loss_avg)
        print("Test binary acc: ", result_test['binary_acc'])
        metric_record[epoch,0]=result_test['binary_acc']

        print("Test precision:", result_test['precision'])
        metric_record[epoch, 1] = result_test['precision']

        print("Test recall: ", result_test['recall'])
        metric_record[epoch, 2] = result_test['recall']

        print("Test f1: ", result_test['f1'])
        metric_record[epoch, 3] = result_test['f1']

        print("Test AUC: ", result_test['AUC'])
        metric_record[epoch, 4] = result_test['AUC']

        print("Test AUPRC: ", result_test['AUPRC'])
        metric_record[epoch, 5] = result_test['AUPRC']

        print("Test mcc: ", result_test['mcc'])
        metric_record[epoch, 6] = result_test['mcc']

        print("Threshold: ", result_test['threshold'])

        print()

        # if (result_test['binary_acc']>0.77) and (result_test['precision']>0.36) and (result_test['recall']>0.58) and (result_test['f1']>0.45) and (result_test['AUC']>0.78) and (result_test['AUPRC']>0.42) and(result_test['mcc']>0.33) and (args.save_model is True):
        if args.save_model is True:
            torch.save(model.state_dict(), os.path.join('./Model/Full_model_{}.pkl'.format(epoch + 1)))

    num_rows, num_cols = metric_record.shape
    ranks = np.zeros((num_rows, num_cols), dtype=int)
    for col_index in range(num_cols):
        
        sorted_indices = np.argsort(metric_record[:, col_index])

        
        for row_index in range(num_rows):
        
            rank = np.where(sorted_indices == row_index)[0][0] + 1

        
            ranks[row_index, col_index] = rank

    index = np.argmax(np.sum(ranks,axis=1))
    print('Best Epoch:',index+1)
    print("Test binary acc: ", metric_record[index,0])

    print("Test precision:", metric_record[index,1])

    print("Test recall: ", metric_record[index,2])


    print("Test f1: ", metric_record[index,3])


    print("Test AUC: ", metric_record[index,4])


    print("Test AUPRC: ", metric_record[index,5])


    print("Test mcc: ", metric_record[index,6])

def train(args):
    with open("./Dataset/Train_335.pkl", "rb") as f:
        Train_335 = pickle.load(f)

    IDs, sequences, labels = [], [], []

    for ID in Train_335:
        IDs.append(ID)
        item = Train_335[ID]
        sequences.append(item[0])
        labels.append(item[1])

    train_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    train_dataframe = pd.DataFrame(train_dic)
    train_full_model(train_dataframe,args)








if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--LAYER', type=int, default=6, help="the number of layers")
    parser.add_argument('--LAMBDA', type=float, default=1.0,
                        help="Hyperparameter for identity mapping")
    parser.add_argument('--ALPHA', type=float, default=0.9,
                        help="Hyperparameter for initial residual")
    parser.add_argument('--MAP_CUTOFF', type=float, default=16,
                        help="Distance threshold")
    parser.add_argument('--save_model', type=bool, default=False,
                        help="Save trained model each epoch or not")
    parser.add_argument('--device', type=int, default=1,
                        help="cuda device")
    args = parser.parse_args()
    print(args)
    train(args)
