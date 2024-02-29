import os
import pandas as pd
from torch.autograd import Variable
from metric import analysis
from torch.utils.data import DataLoader
from utils import ProDataset
from model import LGS_PPIS
import torch,pickle
import warnings,numpy as np
import argparse
# Path
Dataset_Path = "./Dataset/"
Model_Path = "./Model_final/"


def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = {}

    for data in data_loader:
        with torch.no_grad():
            sequence_names, _, labels, node_features, graphs,distance_map,norm = data

            if torch.cuda.is_available():
                node_features = Variable(node_features.cuda(1))
                graphs = Variable(graphs.cuda(1))
                y_true = Variable(labels.cuda(1))
                distance_map = Variable(distance_map.cuda(1))
                norm = Variable(norm.cuda(1))
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

    return epoch_loss_avg, valid_true, valid_pred, pred_dict



def test(test_dataframe,args):
    test_loader = DataLoader(dataset=ProDataset(test_dataframe,args), batch_size=1, shuffle=True, num_workers=0)
    binary_acc = []
    precision = []
    recall = []
    f1 = []
    AUC = []
    AUPRC = []
    MCC = []
    for model_name in sorted(os.listdir(Model_Path)):
        print(model_name)
        model = LGS_PPIS(args.LAYER, 54, 256, 2, 0.1, args.LAMBDA, args.ALPHA)
        if torch.cuda.is_available():
            model.cuda(1)
        model.load_state_dict(torch.load(Model_Path + model_name, map_location='cuda:1'))

        epoch_loss_test_avg, test_true, test_pred, pred_dict = evaluate(model, test_loader)

        result_test = analysis(test_true, test_pred)

        print("========== Evaluate Test set ==========")
        print("Test loss: ", epoch_loss_test_avg)
        print("Test binary acc: ", result_test['binary_acc'])
        print("Test precision:", result_test['precision'])
        print("Test recall: ", result_test['recall'])
        print("Test f1: ", result_test['f1'])
        print("Test AUC: ", result_test['AUC'])
        print("Test AUPRC: ", result_test['AUPRC'])
        print("Test mcc: ", result_test['mcc'])
        print()

        binary_acc.append(result_test['binary_acc'])
        precision.append(result_test['precision'])
        recall.append(result_test['recall'])
        f1.append(result_test['f1'])
        AUC.append(result_test['AUC'])
        AUPRC.append(result_test['AUPRC'])
        MCC.append(result_test['mcc'])

        #print("Threshold: ", result_test['threshold'])

    # print('\n\n\n')
    # print('################################################################')
    # print("Average of Test binary acc: ",np.mean(binary_acc),'\t',np.std(binary_acc))
    # print("Average of Test precision:",np.mean(precision), '\t',np.std(precision))
    # print("Average of Test recall: ",np.mean(recall), '\t',np.std(recall))
    # print("Average of Test f1: ",np.mean(f1), '\t',np.std(f1))
    # print("Average of Test AUC: ",np.mean(AUC), '\t',np.std(AUC))
    # print("Average of Test AUPRC: ",np.mean(AUPRC), '\t',np.std(AUPRC))
    # print("Average of Test mcc: ",np.mean(MCC), '\t',np.std(MCC))
    # print('################################################################')
    # print('\n\n\n')


        # Export prediction
        # with open(model_name.split(".")[0] + "_pred.pkl", "wb") as f:
            # pickle.dump(pred_dict, f)


def test_one_dataset(dataset,args):
    IDs, sequences, labels = [], [], []
    for ID in dataset:
        IDs.append(ID)
        item = dataset[ID]
        sequences.append(item[0])
        labels.append(item[1])
    test_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    test_dataframe = pd.DataFrame(test_dic)
    test(test_dataframe,args)


def test_model(args):
    with open(Dataset_Path + "Test_60.pkl", "rb") as f:
        Test_60 = pickle.load(f)

    with open(Dataset_Path + "Test_315.pkl", "rb") as f:
        Test_315 = pickle.load(f)

    with open(Dataset_Path + "UBtest_31.pkl", "rb") as f:
        UBtest_31 = pickle.load(f)

    Btest_31 = {}
    with open(Dataset_Path + "bound_unbound_mapping.txt", "r") as f:
        lines = f.readlines()[1:]
    for line in lines:
        bound_ID, unbound_ID, _ = line.strip().split()
        Btest_31[bound_ID] = Test_60[bound_ID]

    print("Evaluate LGS-PPIS on Test_60")
    test_one_dataset(Test_60,args)

    print("Evaluate LGS-PPIS on Test_315")
    test_one_dataset(Test_315,args)

    print("Evaluate LGS-PPIS on Btest_31")
    test_one_dataset(Btest_31,args)

    print("Evaluate LGS-PPIS on UBtest_31")
    test_one_dataset(UBtest_31,args)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--LAYER', type=int, default=6,help="the number of layers")
    parser.add_argument('--LAMBDA', type=float, default=1.0,
                        help="Hyperparameter for identity mapping")
    parser.add_argument('--ALPHA', type=float, default=0.9,
                        help="Hyperparameter for initial residual")
    parser.add_argument('--MAP_CUTOFF', type=float, default=16,
                        help="Distance threshold")
    parser.add_argument('--device', type=int, default=1,
                        help="cuda device")
    args = parser.parse_args()
    print(args)
    test_model(args)

