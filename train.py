import torch, os, engine, classify, utils, sys
import numpy as np 
import torch.nn as nn
from sklearn.model_selection import train_test_split

dataset_name = "cifar10"
device = "cuda"
root_path = "./target_model"
log_path = os.path.join(root_path, "target_logs")
model_path = os.path.join(root_path, "target_ckp")
os.makedirs(model_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

def main(args, model_name, trainloader, testloader):
    n_classes = args["dataset"]["n_classes"]
    mode = args["dataset"]["mode"]
    if model_name == "VGG16":
        if mode == "reg": 
            net = classify.VGG16(n_classes)
        elif mode == "vib":
            net = classify.VGG16_vib(n_classes)
    # eval
    elif model_name == "VGG19":
        print("Training eval classifier...")
        net = classify.VGG19()

    elif model_name == "ResNet50":
        net = classify.ResNet50()

    elif model_name == "Densenet121":
        net = classify.Densenet121(n_classes)
        
    else:
        print("Model name Error")
        exit()

    optimizer = torch.optim.SGD(params=net.parameters(),
							    lr=args[model_name]['lr'], 
            					momentum=args[model_name]['momentum'], 
            					weight_decay=args[model_name]['weight_decay'])
	
    epochs = args[model_name]["epochs"]
    criterion = nn.CrossEntropyLoss().cuda()
    net = torch.nn.DataParallel(net).to(device)

    mode = args["dataset"]["mode"]
    n_epochs = args[model_name]['epochs']
    best_ACC = 0
    print("Start Training!")
	
    if mode == "reg":
        best_model, best_acc = engine.train_reg(args, net, criterion, optimizer, trainloader, testloader, n_epochs)
    elif mode == "vib":
        best_model, best_acc = engine.train_vib(args, net, criterion, optimizer, trainloader, testloader, n_epochs)
	
    # torch.save({'state_dict':best_model.state_dict()}, os.path.join(model_path, "{}_evalnew_{:.2f}.tar").format(model_name, best_acc))
    torch.save({'state_dict':best_model.state_dict()}, os.path.join(model_path, "{}_{:.2f}.tar").format(model_name, best_acc))

if __name__ == '__main__':
    file = "./config/classify_cifar.json"
    # file = "./config/eval_cifar.json"
    args = utils.load_json(json_file=file)
    model_name = args['dataset']['model_name']

    log_file = "{}.txt".format(model_name)
    utils.Tee(os.path.join(log_path, log_file), 'w')

    os.environ["CUDA_VISIBLE_DEVICES"] = args['dataset']['gpus']
    print(log_file)
    print("---------------------Training [%s]---------------------" % model_name)
    utils.print_params(args["dataset"], args[model_name], dataset=args['dataset']['name'])

    train_file = args['dataset']['train_file_path']
    test_file = args['dataset']['test_file_path']
    _, trainloader = utils.init_dataloader(args, train_file, mode="train")
    _, testloader = utils.init_dataloader(args, test_file, mode="test")

    main(args, model_name, trainloader, testloader)