# # train_zinc.py
# import argparse
# import os
# import h5py
# import numpy as np

# from models.model_zinc import MoleculeVAE
# from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# import h5py
# import zinc_grammar as G
# import pdb


# rules = G.gram.split('\n')


# MAX_LEN = 277
# DIM = len(rules)
# LATENT = 56
# EPOCHS = 100
# BATCH = 500


# def get_arguments():
#     parser = argparse.ArgumentParser(description='Molecular autoencoder network')
#     parser.add_argument('--load_model', type=str, metavar='N', default="")
#     parser.add_argument('--epochs', type=int, metavar='N', default=EPOCHS,
#                         help='Number of epochs to run during training.')
#     parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT,
#                         help='Dimensionality of the latent representation.')
#     return parser.parse_args()


# def main():
#     # 0. load dataset
#     h5f = h5py.File('data/zinc_grammar_dataset.h5', 'r')
#     data = h5f['data'][:]
#     h5f.close()

#     # 1. split into train/test, we use test set to check reconstruction error and the % of
#     # samples from prior p(z) that are valid
#     XTE = data[0:5000]
#     XTR = data[5000:]

#     np.random.seed(1)
#     # 2. get any arguments and define save file, then create the VAE model
#     args = get_arguments()
#     print('L='  + str(args.latent_dim) + ' E=' + str(args.epochs))
#     model_save = 'results/zinc_vae_grammar_L' + str(args.latent_dim) + '_E' + str(args.epochs) + '_val.hdf5'
#     print(model_save)
#     model = MoleculeVAE()
#     print(args.load_model)

#     # 3. if this results file exists already load it
#     if os.path.isfile(args.load_model):
#         print('loading!')
#         model.load(rules, args.load_model, latent_rep_size = args.latent_dim, max_length=MAX_LEN)
#     else:
#         print('making new model')
#         model.create(rules, max_length=MAX_LEN, latent_rep_size = args.latent_dim)

#     # 4. only save best model found on a 10% validation set
#     checkpointer = ModelCheckpoint(filepath = model_save,
#                                    verbose = 1,
#                                    save_best_only = True)

#     reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
#                                   factor = 0.2,
#                                   patience = 3,
#                                   min_lr = 0.0001)
#     # 5. fit the vae
#     model.autoencoder.fit(
#         XTR,
#         XTR,
#         shuffle = True,
#         nb_epoch = args.epochs,
#         batch_size = BATCH,
#         callbacks = [checkpointer, reduce_lr],
#         validation_split = 0.1)

# if __name__ == '__main__':
#     main()




# train_zinc_pytorch.py
import argparse
import os
import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
import pdb
import zinc_grammar as G

from models.model_zinc import WrapperMoleculeVAE

rules = G.gram.split("\n")

MAX_LEN = 277
DIM = len(rules)
LATENT = 56
EPOCHS = 100
BATCH = 500
LR = 1e-3


def get_arguments():
    parser = argparse.ArgumentParser(description="Molecular autoencoder network")
    parser.add_argument("--load_model", type=str, metavar="N", default="")
    parser.add_argument(
        "--epochs",
        type=int,
        metavar="N",
        default=EPOCHS,
        help="Number of epochs to run during training.",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        metavar="N",
        default=LATENT,
        help="Dimensionality of the latent representation.",
    )
    return parser.parse_args()


def main():
    # 0. load dataset
    h5f = h5py.File("data/zinc_grammar_dataset.h5", "r")
    data = h5f["data"][:]
    h5f.close()

    # 1. split into train/test
    XTE = data[0:5000]
    XTR = data[5000:]

    np.random.seed(1)
    torch.manual_seed(1)

    args = get_arguments()
    print("L=" + str(args.latent_dim) + " E=" + str(args.epochs))
    model_save = (
        "results/zinc_vae_grammar_L"
        + str(args.latent_dim)
        + "_E"
        + str(args.epochs)
        + "_val.pt"
    )
    print(model_save)

    # 创建模型
    model_wrapper = WrapperMoleculeVAE()
    print(args.load_model)
    if os.path.isfile(args.load_model):
        print("loading!")
        model_wrapper.load(
            rules, args.load_model, latent_rep_size=args.latent_dim, max_length=MAX_LEN
        )
    else:
        print("making new model")
        model_wrapper.create(rules, max_length=MAX_LEN, latent_rep_size=args.latent_dim)

    # PyTorch训练准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_wrapper.model.to(device)

    XTR_t = torch.tensor(XTR, dtype=torch.float32)
    train_dataset = TensorDataset(XTR_t, XTR_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)

    # 按原来的做法，使用10%数据作为val
    val_size = int(0.1 * len(XTR_t))
    val_dataset = TensorDataset(XTR_t[:val_size], XTR_t[:val_size])
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False)

    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.2, patience=3, min_lr=1e-4
    )

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss_accum = 0.0
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            x_decoded_logits, z_mean, z_log_var = model(batch_x)
            loss, _, _ = model.vae_loss(batch_x, x_decoded_logits, z_mean, z_log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item() * batch_x.size(0)

        train_loss = train_loss_accum / len(train_loader.dataset)

        # validation
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                x_decoded_logits, z_mean, z_log_var = model(batch_x)
                loss, _, _ = model.vae_loss(
                    batch_x, x_decoded_logits, z_mean, z_log_var
                )
                val_loss_accum += loss.item() * batch_x.size(0)
        val_loss = val_loss_accum / len(val_loader.dataset)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{args.epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}"
        )

        # 与原Keras的ModelCheckpoint逻辑一致，当val_loss改善时保存模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("results", exist_ok=True)
            # 用torch.save保存state_dict到与原始同名的hdf5文件中
            # 虽然格式不同，但文件名和扩展名匹配
            torch.save(model.state_dict(), model_save)
            print("Saved best model")


if __name__ == "__main__":
    main()
