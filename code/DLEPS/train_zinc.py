# # train_zinc_pytorch.py
# import argparse
# import os
# import h5py
# import numpy as np
# import torch
# from torch.utils.data import TensorDataset, DataLoader
# from torch.optim import Adam
# import torch.optim.lr_scheduler as lr_scheduler
# import pdb
# import zinc_grammar as G

# from models.model_zinc import WrapperMoleculeVAE

# rules = G.gram.split("\n")

# MAX_LEN = 277
# DIM = len(rules)
# LATENT = 56
# EPOCHS = 100
# BATCH = 500
# LR = 1e-3


# def get_arguments():
#     parser = argparse.ArgumentParser(description="Molecular autoencoder network")
#     parser.add_argument("--load_model", type=str, metavar="N", default="")
#     parser.add_argument(
#         "--epochs",
#         type=int,
#         metavar="N",
#         default=EPOCHS,
#         help="Number of epochs to run during training.",
#     )
#     parser.add_argument(
#         "--latent_dim",
#         type=int,
#         metavar="N",
#         default=LATENT,
#         help="Dimensionality of the latent representation.",
#     )
#     return parser.parse_args()


# def main():
#     # 0. load dataset
#     h5f = h5py.File("data/zinc_grammar_dataset.h5", "r")
#     data = h5f["data"][:]
#     h5f.close()

#     # 1. split into train/test
#     XTE = data[0:5000]
#     XTR = data[5000:]

#     np.random.seed(1)
#     torch.manual_seed(1)

#     args = get_arguments()
#     print("L=" + str(args.latent_dim) + " E=" + str(args.epochs))
#     model_save = (
#         "results/zinc_vae_grammar_L"
#         + str(args.latent_dim)
#         + "_E"
#         + str(args.epochs)
#         + "_val.pt"
#     )
#     print(model_save)

#     # 创建模型
#     model_wrapper = WrapperMoleculeVAE()
#     print(args.load_model)
#     if os.path.isfile(args.load_model):
#         print("loading!")
#         model_wrapper.load(
#             rules, args.load_model, latent_rep_size=args.latent_dim, max_length=MAX_LEN
#         )
#     else:
#         print("making new model")
#         model_wrapper.create(rules, max_length=MAX_LEN, latent_rep_size=args.latent_dim)

#     # PyTorch训练准备
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model_wrapper.model.to(device)

#     XTR_t = torch.tensor(XTR, dtype=torch.float32)
#     train_dataset = TensorDataset(XTR_t, XTR_t)
#     train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)

#     # 按原来的做法，使用10%数据作为val
#     val_size = int(0.1 * len(XTR_t))
#     val_dataset = TensorDataset(XTR_t[:val_size], XTR_t[:val_size])
#     val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False)

#     optimizer = Adam(model.parameters(), lr=LR)
#     scheduler = lr_scheduler.ReduceLROnPlateau(
#         optimizer, factor=0.2, patience=3, min_lr=1e-4
#     )

#     best_val_loss = float("inf")

#     for epoch in range(args.epochs):
#         model.train()
#         train_loss_accum = 0.0
#         for batch_x, _ in train_loader:
#             batch_x = batch_x.to(device)
#             x_decoded_logits, z_mean, z_log_var = model(batch_x)
#             loss, _, _ = model.vae_loss(batch_x, x_decoded_logits, z_mean, z_log_var)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             train_loss_accum += loss.item() * batch_x.size(0)

#         train_loss = train_loss_accum / len(train_loader.dataset)

#         # validation
#         model.eval()
#         val_loss_accum = 0.0
#         with torch.no_grad():
#             for batch_x, _ in val_loader:
#                 batch_x = batch_x.to(device)
#                 x_decoded_logits, z_mean, z_log_var = model(batch_x)
#                 loss, _, _ = model.vae_loss(
#                     batch_x, x_decoded_logits, z_mean, z_log_var
#                 )
#                 val_loss_accum += loss.item() * batch_x.size(0)
#         val_loss = val_loss_accum / len(val_loader.dataset)

#         scheduler.step(val_loss)

#         print(
#             f"Epoch {epoch+1}/{args.epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}"
#         )

#         # 与原Keras的ModelCheckpoint逻辑一致，当val_loss改善时保存模型
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             os.makedirs("results", exist_ok=True)
#             # 用torch.save保存state_dict到与原始同名的hdf5文件中
#             # 虽然格式不同，但文件名和扩展名匹配
#             torch.save(model.state_dict(), model_save)
#             print("Saved best model")


# if __name__ == "__main__":
#     main()







import argparse
import os
import h5py
import numpy as np
import logging
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.utils import Sequence
from models.model_zinc import MoleculeVAE
import zinc_grammar as G

# Grammar rules and settings
rules = G.gram.split('\n')
MAX_LEN = 277
DIM = len(rules)
LATENT = 56
EPOCHS = 10000
BATCH = 2000

# Configure logging
log_file = "training_log.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(log_file),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

# Custom callback to log epoch details
class LogEpochDetails(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        logger.info(f"Epoch {epoch + 1}/{self.params['epochs']} started")

    def on_epoch_end(self, epoch, logs=None):
        logger.info(f"Epoch {epoch + 1} ended: Loss={logs['loss']:.4f}, Val_Loss={logs.get('val_loss', 'N/A'):.4f}")

    def on_train_batch_end(self, batch, logs=None):
        logger.debug(f"Batch {batch} completed: Loss={logs['loss']:.4f}")

# Data Generator
class ZincDataGenerator(Sequence):
    def __init__(self, h5_file, batch_size, max_len, charset_length, is_train=True, shuffle=True):
        self.h5_file = h5_file
        self.batch_size = batch_size
        self.max_len = max_len
        self.charset_length = charset_length
        self.is_train = is_train
        self.shuffle = shuffle
        self.indexes = np.arange(len(self))
        self.on_epoch_end()

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            length = len(f['data']) // self.batch_size
            logger.info(f"Data length: {length} batches")
            return length

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size

        with h5py.File(self.h5_file, 'r') as f:
            data_batch = f['data'][start_idx:end_idx]

        logger.debug(f"Generated batch {index}: Start={start_idx}, End={end_idx}")
        return data_batch, data_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
            logger.info("Shuffled data indices for the next epoch")

# Argument parser
def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('--load_model', type=str, metavar='N', default="")
    parser.add_argument('--epochs', type=int, metavar='N', default=EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT,
                        help='Dimensionality of the latent representation.')
    return parser.parse_args()

# Main training loop
def main():
    h5_file = 'zinc_grammar_dataset.h5'
    max_len = MAX_LEN
    charset_length = DIM
    batch_size = BATCH

    # Create data generators for training and validation
    train_generator = ZincDataGenerator(h5_file, batch_size, max_len, charset_length, is_train=True)

    np.random.seed(1)

    # Get command-line arguments and initialize model
    args = get_arguments()
    logger.info(f"Training with latent dimension: {args.latent_dim}, epochs: {args.epochs}")
    model_save = f'results/zinc_vae_grammar_L{args.latent_dim}_E{args.epochs}_val.hdf5'
    logger.info(f"Model save path: {model_save}")

    model = MoleculeVAE()

    # Load pre-trained model or create a new one
    if os.path.isfile(args.load_model):
        logger.info(f"Loading model from {args.load_model}")
        model.load(rules, args.load_model, latent_rep_size=args.latent_dim, max_length=MAX_LEN)
    else:
        logger.info("Creating a new model")
        model.create(rules, max_length=MAX_LEN, latent_rep_size=args.latent_dim)

    # Callbacks: Save best model and reduce learning rate on plateau
    checkpointer = ModelCheckpoint(filepath=model_save, verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
    log_epoch_details = LogEpochDetails()

    logger.info("Starting training...")
    model.autoencoder.fit(
        train_generator,
        epochs=args.epochs,
        callbacks=[checkpointer, reduce_lr, log_epoch_details],
        verbose=1
    )

    logger.info("Training complete. Model saved to " + model_save)

if __name__ == '__main__':
    main()