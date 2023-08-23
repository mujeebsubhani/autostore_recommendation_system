
from src.utils.utils import *

class ProductTrainDataset(Dataset):
    def __init__(self, multiple_order_df, unique_product_ids):
        self.users, self.items, self.labels = self.get_dataset(multiple_order_df, unique_product_ids)

    def __len__(self):
        return len(self.users)
  
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, multiple_order_df, unique_product_ids):
        users, items, labels = [], [], []
        user_item_set = set(zip(multiple_order_df['user_id'], multiple_order_df['product_id']))

        num_negatives = 12
        for u, i in user_item_set:
            users.append(u)
            items.append(i)
            labels.append(1)
            for _ in range(num_negatives):
                negative_item = np.random.choice(unique_product_ids)
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(unique_product_ids)
                users.append(u)
                items.append(negative_item)
                labels.append(0)

        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)


class NeuralCollaborativeFiltering(pl.LightningModule):
    def __init__(self, num_users, num_items, product_ratings, all_product_ids):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=8)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=8)
        self.fc1 = nn.Linear(in_features=16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=16)
        self.fc4 = nn.Linear(in_features=16, out_features=8)
        self.output = nn.Linear(in_features=8, out_features=1)
        self.product_ratings = product_ratings
        self.all_product_ids = all_product_ids
        
    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        vector = torch.cat([user_embedded, item_embedded], dim=-1)
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))
        vector = nn.ReLU()(self.fc3(vector))  
        vector = nn.ReLU()(self.fc4(vector)) 
        pred = nn.Sigmoid()(self.output(vector))
        return pred
    
    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())

        predicted_labels_binary = (predicted_labels >= 0.5).float()
        accuracy = (predicted_labels_binary == labels.view(-1, 1).float()).sum().item() / labels.size(0)

        tqdm_dict = {'train_loss': loss, 'train_accuracy': accuracy}
        self.log_dict(tqdm_dict, on_epoch=True, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(ProductTrainDataset(self.product_ratings, self.all_product_ids),
                          batch_size=512, num_workers=4)
    
