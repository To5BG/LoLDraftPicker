import torch


class EmbeddingTrainer:
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device

    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        for batch in dataloader:
            features, targets = batch
            features = features.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            embeddings = self.model(features)
            loss = criterion(embeddings, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                features, targets = batch
                features = features.to(self.device)
                targets = targets.to(self.device)

                embeddings = self.model(features)
                loss = criterion(embeddings, targets)
                total_loss += loss.item()
        return total_loss / len(dataloader)


class PickerTrainer:
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device

    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        for batch in dataloader:
            enemy_adc, enemy_support, my_support, my_team_cc, enemy_team_cc, target = (
                batch
            )
            enemy_adc = enemy_adc.to(self.device)
            enemy_support = enemy_support.to(self.device)
            my_support = my_support.to(self.device)
            my_team_cc = my_team_cc.to(self.device)
            enemy_team_cc = enemy_team_cc.to(self.device)
            target = target.to(self.device)

            optimizer.zero_grad()
            predicted_emb = self.model(
                enemy_adc, enemy_support, my_support, my_team_cc, enemy_team_cc
            )
            loss = criterion(predicted_emb, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                (
                    enemy_adc,
                    enemy_support,
                    my_support,
                    my_team_cc,
                    enemy_team_cc,
                    target,
                ) = batch
                enemy_adc = enemy_adc.to(self.device)
                enemy_support = enemy_support.to(self.device)
                my_support = my_support.to(self.device)
                my_team_cc = my_team_cc.to(self.device)
                enemy_team_cc = enemy_team_cc.to(self.device)
                target = target.to(self.device)

                predicted_emb = self.model(
                    enemy_adc, enemy_support, my_support, my_team_cc, enemy_team_cc
                )
                loss = criterion(predicted_emb, target)
                total_loss += loss.item()
        return total_loss / len(dataloader)
