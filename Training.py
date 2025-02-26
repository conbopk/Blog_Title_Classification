from GRU import *
import torch
from tqdm import tqdm


def train(model,  iterator, optimizer, criterion):
    model.train()

    epoch_loss = 0
    epoch_acc = 0

    for batch in tqdm(iterator, desc="Training", leave=True, colour='green'):
        optimizer.zero_grad()

        text, labels = batch
        text = text.to(device)
        labels = labels.to(device)

        predictions = model(text)

        loss = criterion(predictions, labels)

        acc = (predictions.argmax(1) == labels).float().mean()

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for batch in iterator:
            text, labels = batch
            text = text.to(device)
            labels = labels.to(device)

            predictions = model(text)

            loss = criterion(predictions, labels)
            acc = (predictions.argmax(1) == labels).float().mean()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


#Training
N_EPOCHS = 20
best_val_loss = float('inf')

if __name__=='__main__':
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model=model, iterator=train_loader, optimizer=optimizer, criterion=criterion)
        val_loss, val_acc = evaluate(model=model, iterator=val_loader, criterion=criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best-model.pt')

        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}')
        print(f'\tVal Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}')




