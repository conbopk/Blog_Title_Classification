from Training import *
from sklearn.metrics import confusion_matrix, classification_report


#Load model
model.load_state_dict(torch.load('best-model.pt'))

#Validation
val_loss, val_acc = evaluate(model, val_loader, criterion)
print(f'Validation Loss: {val_loss:.3f} | Validation Acc: {val_acc:.2f}')

model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for batch in val_loader:
        text, labels = batch
        text = text.to(device)

        predictions = model(text)
        predictions_classes = predictions.argmax(1)

        y_pred.extend(predictions_classes.cpu().numpy())
        y_true.extend(labels.numpy())

#Confusion matrix
conf_mat = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel('Pred Label')
plt.ylabel('True Label')
plt.show()

print(classification_report(y_true, y_pred))


