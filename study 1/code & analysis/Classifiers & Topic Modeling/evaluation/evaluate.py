from sklearn.metrics import classification_report, confusion_matrix
import torch

def evaluate_model(model, dataloader, device):
    model.eval()
    true_labels = []
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['ids'].to(device)
            attention_masks = batch['mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(inputs, attention_mask=attention_masks)
            _, preds = torch.max(outputs, dim=1)

            true_labels.extend(labels.cpu().tolist())
            predictions.extend(preds.cpu().tolist())

    return true_labels, predictions

def compute_metrics(true_labels, predictions):
    report = classification_report(true_labels, predictions, target_names=["non-moral", "moral"], output_dict=True)
    conf_matrix = confusion_matrix(true_labels, predictions)

    return report, conf_matrix
