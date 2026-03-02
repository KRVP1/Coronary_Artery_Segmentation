import tqdm
import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output
from cldice import soft_dice


def save_checkpoint(path, model, optimizer, train_epoch_losses, val_epoch_losses, val_dice_scores):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_epoch_losses': train_epoch_losses,
        'val_epoch_losses': val_epoch_losses,
        'val_dice_scores': val_dice_scores,
    }, path)


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_epoch_losses = checkpoint.get('train_epoch_losses', [])
    val_epoch_losses = checkpoint.get('val_epoch_losses', [])
    val_dice_scores = checkpoint.get('val_dice_scores', [])

    return model, optimizer, train_epoch_losses, val_epoch_losses, val_dice_scores


def plot_train_history(train_epoch_losses, val_epoch_losses, val_dice_scores):
    clear_output()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
    epochs = range(1, len(train_epoch_losses) + 1)
    ax1.plot(epochs, train_epoch_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_epoch_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График Dice Score
    ax2.plot(epochs, val_dice_scores, 'g-', label='Val Dice Score', linewidth=2)
    ax2.set_title('Validation Dice Score', fontsize=14)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Dice Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    
def plot_image_2d(img, label, pred):
    slice_idx = img.shape[2] // 2

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Маска (сосуды)")
    plt.imshow(img[:, :, slice_idx], cmap='gray', alpha=0.8)
    plt.imshow(label[:, :, slice_idx], cmap='jet', alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.title("Предсказанная маска (сосуды)")
    plt.imshow(img[:, :, slice_idx], cmap='gray', alpha=0.8)
    plt.imshow(pred[:, :, slice_idx], cmap='jet', alpha=0.5)

    plt.show()

    
def plot_sample(n_samples, model, val_loader, device):
    for _, batch in zip(range(n_samples), iter(val_loader)):
        model.eval()
        imgs = []
        labels = []
        with torch.no_grad():
            img, label = batch['image'], batch["label"]
            img = img.to(device)
            pred = model(img)

            pred = torch.argmax(pred[0], dim=0).cpu().numpy()
            label = label[0].cpu().squeeze().numpy()
            img = img[0].cpu().squeeze().numpy()
            
            plot_image_2d(img, label, pred)


def validation(model, loss_function, val_loader, device):
    with torch.no_grad():
        epoch_loss = 0.0
        dice_loss = 0.0

        model.eval()
        for batch_data in tqdm.tqdm(val_loader):
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
   
            outputs = model(inputs)
            outputs = outputs[:, -1, :, :, :]

            epoch_loss += loss_function(labels, outputs).item()
            dice_loss += soft_dice(labels, outputs).item()

    metric = 1 - dice_loss / len(val_loader)
    return epoch_loss / len(val_loader), metric


def train(model, loss_function, optimizer, train_loader, val_loader, epochs, plot_loss, device):
    train_epoch_losses = []
    val_epoch_losses = []
    val_dice_scores = []
    for epoch in range(epochs):
        step = 0
        epoch_loss = 0.0
        dice_loss = 0.0
        for batch_data in tqdm.tqdm(train_loader):
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            
            optimizer.zero_grad()
            # with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            # print(inputs.shape)
            outputs = model(inputs)
            # print(outputs.shape)
            # outputs = F.softmax(outputs)
            outputs = outputs[:, -1, :, :, :]
            loss = loss_function(labels, outputs)
                
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            # dice_loss += soft_dice(labels.detach(), outputs.detach()).item()
            
            # if step % 20 == 0:  # Чаще выводим, так как batch_size=1
            #     print(f"  Шаг {step}, Loss: {epoch_loss / step:.4f}")
            #     print(f"  Шаг {step}, Dice Loss: {dice_loss / step:.4f}")
        train_epoch_losses.append(epoch_loss / len(train_loader))

        val_epoch_loss, val_dice = validation(model, loss_function, val_loader, device)
        if val_dice > max(val_dice_scores):
            torch.save(model.state_dict(), 'best_model_weights.pth')
            print(f'New bast model with Dice: {val_dice}:.4f save')

        val_epoch_losses.append(val_epoch_loss)
        val_dice_scores.append(val_dice)

        if plot_loss:
            plot_train_history(train_epoch_losses, val_epoch_losses, val_dice_scores)


    # torch.save(model.state_dict(), 'model_weights.pth')