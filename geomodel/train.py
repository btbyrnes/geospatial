import os
import time
import torch

def epoch(model, loss_function, optimizer, train_dataloader, test_dataloader=None, model_save_path=None):

    epoch_time = time.time()

    epoch = 0
    train_loss = []
    test_loss = []

    if os.path.exists(model_save_path):

        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        train_loss = checkpoint["train_loss"]
        if "test_loss" in checkpoint.keys():
            test_loss = checkpoint["test_loss"]

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(train_dataloader):

        t = time.time()
        
        X = batch["context"]
        y = batch["target"]

        optimizer.zero_grad()
        pred = model(X)

        # print(pred.shape, y.shape,X.shape)
        loss = loss_function(pred,y)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if i % 10 == 0:
            l = loss.item()
            current = i
            print(f"loss:  {l:>5f} | time {time.time()-t:2f}(s) | Batch [{current:>5d}/{len(train_dataloader):>5d}]")

    train_loss.append(epoch_loss)

    epoch_time = (time.time() - epoch_time)

    print("epoch {epoch} done | loss {epoch_loss:5f} | time {epoch_time:2f}(s)".format(
        epoch=epoch, epoch_loss=epoch_loss, epoch_time=epoch_time))


    if test_dataloader:

        model.eval()
        eval_loss = 0

        for i, batch in enumerate(test_dataloader):
            
            X = batch["context"]
            y = batch["target"]
            
            pred = model(X)
            loss = loss_function(pred,y)
            eval_loss += loss.item()
        
        test_loss.append(eval_loss)

        print("epoch {epoch} | test loss {eval_loss:5f}".format(
            epoch=epoch, eval_loss=eval_loss))

    epoch += 1

    if model_save_path:
        
        checkpoint = {"epoch":epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss":train_loss,
            "test_loss":test_loss}
        
        torch.save(checkpoint, model_save_path)
    
    if test_dataloader:
        return train_loss,test_loss
    else:
        return train_loss
