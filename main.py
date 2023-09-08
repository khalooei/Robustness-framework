'''
Developted by : Mohammad Khalooei (Khalooei@aut.ac.ir)

We use Pytorch lightning for training and inference phase
More information about it is located at plmodel.py file which focus on pytorch lightning class model.

Our main finction is `khalooei_app` which integrate all modules in our framework based on pytorch lightning.
'''

from plmodel import *

@hydra.main(version_base=None, config_path="configs", config_name="training_cifar10")
def khalooei_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    model = PLModel(cfg)

    s_dir_name = f"{cfg.global_params.dataset}-{cfg.global_params.architecture}-{cfg.training_params.type}_e{cfg.training_params.epoch}_note-{cfg.training_params.note}_"
    if cfg.training_params.type == "AT":
        s_dir_name+=f"{cfg.adversarial_training_params.name}_eps{cfg.adversarial_training_params.eps}"

    callbacks = [
        # save_top_k -> save the best model   >>>  best model is max in val_acc
        LearningRateMonitor(logging_interval="step"), 
        ModelCheckpoint(save_top_k=2, mode="max", monitor="clean_val_acc"),  # saved best model based on Maximize the validation accuracy
        CustomTimeCallback(),
    ]

    # configure logging on module level, redirect to file
    p = pathlib.Path(f'logs/{s_dir_name}')
    p.mkdir(parents=True, exist_ok=True)
    consolelogger = logging.getLogger("lightning.pytorch.core")
    # consolelogger.addHandler(logging.FileHandler(f"logs/{s_dir_name}/core.log"))
    
    s_experiment_starting_time = f"{time.strftime('%Y%m%d%H%M%S')}"
    trainer = pl.Trainer(max_epochs=cfg.training_params.epoch,
                        devices=cfg.global_params.devices,
                        num_nodes=1,
                        strategy="ddp",
                        callbacks=callbacks,
                        logger=[CSVLogger(save_dir=f'logs/{s_dir_name}/',version=s_experiment_starting_time),
                                TensorBoardLogger(f'logs/{s_dir_name}/',version=s_experiment_starting_time),
                                KhalooeiLoggingLogger(save_dir=f'logs/{s_dir_name}',version=s_experiment_starting_time),],
                        accelerator='gpu', #reproducibility,
                        # deterministic=True, #reproducibility,
                        inference_mode=False,
                        # auto_lr_find = True # to find better lr
                        )


    # tuning 
    # from pytorch_lightning.tuner.tuning import Tuner
    # tuner = Tuner(trainer)
    # # lr_finder= tuner.lr_find(model, datamodule=model.train_dataloader())

    trainer.fit(model)

    # results = trainer.tune(model=model, datamodule=model.train_dataloader()) # for tuning
    # results['lr_find'].plot(suggest=True)
    # trainer.save_checkpoint('model.ckpt') #save manually

    print('-----TEST ACC----------')
    # test_acc = trainer.test(ckpt_path='best')  #dataloaders=test_loader
    # best model is a model with best ones 

    # print('-----VAL ACC----------')
    # val_acc = trainer.test(dataloaders=model.val_loader,ckpt_path='best')  #dataloaders=test_loader
    # print(val_acc)
    # print('-----Train ACC----------')
    # train_acc = trainer.test(dataloaders=model.train_loader,ckpt_path='best')  #dataloaders=test_loader
    # print(train_acc)
    print('-----TEST ACC----------')
    model.test_dataloader()
    test_acc = trainer.test(dataloaders=model.test_loader,ckpt_path='best')  #dataloaders=test_loader
    print(test_acc)
    # every saved log is in lightning-Logs


    # print('==========================')
    # model.eval()
    # test_dataloader = model.test_dataloader()
    # acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
    # lst_acc = []
    # for batch in test_dataloader:
    #     x,y = batch
    #     with torch.inference_mode():
    #         logits = model(x)
    #     predicted = torch.argmax(logits, dim=1)
    #     lst_acc.append(acc(predicted, y))

    # test_acc = torch.Tensor(lst_acc).mean()
    # print(f'Test accuracy {test_acc:.4f} ({test_acc*100:.2f}%)')



if __name__ == "__main__":
    khalooei_app()