import training.trainer as tr
import datasets


def run_exp(args, trainer=None):
    train_loader = None
    # load test set
    loader = datasets.fetch_dataloader('test', 'test', args)
    unpaired_images_loader = datasets.fetch_dataloader('test', 'test', args)

    # unpaired segmentation for training
    # unpaired_segm_loader = datasets.fetch_dataloader('test', None, args)
    # initialize network with boilerplate objects
    if trainer == None:
        trainer = tr.Trainer(train_loader, [], unpaired_images_loader, args)

    # evaluate on the test set
    trainer.test(loader)


if __name__ == "__main__":
    from config import args
    run_exp(args)
