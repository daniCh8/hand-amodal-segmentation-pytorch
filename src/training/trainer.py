import torch
from models.model import ModelWrapper
from tqdm import tqdm
import utils
import vis
from utils import push_images, log_dict, Experiment
import time
import numpy as np
import os.path as op
import itertools
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class Trainer():
    def __init__(self, train_loader, val_loaders, unpaired_images_loader, args):
        super().__init__()
        self.args = args
        self.train_loader = train_loader
        self.val_loaders = val_loaders
        self.which_datasets = args.datasets
        self.pre_epoch = args.pre
        self.init_epoch = args.init
        self.final_epoch = args.final
        self.with_additional = False
        if args.final_splitted > 0 or args.init_splitted > 0:
            self.with_additional = True
            self.final_additional = args.final_splitted
            self.init_additional = args.init_splitted
        self.name_comet = args.comet_name

        self.unp_img_loader = unpaired_images_loader 
        
        if args.use_comet:
            import comet_ml
        
            # Create an experiment with your api key
            self.comet_experiment = comet_ml.Experiment(
                api_key="aASgA0tUpW7Y4FxbuD0egr84z",
                project_name="hand-segmentation-mp",
                workspace="pierremotard",
            )

            self.comet_experiment.add_tag(self.name_comet)

        self.model = ModelWrapper(segmentation_model_g1=args.segmentation_model_g1,
                                  segmentation_model_g2=args.segmentation_model_g2,
                                  lr=args.lr,
                                  segm_loss_weight=args.segm_loss_w,
                                  main_loss_weight=args.main_loss_w,
                                  saved_weights_path_g1=args.saved_weights_path_g1,
                                  discriminator_w=args.discriminator_w,
                                  saved_weights_path_d=args.saved_weights_path_D,
                                  old_weights=args.use_old_weights,
                                  main_loss=args.loss,
                                  use_weights=args.use_weights)
        
        if self.args.load_ckpt != '':
            self.load_ckpt(self.args.load_ckpt)

        self.model.cuda()

        self.current_epoch = 0
        self.global_step = 0

        # experiment key
        self.experiment = Experiment(args)
        self.args.exp_key = self.experiment.get_key()[:9]

        self.highlight_1 = '\033[92m+++++++ '
        self.highlight_2 = ' +++++++\033[0m'
        
        print('Experiment Key: %s' % (self.args.exp_key))

        # folder containing info of this experiment
        if args.save_path == "":
            self.exp_path = op.join('logs', self.args.exp_key)
            self.save_path = op.join(self.exp_path, 'latest.pt')  # model dumping
        else:
            self.exp_path = args.save_path
            self.save_path = op.join(self.exp_path, 'latest.pt')
        utils.mkdir_p(self.exp_path)

    def pretrain_epoch(self):
        assert self.train_loader is not None
        torch.cuda.empty_cache()
        model = self.model
        train_loader = self.train_loader
        model.train()
        running_loss_disc = 0.0
        running_loss = 0.0
        iterations = len(train_loader) 
        pbar = tqdm(enumerate(train_loader, 0), total=iterations)
        epoch = self.current_epoch

        # training loop for epoch
        for i, batch in pbar:
            # push things to CUDA
            inputs, targets, meta_info = utils.things2dev(batch, 'cuda')
            #inputs, targets, meta_info = batch
            loss_dict = model(inputs, targets, meta_info, 'train', None, 'pre')
            total_loss = loss_dict['loss_segm'] 
            if loss_dict['loss_disc']:
                total_loss_disc = loss_dict['loss_disc']
                running_loss_disc += total_loss_disc.item()

            if self.args.use_comet:
                self.comet_experiment.log_metrics(dic=loss_dict, epoch=epoch)
            
            running_loss = running_loss + total_loss.item()
            
            self.global_step += 1

        print("{}Epoch ({}): Pre-Training (Training G1 and Discriminator) --> Loss: {:.5f}; Avg. Loss: {:.5f}; Disc Loss {:.5f}, Avg Disc Loss {:.5f}; {}".format(self.highlight_1, epoch+1, running_loss, running_loss/iterations, running_loss_disc, running_loss_disc/iterations, self.highlight_2))

    def unpaired_epoch(self):
        assert self.train_loader is not None
        torch.cuda.empty_cache()
        model = self.model
        epoch = self.current_epoch

        model.train()
        running_loss = 0.0

        semi_loader = self.unp_img_loader
        semi_iterations = len(semi_loader)
        semi_iterator = iter(semi_loader)

        # training loop for epoch
        for i in tqdm(range(semi_iterations)):
            # push things to CUDA
            try:
                inputs_i, _ = utils.things2dev(next(semi_iterator), 'cuda')
                inputs = utils.things2dev(inputs_i, 'cuda')
                #inputs, targets, meta_info = batch
                loss_dict = model(None, None, None, 'train', inputs, 'init')
                total_loss = loss_dict['loss_segm'] 

            except Exception as e:
                print("Exception encountered and passed: {}".format(e))
                pass

            if self.args.use_comet:
                self.comet_experiment.log_metrics(dic=loss_dict, epoch=epoch)
            
            running_loss = running_loss + total_loss.item()
            self.global_step += 1

        print("{}Epoch ({}): Init Training (Training G2 with generated samples from G1 and Unpaired Images) --> Loss: {:.5f}; Avg. Loss: {:.5f}; {}".format(self.highlight_1, epoch+1, running_loss, running_loss/semi_iterations, self.highlight_2))

    def paired_epoch(self):
        assert self.train_loader is not None
        torch.cuda.empty_cache()
        model = self.model
        train_loader = self.train_loader
        epoch = self.current_epoch

        model.train()
        running_loss = 0.0

        iterations = len(train_loader) 
        pbar = tqdm(enumerate(train_loader, 0), total=iterations)

        # training loop for epoch
        for i, batch in pbar:
            # push things to CUDA
            inputs, targets, meta_info = utils.things2dev(batch, 'cuda')
            loss_dict = model(inputs, targets, meta_info, 'train', None, 'tr')
            total_loss = loss_dict['loss_segm'] 

            if self.args.use_comet:
                self.comet_experiment.log_metrics(dic=loss_dict, epoch=epoch)

            running_loss = running_loss + total_loss.item()
            self.global_step += 1

        print("{}Epoch ({}): Final Training (Training G2 on the paired dataset, with the adversarial loss) --> Loss: {:.5f}; Avg. Loss: {:.5f};{}".format(self.highlight_1, epoch+1, running_loss, running_loss/iterations, self.highlight_2))

    def eval_epoch(self, val_loader_dict, phase):
        # evaluate on a data loader
        assert isinstance(val_loader_dict, dict)
        model = self.model
        val_loader = val_loader_dict['loader']
        postfix = val_loader_dict['postfix']

        pbar = tqdm(enumerate(val_loader, 0), total=len(val_loader))
        out_list = []
        model.eval()
        with torch.no_grad():
            for i, batch in pbar:
                pbar.set_description('Validation epoch on {}'.format(postfix))
                inputs, targets, meta_info = utils.things2dev(
                    batch, 'cuda')
                out = model(inputs, targets, meta_info, 'val', None, phase)
                out_list.append(out)

        # aggregate outputs from each batch
        out_dict = utils.ld2dl(utils.things2dev(out_list, 'cpu'))

        miou = np.nanmean(np.concatenate(out_dict['ious'], axis=0))
        print('{}miou_{}: {:.5f}{}'.format(self.highlight_1, postfix, miou, self.highlight_2))
        metric_dict = {'miou_{}'.format(postfix): miou}
        return metric_dict

    def load_ckpt(self, ckpt_path):
        sd = torch.load(ckpt_path)
        print(self.model.load_state_dict(sd))

    def test_epoch(self, val_loader):
        assert val_loader is not None
        model = self.model

        pbar = tqdm(enumerate(val_loader, 0), total=len(val_loader))
        out_list = []
        model.eval()
        with torch.no_grad():
            for i, batch in pbar:
                inputs, meta_info = utils.things2dev(
                    batch, 'cuda')
                # forward without providing segmentation
                out = model.forward_test(inputs, meta_info)
                out_list.append(out)

        # aggregate segmentation predictions
        out_dict = utils.ld2dl(utils.things2dev(out_list, 'cpu'))
        segm_l = torch.cat(out_dict['segm_l'], dim=0)
        segm_r = torch.cat(out_dict['segm_r'], dim=0)
        im_path = sum(out_dict['im_path'], [])
        return (segm_l, segm_r, im_path)

    def test(self, val_loader):
        # evaluate a model in the test set. See test.py
        # package the prediction into test.tar.gz, which is used for submission
        segm_l, segm_r, im_path = self.test_epoch(val_loader)
        out_test_path = op.join(self.exp_path, 'test.lmdb')
        tar_path = out_test_path + '.tar.gz'

        utils.package_lmdb(out_test_path, segm_l, segm_r, im_path)
        im_path = [imp + '.jpg' for imp in im_path]
        torch.save(im_path, out_test_path + '/im_path.pt')
        print('Done writing test to: %s' % (out_test_path))

        utils.make_tarfile(tar_path, out_test_path)
        print('Done zipping test to: %s' % (tar_path))

    def visualize_batches(self, batch, postfix, num_examples, no_tqdm=True):
        # visualize a given batch

        model = self.model
        im_list = []
        model.eval()

        tic = time.time()
        with torch.no_grad():
            inputs, targets, meta_info = utils.things2dev(
                    batch, 'cuda')

            # get objects for visualizaton
            vis_dict = model(inputs, targets, meta_info, None, None, 'vis')

            # visualization plots
            curr_im_list = vis.visualize_all(
                    vis_dict, num_examples,
                    postfix=postfix, no_tqdm=no_tqdm)

            # push images to logger
            push_images(
                    self.experiment, curr_im_list, self.global_step)
            im_list += curr_im_list

        print('Done rendering (%.1fs)' % (time.time() - tic))
        return im_list

    def save_model(self):
        sd = self.model.state_dict()
        sd = utils.things2dev(sd, 'cpu')
        torch.save(sd, self.save_path)
        print('Saved model to: %s' % (self.save_path))
    
    def train(self):
        final_string = "[1/2]" if self.with_additional else ""
        self.train_helper(self.final_epoch, self.pre_epoch, self.init_epoch, final_string)
        if self.with_additional:
            self.train_helper(self.final_additional, 0, self.init_additional, "[2/2]")

    def train_helper(self, final, pre, init, out):
        for i in range(pre):
            self.pretrain_epoch()
            self.eval_boilerplate("pre")
        for i in range(init):
            self.unpaired_epoch()
            self.eval_boilerplate(None)
        for i in range(final):
            self.paired_epoch()
            self.eval_boilerplate("tr")

            self.save_model()
            
        self.save_model()
        print('Finished training. {}'.format(out))
    
    def eval_boilerplate(self, phase):
        self.current_epoch += 1

        if self.current_epoch % self.args.eval_every_epoch == 0:
            # evaluate on a list of loaders
            for loader in self.val_loaders:
                # metric performance on each loader
                metric_dict = self.eval_epoch(loader, phase)

                if self.args.use_comet:
                    # push metrics to logger
                    self.comet_experiment.log_metrics(dic=metric_dict, epoch=self.current_epoch)