
import torch
import math
from torch.utils.data import DataLoader

from load_data.vgg16 import vgg16_bn

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x



#########################################################################################
#   Extracting Features Class. 
#   This models is ImageNet preatrined model provided by Pytorch official
#########################################################################################
class ExtractFeature():
    def __init__(self, eval_embedder, pretrained, batch_size, device, test_num):
        """
        Loading pretrained embedder for evaluation.
        args : 
                eval_embedder : type of embedder for extracting features
                                inceptionV3 | vgg16_bn | vgg16
                batch_size    : number of batch size for input embedder
                device        : gpu number
                test_num      : number of features in generated and reference datas
        """
        super(ExtractFeature, self).__init__()
        self.eval_embedder = eval_embedder
        self.device = device
        self.test_num = test_num
        self.pretrained = pretrained

        self.load_Model(eval_embedder)
        
        if batch_size is None:
            self.batch_size = 32
        else:
            self.batch_size = batch_size
        self.eval()


    def eval(self):
        self.eval_embedder.eval()


    def load_Model(self, eval_embedder):
        try:
            if eval_embedder == 'vgg16bn':
                self.eval_embedder = vgg16_bn(pretrained=self.pretrained, progress=True)
            elif eval_embedder == 'inceptionV3':
                # To do
                raise SystemError
            else:
                raise Exception("Cannot load eval embedder ", eval_embedder)
        except Exception as e:
            print(e)
            raise SystemError
        self.eval_embedder = self.eval_embedder.to(self.device)
        

    def get_embeddings_from_loaders(self, dataloader):
        features_list = []
        total_instance = len(dataloader.dataset)
        num_batches = math.ceil(float(total_instance) / float(self.batch_size))
        data_iter = iter(dataloader)
        start_idx = 0

        with torch.no_grad():
            for _ in tqdm(range(0, num_batches)):
                
                try:
                    images = torch.tensor(next(data_iter)[0]).to(self.device)
                except StopIteration:
                    break

                features = self.eval_embedder(images).to(self.device)
                features = features.detach().cpu().numpy()
                features_list[start_idx:start_idx+ features.shape[0]] = features
                start_idx = start_idx + features.shape[0]
        return features_list


def get_features(dataset, args, device):
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    if isinstance(dataloader, DataLoader):
        extractor = ExtractFeature('vgg16bn', args.pretrained, args.batch_size, device, args.num_samples)
        feats = extractor.get_embeddings_from_loaders(dataloader)
        return feats

def get_featurs_from_real_and_fake(real_dataset, fake_datasets, args, device):
    real_feats = get_features(real_dataset, args, device)
    fake_feats = dict()
    for fake_dir, dataset in fake_datasets.items():
        fake_feats[fake_dir] = get_features(dataset)
    return real_feats, fake_feats
