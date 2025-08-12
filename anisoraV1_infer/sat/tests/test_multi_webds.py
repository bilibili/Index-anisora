from sat.data_utils import make_loaders
from sat import get_args
from sat.data_utils import MetaDistributedWebDataset

args = get_args(['--batch-size', '10', '--train-data', '/mnt/shared/img_datasets/clay1b_dataset/coyo_700m_hqaes2/part-00000/00000{0..9}.tar', '/mnt/shared/img_datasets/laion_aes_3m/part-00001/10000{0..4}.tar', '--train-data-weights', '2', '1', '--iterable-dataset', '--split','1', '--num-workers', '1', '--batch-from-same-dataset'])

def process_fn(src):
    for x in src:
        if x["image_phash"] is not None:
            yield 1
        else:
            yield 0

def create_func(path, args):
    return MetaDistributedWebDataset(path, process_fn, seed=0, meta_names=['image_phash'])

train_loader, val_loader, test_loader = make_loaders(args,create_dataset_function=create_func)

a, b, c = 0, 0, 0
for i, x in enumerate(train_loader):
    if i >= 100:
        break
    print(x)
    # if all x[i] is 0
    if all([y == 0 for y in x]):
        a += 1
    elif all([y == 1 for y in x]):
        b += 1
    else:
        c += 1

print(a, b, c)



