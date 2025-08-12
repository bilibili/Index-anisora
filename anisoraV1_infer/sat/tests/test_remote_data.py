from sat.data_utils import make_loaders
from sat import get_args
from sat.data_utils import SimpleDistributedWebDataset

import time 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--world_size', type=int, default=1)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--master_addr', type=str, default='')
parser.add_argument('--master_port', type=int, default=7878)
parser.add_argument('--batch-size', type=int, default=32)
args = parser.parse_args()

args = get_args(['--batch-size', str(args.batch_size), 
    # '--train-data', 'rclone://r2train:cc3m-cc12m-sbu/part-00001/1000{00..25}.tar', 
    '--train-data', 'boto3://cc3m-cc12m-sbu/part-00001/1000{00..25}.tar', 
    '--iterable-dataset', 
    '--split','1', 
    '--num-workers', '1',
    '--mode', 'inference',
    '--prefetch-factor', '4'
    ])

def process_fn(src):
    for x in src:
        yield x

def create_func(path, args):
    return SimpleDistributedWebDataset(path, process_fn, seed=0)

train_loader, val_loader, test_loader = make_loaders(args, create_dataset_function=create_func)

for i, x in enumerate(train_loader):
    if i==0:
        start_time = time.time()
    time.sleep(1)
    if i > 50:
        # save x['jpg'] to disk
        with open('test.jpg', 'wb') as f:
            f.write(x['jpg'][0])
        break

print('Time taken: ', time.time() - start_time)
print('Avarage time per image: ', (time.time() - start_time)/args.batch_size/50)



