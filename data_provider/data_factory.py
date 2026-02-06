from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_Solar, Dataset_PEMS
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np
import time

_DATA_CACHE = {}

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS
}


def get_cached_data(root_path, data_path, dataset_type):
    """Get data from cache; load and cache if not present."""
    cache_key = os.path.join(root_path, data_path)
    
    if cache_key in _DATA_CACHE:
        print(f"Using cached data for {data_path}")
        return _DATA_CACHE[cache_key]['data']
    
    if dataset_type == 'Solar':
        df_raw = []
        with open(cache_key, "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        data = pd.DataFrame(df_raw)
    elif dataset_type == 'PEMS':
        data_file = cache_key
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]
    else:
        data = pd.read_csv(cache_key)
    
    _DATA_CACHE[cache_key] = {'data': data}
    print(f"Cached data for {data_path}")
    
    return data


def preload_all_data(args):
    """Preload train/val/test datasets into memory."""
    print("Preloading all datasets...")
    start_time = time.time()
    
    get_cached_data(args.root_path, args.data_path, args.data)
    
    train_data, train_loader = data_provider(args, 'train')
    val_data, val_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')
    
    print(f"All datasets preloaded in {time.time() - start_time:.2f} seconds")
    
    return train_data, train_loader, val_data, val_loader, test_data, test_loader


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    args._cached_data = get_cached_data(args.root_path, args.data_path, args.data)

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        cycle=args.cycle
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader