from hyperopt import hp
import socket


def get_space(model):
    combined_space = dict()
    combined_best_params = dict()
    # -------- BASIC ----------------
    base_space = {
        'dropout': hp.choice('dropout', [0.1, 0.2]),
        'nhid_ffnn': hp.choice('nhid_ffnn', [256]),
        'lr': hp.choice('lr', [0.0001, 0.00005])
    }
    combined_space.update(base_space)

    current_best_params = {
        "nhid_ffnn": 1,
        "dropout": 1,
        "lr": 0
    }
    combined_best_params.update(current_best_params)

    if model == 'BILSTM':
        lstm_space = {
            'nhid_lstm': hp.choice('nhid_lstm', [128, 256, 400, 512]),
            'nlayers_lstmEncoders': hp.choice('nlayers_lstmEncoders', [1, 2])
        }
        combined_space.update(lstm_space)

        current_best_params = {
            "nhid_lstm": 3,
            "nlayers_lstmEncoders": 1
        }
        combined_best_params.update(current_best_params)

    if model == 'Transformer':
        trans_space = {
            'd_ffw': hp.choice('d_ffw', [512, 1024]),
            'nhead': hp.choice('nhead', [4, 5, 6]),
            'nlayers_attnEncoders': hp.choice('nlayers_attnEncoders', [4, 5, 6])
        }
        combined_space.update(trans_space)

        current_best_params = {
            "nhead": 2,
            "nlayers_attnEncoders": 0,
            "d_ffw": 0
        }
        combined_best_params.update(current_best_params)

    if model == 'CenteredLSTM':
        trans_space = {
            'nhid_lstm': hp.choice('nhid_lstm', [128, 256, 400, 512]),
            'nlayers_lstmEncoders': hp.choice('nlayers_lstmEncoders', [1, 2])
        }
        combined_space.update(trans_space)

        current_best_params = {
            "nhid_lstm": 3,
            "nlayers_lstmEncoders": 1
        }
        combined_best_params.update(current_best_params)

    if model == 'AttnLSTM':
        trans_space = {
            'nhid_lstm': hp.choice('nhid_lstm', [128, 256, 400, 512]),
            'nlayers_lstmEncoders': hp.choice('nlayers_lstmEncoders', [1, 2])
        }
        combined_space.update(trans_space)

        current_best_params = {
            "nhid_lstm": 3,
            "nlayers_lstmEncoders": 1
        }
        combined_best_params.update(current_best_params)

    if model == 'PBR':
        trans_space = {
            'd_ffw': hp.choice('d_ffw', [512]),
            'nhead': hp.choice('nhead', [6]),
            'nlayers_attnEncoders': hp.choice('nlayers_attnEncoders', [4, 6]),
            'K': hp.choice('K', [3]),
            'hop': hp.choice('hop', [3]),
            'local_attn_dim': hp.choice('local_attn_dim', [128, 256])
        }
        combined_space.update(trans_space)

        current_best_params = {
            "nhead": 2,
            "nlayers_attnEncoders": 1,
            "d_ffw": 0,
            "K": 0,
            "hop": 0,
            "local_attn_dim": 1
        }
        combined_best_params.update(current_best_params)

    # print(combined_space)
    return combined_space, [combined_best_params]


def get_resources():
    server_ip = get_host_ip()
    elif server_ip == '':
        basic_server = Server('', cpu=0, gpu=0, mem=0, num_samples=0)

    else:
        raise Exception("[Unseen Server]. '{}' isn't concluded in function 'get_resources'".format(server_ip))

    return basic_server


class Server:
    def __init__(self, name=None, cpu=0, gpu=0, mem=0, num_samples=1):
        self.name = name
        self.cpu = cpu
        self.gpu = gpu
        self.gpu_mem = mem
        self.num_samples = num_samples
        self.gpu_per_trial = 0.5
        self.cpu_per_trial = 0  



def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(('10.0.0.1',8080))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip