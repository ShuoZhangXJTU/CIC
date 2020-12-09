import argparse
from tune_space import get_host_ip

def get_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cmp', action='store_true', default=False, help

    # -- model parameters
    parser.add_argument('--dropout', type=float, default=0.2)
    # - embedding
    parser.add_argument('--emsize', type=int, default=300, help='embedding dimension')
    # - lstm
    parser.add_argument('--nhid_lstm', type=int, default=516, help='lstm hidden')
    parser.add_argument('--nlayers_lstmEncoders', type=int, default=2, help='num of LSTM encoder layers')
    # - transformer
    parser.add_argument('--d_model', type=int, default=300, help='the number of expected features in the input')
    parser.add_argument('--d_ffw', type=int, default=1024, help='the number of expected features in the input')
    parser.add_argument('--nlayers_attnEncoders', type=int, default=4, help='num of attn encoder layers')
    parser.add_argument('--nhead', type=int, default=4, help='num of attn heads')
    # - PBR
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--N', type=int, default=3)
    parser.add_argument('--local_attn_dim', type=int, default=512)
    # - FFNN
    parser.add_argument('--nhid_ffnn', type=int, default=256, help='hdim of decoder FFNN')

    # -- training setting
    parser.add_argument('--model_name', type=str, default='default')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--dataset', type=str, default='cn', choices=['cn', 'en'])
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'tune', 'pred'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--print_wt', action='store_true', default=False)

    parser.add_argument('--use_l2', action='store_true', default=False)
    parser.add_argument('--reg_lambda', type=float, default=0.1)

    # -- data path / filename
    server_ip = get_host_ip()
    if server_ip == '':
        base_path = ''

    parser.add_argument('--path_text_processed', type=str, default=base_path+'data/processed')
    parser.add_argument('--path_doc', type=str, default=base_path + 'data/doc/')
    parser.add_argument('--path_rawEngEleBase', type=str, default=base_path + 'data/'
                                                                              'labelled_contracts/'
                                                                              'elements_contracts/')
    # - embeddings
    parser.add_argument('--path_emb', type=str, default=base_path+'data/embeddings/')
    # - test paras
    parser.add_argument('--path_model_para', type=str, default=base_path+'data/model_para/')

    return parser.parse_args()