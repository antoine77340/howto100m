import argparse

def get_args(description='Youtube-Text-Video'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--train_csv',
        type=str,
        default='data/HowTo100M_v1.csv',
        help='train csv')
    parser.add_argument(
        '--features_path_2D',
        type=str,
        default='feature_2d',
        help='feature path for 2D features')
    parser.add_argument(
        '--features_path_3D',
        type=str,
        default='feature_3d',
        help='feature path for 3D features')
    parser.add_argument(
        '--caption_path',
        type=str,
        default='data/caption.pickle',
        help='caption pickle file path')
    parser.add_argument(
        '--word2vec_path',
        type=str,
        default='data/GoogleNews-vectors-negative300.bin',
        help='word embedding path')
    parser.add_argument(
        '--pretrain_path',
        type=str,
        default='',
        help='pre train model path')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='',
        help='checkpoint model folder')
    parser.add_argument('--num_thread_reader', type=int, default=1,
                                help='')
    parser.add_argument('--embd_dim', type=int, default=2048,
                                help='embedding dim')
    parser.add_argument('--lr', type=float, default=0.0001,
                                help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20,
                                help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256,
                                help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500,
                                help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9,
                                help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=10,
                                help='Information display frequence')
    parser.add_argument('--feature_dim', type=int, default=4096,
                                help='video feature dimension')
    parser.add_argument('--we_dim', type=int, default=300,
                                help='word embedding dimension')
    parser.add_argument('--seed', type=int, default=1,
                                help='random seed')
    parser.add_argument('--verbose', type=int, default=1,
                                help='')
    parser.add_argument('--max_words', type=int, default=20,
                                help='')
    parser.add_argument('--min_words', type=int, default=0,
                                help='')
    parser.add_argument('--feature_framerate', type=int, default=1,
                                help='')
    parser.add_argument('--min_time', type=float, default=5.0,
                                help='Gather small clips')
    parser.add_argument('--margin', type=float, default=0.1,
                                help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5,
                                help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1,
                                help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1,
                                help='Num of pair to output from data loader')
    parser.add_argument('--lsmdc', type=int, default=0,
                                help='Train on LSDMC data')
    parser.add_argument('--youcook', type=int, default=0,
                                help='Train on YouCook2 data')
    parser.add_argument('--msrvtt', type=int, default=0,
                                help='Train on MSRVTT data')
    parser.add_argument('--eval_lsmdc', type=int, default=0,
                                help='Evaluate on LSMDC data')
    parser.add_argument('--eval_msrvtt', type=int, default=0,
                                help='Evaluate on MSRVTT data')
    parser.add_argument('--eval_youcook', type=int, default=0,
                                help='Evaluate on YouCook2 data')
    parser.add_argument('--sentence_dim', type=int, default=-1,
                                help='sentence dimension')
    parser.add_argument(
        '--youcook_train_path',
        type=str,
        default='data/youcook_train.pkl',
        help='')
    parser.add_argument(
        '--youcook_val_path',
        type=str,
        default='data/youcook_val.pkl',
        help='')
    parser.add_argument(
        '--msrvtt_test_csv_path',
        type=str,
        default='data/MSRVTT_JSFUSION_test.csv',
        help='')
    parser.add_argument(
        '--msrvtt_train_csv_path',
        type=str,
        default='data/MSRVTT_train.csv',
        help='')
    parser.add_argument(
        '--msrvtt_train_json_path',
        type=str,
        default='data/MSRVTT_data.json',
        help='')
    parser.add_argument(
        '--msrvtt_train_features_path',
        type=str,
        default='data/MSRVTT_train.pth',
        help='')
    parser.add_argument(
        '--msrvtt_test_features_path',
        type=str,
        default='data/MSRVTT_JSFUSION_test.pth',
        help='')
    parser.add_argument(
        '--lsmdc_test_csv_path',
        type=str,
        default='data/LSMDC16_challenge_1000_publictect.csv',
        help='')
    parser.add_argument(
        '--lsmdc_test_features_path',
        type=str,
        default='data/LSMDC_test.pth',
        help='')
    parser.add_argument(
        '--lsmdc_train_csv_path',
        type=str,
        default='data/LSMDC16_annos_training.csv',
        help='')
    parser.add_argument(
        '--lsmdc_train_features_path',
        type=str,
        default='data/LSMDC_train.pth',
        help='')
    args = parser.parse_args()
    return args
