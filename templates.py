def set_template(args):
    if args.template is None:
        return

    elif args.template.startswith('train_bert'):
        args.mode = 'train'

        args.dataset_code = 'ml-1m'
        # 注意：这里 min_rating 调整为 4，以匹配你之前较好结果的运行。
        # 这会影响 get_user_item_nums 返回的物品数量变为 3533。
        args.min_rating = 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'leave_one_out'

        args.dataloader_code = 'bert'
        # 批量大小调整为 64，以匹配你之前较好结果的运行。
        batch = 64 # 从 128 改为 64
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        args.train_negative_sampler_code = 'random'
        # train_negative_sample_size 调整为 100，以匹配你之前较好结果的运行。
        args.train_negative_sample_size = 100 # 从 0 改为 100
        args.train_negative_sampling_seed = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 100
        # test_negative_sampling_seed 调整为 0，以匹配你之前较好结果的运行。
        args.test_negative_sampling_seed = 0 # 从 98765 改为 0
        
        args.trainer_code = 'bert'
        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        # 学习率调整为 0.001，以匹配你之前较好结果的运行。
        args.lr = 0.001 # 从 0.0001 改为 0.001
        # 学习率衰减参数调整为 StepLR 的旧配置，以匹配你之前较好结果的运行。
        args.decay_step = 15 # 从 25 改为 15
        args.gamma = 1.0 # 从 1.0 改为 0.1
        # num_epochs 调整为 100，以匹配你之前较好结果的运行。
        args.num_epochs = 100 # 从 300 改为 100
        args.metric_ks = [10, 20, 50] # 调整为之前的 metric_ks
        args.best_metric = 'NDCG@10'

        args.model_code = 'bert'
        args.model_init_seed = 0
        # 这里 get_user_item_nums 将根据 min_rating=4 返回 3533 个物品。
        # 这与你之前命令行中手动指定 3952 不同，但更符合代码内部逻辑。
        # 建议保持 num_items 由 get_user_item_nums 动态计算。
        num_users, num_items = get_user_item_nums(args) 

        args.bert_dropout = 0.1
        args.bert_hidden_units = 64
        # bert_mask_prob 调整为 0.2，以匹配你之前较好结果的运行。
        args.bert_mask_prob = 0.2 # 从 0.15 改为 0.2
        # bert_max_len 调整为 50，以匹配你之前较好结果的运行。
        args.bert_max_len = 50 # 从 200 改为 50
        args.bert_num_blocks = 2
        # bert_num_heads 调整为 4，以匹配你之前较好结果的运行。
        args.bert_num_heads = 4 # 从 2 改为 4
        args.bert_num_items = num_items # 保持由 get_user_item_nums 动态设置
        # weight_decay 调整为 0，以匹配你之前较好结果的运行。
        args.weight_decay = 0 # 从 0.01 改为 0

def get_user_item_nums(args):
    if args.dataset_code == 'ml-1m':
        if args.min_rating == 4 and args.min_uc == 5 and args.min_sc == 0:
            return 6034, 3533
        elif args.min_rating == 0 and args.min_uc == 5 and args.min_sc == 0:
            return 6040, 3706
    raise ValueError()
