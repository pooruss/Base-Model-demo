# model config
def init_bert_net_config(args, logger, print_config=True):
    config = dict()
    config["initializer_range"] = args.initializer_range
    config["mask_id"] = args.mask_id
    config["do_train"] = args.do_train
    config["do_test"] = args.do_test
    config["pretrained_path"] = args.pretrained_path
    if print_config is True:
        logger.info('----------- BMKG base model Network Configuration -------------')
        for arg, value in config.items():
            logger.info('%s: %s' % (arg, value))
        logger.info('------------------------------------------------')
    return config


# train config
def init_train_config(args, logger, print_config=True):
    config = dict()
    config["do_train"] = args.do_train
    config["do_test"] = args.do_test
    config["batch_size"] = args.batch_size
    config["vocab_size"] = args.vocab_size
    config["epoch"] = args.epoch
    config["pad_id"] = args.padding_id
    config["mask_id"] = args.mask_id
    config["learning_rate"] = args.learning_rate
    config["weight_decay"] = args.weight_decay
    config["use_cuda"] = args.use_cuda
    config["gpus"] = args.gpus
    config["gpu_ids"] = args.gpu_ids
    config["node"] = args.node
    config["model_name"] = args.model_name
    config["save_path"] = args.save_path
    config["use_ema"] = args.use_ema
    config["ema_decay"] = args.ema_decay
    config["bmtrain"] = args.bmtrain
    config["checkpoint_num"] = args.checkpoint_num
    config["task_name"] = args.task_name

    if print_config is True:
        logger.info('----------- Train Configuration -------------')
        for arg, value in config.items():
            logger.info('%s: %s' % (arg, value))
        logger.info('------------------------------------------------')
    return config


def init_kepler_train_config(args, logger, print_config=True):
    config = dict()
    config["do_train"] = args.do_train
    config["do_val"] = args.do_val
    config["do_test"] = args.do_test
    config["batch_size"] = args.batch_size
    config["epoch"] = args.epoch
    config["pad_id"] = args.padding_id
    config["learning_rate"] = args.learning_rate
    config["weight_decay"] = args.weight_decay
    config["use_cuda"] = args.use_cuda
    config["gpu_ids"] = args.gpu_ids
    config["node"] = args.node
    config["model_name"] = args.model_name
    config["skip_steps"] = args.skip_steps
    config["save_path"] = args.save_path
    config["bmtrain"] = args.bmtrain
    config["checkpoint_num"] = args.checkpoint_num
    config["num_classes"] = args.num_classes
    config["base_model"] = args.base_model
    config["num_classes"] = args.num_classes
    config["pooler_dropout"] = args.pooler_dropout
    config["gamma"] = args.gamma
    config["nrelation"] = args.nrelation
    config["ke_model"] = args.ke_model
    config["padding_idx"] = args.padding_idx
    config["model_root"] = args.model_root
    if print_config is True:
        logger.info('----------- Train Configuration -------------')
        for arg, value in config.items():
            logger.info('%s: %s' % (arg, value))
        logger.info('------------------------------------------------')
    return config
