import torch
from torch.utils.data import DataLoader
from torch import nn
from pytorch_transformers import AdamW, WEIGHTS_NAME, WarmupLinearSchedule
import csv
import numpy as np
import os
import logging
from fp16 import FP16_Module, FP16_Optimizer
from parallel import DataParallelModel, DataParallelCriterion
from collections import OrderedDict
from utils import *
from settings import args, TASK_DICT, init_logging, MODEL_CONFIG, MODEL_CLASS, SPECIAL_TOKENS, CONFIG_CLASS
from settings import TOKENIZER, SPECIAL_TOKEN_IDS, FILL_VAL, SAVE_NAME, FINAL_SAVE_NAME, TOKENS_WEIGHT, CONFIG_NAME
from scheduler import AnnealingLR
from regularizers import REG_TYPES, REG_TYPE_KEYS, Weight_Regularized_AdamW, Weight_Regularized_SGD
from torch.nn import CrossEntropyLoss
import random
logger = logging.getLogger(__name__)
seed_bias=4369
random.seed(args.seed+seed_bias)
np.random.seed(args.seed+seed_bias)
torch.manual_seed(args.seed+seed_bias)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

RESTORE_NAME=''



def train(task_ids, model):
    tasks = [args.tasks[task_id] for task_id in task_ids]

    logger.info("start to train { task: %s, seq train type: %s }" % (tasks, args.seq_train_type))
    model_dir = get_model_dir(tasks)
    make_dir(model_dir)

    train_dataset = [TASK_DICT[t]["train"] for t in tasks]
    train_extra_data = []
    if "lll" in args.seq_train_type and task_ids[0] > 0 and not args.skip_tasks:
        prev_task = args.tasks[task_ids[0]-1]
        with torch.no_grad():


            create_extra_data(tasks[0], prev_task, model, train_extra_data)


    elif "gem" in args.seq_train_type and task_ids[0] > 0: 
        get_real_data(tasks[0], train_extra_data, accum=False, encode=True)
        args.memory_data.append(train_extra_data)
        train_extra_data = []
    logger.info('extra training data size: {}'.format(len(train_extra_data)))



    if not model:
        # which_model_to_load = model_dir if os.path.isfile(os.path.join(model_dir, FINAL_SAVE_NAME)) else args.model_name
        # model = MODEL_CLASS.from_pretrained(args.pretrain_path).cuda()
        logger.info('It is the first time to load model. Loading model from {}'.format(args.pretrain_path))
        # model = MODEL_CLASS.from_pretrained(args.pretrain_path).cuda()
        model = MODEL_CLASS(MODEL_CONFIG)
        pretrained_state_dict = torch.load(os.path.join(args.pretrain_path, 'pytorch_model.bin'))
        model_state_dict = model.state_dict()
        #print(model_state_dict.keys())
        temp_dict = OrderedDict()
        for k, v in pretrained_state_dict.items():
            if 'transformer.' + k in model_state_dict:
                temp_dict['transformer.' + k] = v
        model_state_dict.update(temp_dict)
        model.load_state_dict(model_state_dict)
        model.resize_token_embeddings(len(TOKENIZER))
        if not args.fp32:
            model = FP16_Module(model)

    MODEL_CONFIG.vocab_size = len(TOKENIZER)#move it from settings.py because init model can not update transformer.wte
    gen_token = get_gen_token(tasks[0])
    TOKENIZER.add_tokens([gen_token])
    TOKENIZER.save_pretrained(model_dir)
    SPECIAL_TOKENS[tasks[0]] = gen_token
    SPECIAL_TOKEN_IDS[tasks[0]] = TOKENIZER.convert_tokens_to_ids(gen_token)
    logger.info('gen token = {} , gen token id = {}'.format(gen_token, SPECIAL_TOKEN_IDS[tasks[0]]))
    MODEL_CONFIG.vocab_size = len(TOKENIZER)
    MODEL_CONFIG.to_json_file(os.path.join(model_dir,CONFIG_NAME))
    global TOKENS_WEIGHT
    if len(TOKENIZER) != TOKENS_WEIGHT.shape[0]:
        TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))

    if args.skip_tasks and len(tasks) == 1:
        logger.info("*********** skip task: {} ***********".format(tasks[0]))
        if tasks[0] in args.skip_tasks:
            if len(args.skip_tasks) == 1:
                model_dir = get_model_dir(tasks)
                model_path = os.path.join(model_dir, FINAL_SAVE_NAME)#RESTORE_NAME)
                config_path = os.path.join(model_dir,CONFIG_NAME)
                model_config = CONFIG_CLASS.from_json_file(config_path)
                model = MODEL_CLASS(model_config).cuda()
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict)
                if not args.fp32:
                    model = FP16_Module(model)
                if args.seq_train_type in REG_TYPE_KEYS:
                    logger.info("calulating reg_params ...")
                    train_qadata = QADataset(train_dataset, "train", SPECIAL_TOKEN_IDS[tasks[0]], train_extra_data)
                    max_train_batch_size = max(len(train_qadata) // args.min_n_steps, args.min_batch_size)
                    train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
                    parallel_model = DataParallelModel(WrapModel(model), args.device_ids)
                    regularizer = REG_TYPES[args.seq_train_type](model, parallel_model, [train_dataloader], tasks[0])
                    regularizer.task_start_do()
                    regularizer.task_end_do()
                    torch.save(model.state_dict(), os.path.join(model_dir, FINAL_SAVE_NAME))
                    logger.info("done reg_params!")
            args.skip_tasks.remove(tasks[0])
            #if args.restore_epoch in ['9','finish',9]:

            return model

    model.resize_token_embeddings(len(TOKENIZER))

    if not args.fp32:  # again because resize_token_embeddings makes embedding layer fp32
        model = FP16_Module(model)
    
    parallel_model = DataParallelModel(WrapModel(model), args.device_ids)

    train_qadata = QADataset(train_dataset, "train", SPECIAL_TOKEN_IDS[tasks[0]], train_extra_data)
    max_train_batch_size = max(len(train_qadata) // args.min_n_steps, args.min_batch_size)
    train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
    if not args.unbound and args.seq_train_type != "multitask":
        #n_train_epochs = TASK_DICT[tasks[0]]["n_train_epochs"]
        n_train_epochs = args.n_train_epochs[tasks[0]]
    else:
        n_train_epochs = args.n_train_epochs['_'.join(tasks)]
    
    #added by Han Wang 2022/1/21
    if args.generate_mode in [6,7]:
        turns = n_train_epochs//args.train_gap
        if turns%2>0:
            new_train_epochs = (1+turns)*args.train_gap
            logger.info('Mode %s ,train_epoch=%s is changed to %s for gpt2 and vae can be trained equally.'%(args.generate_mode,n_train_epochs,new_train_epochs))
            n_train_epochs = int(new_train_epochs)
    
    n_train_optimization_steps = len(train_qadata) * n_train_epochs
    logger.info('len of train dataset: {} , max train batch size {} , num of opt steps: {}'.format(
        len(train_qadata), max_train_batch_size, n_train_optimization_steps))

    param_optimizer = list(model.named_parameters())
    #for n, p in param_optimizer:
        #print(n,p.type())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if "gem" in args.seq_train_type:
        model.task_id = task_ids[0]
        if not hasattr(model, "grad_dims"):
            model.grad_dims = []
            for param in model.parameters():
                model.grad_dims.append(param.data.numel())
        if not hasattr(model, "grads"):
            model.grads = torch.zeros(sum(model.grad_dims),len(args.tasks))
            model.grads = model.grads.cuda()

    if args.seq_train_type in REG_TYPE_KEYS:
        optimizer = Weight_Regularized_AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #print(optimizer.param_groups)
    if not args.fp32:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=None, dynamic_loss_scale=True,dynamic_loss_args={'scale_window': 100, 'min_scale': 1, 'delayed_shift': 2})


    mid_iters=-1


    scheduler = AnnealingLR(optimizer, start_lr=args.learning_rate, warmup_iter=int(args.n_warmup_ratio*len(train_qadata)),
            num_iters=int(n_train_optimization_steps), decay_style=args.decay_style,generate_mode=args.generate_mode,mid_iter=int(mid_iters))
    train_loss_fct = DataParallelCriterion(CrossEntropyLoss(ignore_index=FILL_VAL, weight=TOKENS_WEIGHT), args.device_ids)

    if args.seq_train_type in REG_TYPE_KEYS:
        copy_train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
        prev_task = args.tasks[task_ids[0]-1]
        regularizer = REG_TYPES[args.seq_train_type](model, parallel_model, [copy_train_dataloader], tasks[0], prev_task)
        regularizer.task_start_do()
    mid_epoch = n_train_epochs//args.mid_split
    warmup={
        'beta':{
            'start_epoch':0 if args.generate_mode<2 else mid_epoch - 1,
            'end_epoch':int(0.8*n_train_epochs),
            'factor':0.25
        }
    }
    tot_n_steps = 0
    train_once = TrainStep(model, optimizer, scheduler)
    if "gem" in args.seq_train_type and task_ids[0] != 0:
        gem_step = GEMStep(model, parallel_model, train_loss_fct, optimizer)
    model.train()
    begin_epoch = int(args.restore_epoch) if args.restore_inplace else 0
    args.restore_inplace=False
    freeze_flag=False 
    freeze_gpt2=False

    gpt2_eps=[]
    vae_eps=[]
    for i in range(n_train_epochs):
        change = i//args.train_gap
        if change%2 == 0:
            gpt2_eps.append(i)
        else:
            vae_eps.append(i)
    if args.generate_mode in [6,7]:
        print('Mode %s, train GPT2 in epochs %s and VAE in epochs %s'%(args.generate_mode,gpt2_eps.__str__(),vae_eps.__str__()))
    if args.generate_mode<0:
        model.use_vae=True
        parallel_model.use_vae=True
    
    for ep in range(begin_epoch,n_train_epochs):
        cum_loss, cum_qa_loss, cum_lm_loss, cum_idt_loss, cum_reconstruct_loss,cum_kld_loss, cur_n_inputs = 0, 0, 0, 0, 0, 0, 0
        cur_epoch_loss = []
        print('generate_mode=',args.generate_mode,'ep=',ep,'freeze_flag=',freeze_flag,'freeze_gpt2=',freeze_gpt2)
        
        if args.generate_mode in [6,7]:
            if ep in gpt2_eps:
                logger.info('Mode %s, train gpt2 only'%args.generate_mode)
                model.use_vae=False
                parallel_model.use_vae=False
                model=freeze_params_or_not(model, ['transformer'],freeze=False)
                freeze_gpt2=False
                model=freeze_params_or_not(model, ['vae'],freeze=True)
                freeze_flag = True
            if ep in vae_eps:
                model.use_vae=True
                parallel_model.use_vae=True

                logger.info('Mode %s, train both gpt2 and vae'%args.generate_mode)
                model=freeze_params_or_not(model, ['vae'],freeze=False)
                freeze_flag = False

        for n_steps, (_, _, cqa, _, Y, gen_X, gen_Y,idt_X,idt_Y,task_ids,is_extra) in enumerate(train_dataloader):

            n_inputs = sum(_cqa.shape[0] for _cqa in cqa)

            for i in range(len(cqa)):#len(devices)
                cqa[i] = (cqa[i].to(args.device_ids[i]),)
                Y[i] = Y[i].to(args.device_ids[i])
                gen_X[i] = (gen_X[i].to(args.device_ids[i]),)
                gen_Y[i] = gen_Y[i].to(args.device_ids[i])
                idt_X[i] = (idt_X[i].to(args.device_ids[i]),)
                idt_Y[i] = idt_Y[i].to(args.device_ids[i])
                task_ids[i] = task_ids[i].to(args.device_ids[i])
                is_extra[i] = is_extra[i].to(args.device_ids[i])

            if args.use_id_task and args.id_task_gamma>=0:
                losses = get_idt_losses(parallel_model, cqa, Y, gen_X, gen_Y, idt_X, idt_Y,task_ids,is_extra,train_loss_fct,warmup,ep,freeze_gpt2)
            else:
                losses = get_losses(parallel_model, cqa, Y, gen_X, gen_Y, train_loss_fct)
            loss = sum([losses['qa_loss'],losses['lm_loss'],losses['idt_loss'],losses['reconstruct_loss'],losses['kld']])
            if "gem" in args.seq_train_type and task_ids[0] != 0:
                gem_step(task_ids[0])
            train_once(loss, n_inputs)

            qa_loss = losses['qa_loss'].item() * n_inputs
            lm_loss = losses['lm_loss'].item() * n_inputs
            idt_loss = losses['idt_loss'].item()*n_inputs if args.use_id_task else 0.0
            reconstruct_loss = losses['reconstruct_loss'].item()*n_inputs if args.vae_idx>-1 else 0.0
            kld_loss = losses['kld'].item()*n_inputs if args.vae_idx>-1 else 0.0

            cum_loss += (qa_loss + lm_loss + idt_loss + reconstruct_loss + kld_loss)
            cum_qa_loss += qa_loss
            cum_lm_loss += lm_loss
            cum_idt_loss += idt_loss
            cum_reconstruct_loss += reconstruct_loss
            cum_kld_loss += kld_loss
            cur_n_inputs += n_inputs
            cur_epoch_loss.append([cum_loss/cur_n_inputs, cum_qa_loss/cur_n_inputs, cum_lm_loss/cur_n_inputs,cum_idt_loss/cur_n_inputs,cum_reconstruct_loss/cur_n_inputs, cum_kld_loss/cur_n_inputs])
            if (n_steps + 1 ) % args.logging_steps == 0:
                logger.info('progress {:.3f} , lr {:.1E} , loss {:.3f} , qa loss {:.3f} , lm loss {:.3f} , idt_loss {:.3f}, rec_loss {:.3f}, kld_loss {:.3f}, avg batch size {:.1f}'.format(
                    ep + cur_n_inputs/len(train_qadata), scheduler.get_lr(),
                    cum_loss/cur_n_inputs, cum_qa_loss/cur_n_inputs, cum_lm_loss/cur_n_inputs,cum_idt_loss/cur_n_inputs,cum_reconstruct_loss/cur_n_inputs, cum_kld_loss/cur_n_inputs,
                    cur_n_inputs/(n_steps + 1)
                ))
        np.savetxt(os.path.join(model_dir, 'loss_all_%s.txt'%(ep+1)), np.asarray(cur_epoch_loss), fmt="%s",
                   delimiter=",",
                   encoding='utf-8')
        cur_epoch_loss.clear()
        torch.save(model.state_dict(), os.path.join(model_dir, SAVE_NAME+str(ep+1)))
        tot_n_steps += (n_steps + 1)
        logger.info('epoch {}/{} done , tot steps {} , lr {:.1E} , loss {:.2f}, qa loss {:.2f}, lm loss {:.2f}, idt_loss {:.2f}, rec_loss {:.2f}, kld_loss {:.2f}, avg batch size {:.1f}'.format(
            ep+1, n_train_epochs, tot_n_steps, scheduler.get_lr(),
            cum_loss/cur_n_inputs, cum_qa_loss/cur_n_inputs, cum_lm_loss/cur_n_inputs, cum_idt_loss/cur_n_inputs,cum_reconstruct_loss/cur_n_inputs, cum_kld_loss/cur_n_inputs,
            cur_n_inputs/(n_steps+1)
        ))

    # task end do for reg
    if args.seq_train_type in REG_TYPE_KEYS:
        regularizer.task_end_do()
    torch.save(model.state_dict(), os.path.join(model_dir, FINAL_SAVE_NAME))
    

    
    model.use_vae=True
    model=freeze_params_or_not(model, ['vae','transformer'],freeze=False)
    return model





if __name__ == '__main__':

    if not args.debug:
        logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
        logging.getLogger("pytorch_transformers.tokenization_utils").setLevel(logging.CRITICAL)

    make_dir(args.model_dir_root)
    
    init_logging(os.path.join(args.model_dir_root, 'log_train.txt'))
    logger.info('args = {}'.format(str(args)))
    



    RESTORE_NAME='model-%s'%args.restore_epoch

    model = None
    if args.seq_train_type == "multitask":
        model = train(list(range(len(args.tasks))), model)
    else:
        if args.unbound:
            TASK_DICT = lll_unbound_setting(split_size=args.unbound)
        
        begin_id=0

        for task_id in range(begin_id,len(args.tasks)):
            model = train([task_id], model)

