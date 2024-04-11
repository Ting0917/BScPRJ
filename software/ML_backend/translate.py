"""
Author: Ting-Chen Chen
Original Authors: Xinyun Chen and Chang Liu and Dawn Song
paper: Tree-to-tree Neural Networks for Program Translation
url: http://arxiv.org/abs/1802.03691
"""
import csv
import os
import random
import sys
import time
import argparse

from six.moves import xrange
import json

import torch
from torch import cuda
from torch.nn.utils import clip_grad_norm

import data_utils
import network

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, default='tree2tree',
                    choices=['tree2tree'])
parser.add_argument('--param_init', type=float, default=0.1,
                    help='Parameters are initialized over uniform distribution in (-param_init, param_init)')
parser.add_argument('--num_epochs', type=int, default=1,
                    help='number of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.005,
                    help='learning rate')
parser.add_argument('--learning_rate_decay_factor', type=float, default=0.8,
                    help='learning rate decays by this much')
parser.add_argument('--learning_rate_decay_steps', type=int, default=2000,
                    help='decay the learning rate after certain steps')
parser.add_argument('--max_gradient_norm', type=float, default=5.0,
                    help='clip gradients to this norm')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size')
parser.add_argument('--max_depth', type=int, default=100,
                    help='max depth for tree models')
parser.add_argument('--hidden_size', type=int, default=256,
                    help='size of each model layer')
parser.add_argument('--embedding_size', type=int, default=256,
                    help='size of the embedding')
parser.add_argument('--dropout_rate', type=float, default=0.5,
                    help='dropout rate')
parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the model')
parser.add_argument('--source_vocab_size', type=int, default=0,
                    help='source vocabulary size (0: no limit)')
parser.add_argument('--target_vocab_size', type=int, default=0,
                    help='target vocabulary size (0: no limit)')
parser.add_argument('--train_dir', type=str, default='../model_ckpts/tree2tree/',
                    help='training directory')
parser.add_argument('--load_model', type=str, default="best_loss_translate_200.ckpt",
                    help='path to the pretrained model')
parser.add_argument('--vocab_filename', type=str, default=None,
                    help='filename for the vocabularies')
parser.add_argument('--steps_per_checkpoint', type=int, default=1,
                    help='number of training steps per checkpoint')
parser.add_argument('--max_source_len', type=int, default=115,
                    help='max length for input')
parser.add_argument('--max_target_len', type=int, default=315,
                    help='max length for output')
parser.add_argument('--test', action='store_true',
                    help='set to true for testing')
parser.add_argument('--input_format', type=str,
                    default='tree', choices=['seq', 'tree'])
parser.add_argument('--output_format', type=str,
                    default='tree', choices=['seq', 'tree'])
parser.add_argument('--no_attention', action='store_true',
                    help='set to true to disable attention')
parser.add_argument('--no_pf', action='store_true',
                    help='set to true to disable parent attention feeding')

# Pascal-Java Dataset
parser.add_argument('--train_data', type=str, default='source_pascal_target_java_train.json',
                    help='training data')
parser.add_argument('--val_data', type=str, default='../../parser/data/source_pascal_target_java_validation.json',
                    help='training data')
parser.add_argument('--test_data', type=str, default='../../parser/data/source_pascal_target_java_atom_test.json',
                    help='test data')

parser.add_argument('--result_dir', default='../result/',
                    help='director to save the translation result.')
parser.add_argument('--train_process_data_dir', type=str, default='train_process_data.csv',
                    help='directory to save data during training process')
parser.add_argument('--source_lang', type=str, default='py',
                    help='the source language of the translator')
parser.add_argument('--target_lang', type=str, default='js',
                    help='the target language of the translator')

args = parser.parse_args()

def create_model(source_vocab_size, target_vocab_size, source_vocab_list, target_vocab_list, dropout_rate, max_source_len, max_target_len):
    """
    Initializes the Tree2Tree model with given parameters and loads pretrained model if available.
    
    Parameters:
    - source_vocab_size: Integer, the size of the source vocabulary.
    - target_vocab_size: Integer, the size of the target vocabulary.
    - source_vocab_list: List of tokens in the source vocabulary.
    - target_vocab_list: List of tokens in the target vocabulary.
    - dropout_rate: Float, the dropout rate used for regularization.
    - max_source_len: Integer, the maximum length of the source sequences.
    - max_target_len: Integer, the maximum length of the target sequences.
    
    return: model, An instance of the Tree2TreeModel with initialized weights or loaded from a pretrained model.
    """
    
    model = network.Tree2TreeModel(
        source_vocab_size,
        target_vocab_size,
        source_vocab_list,
        target_vocab_list,
        args.max_depth,
        args.embedding_size,
        args.hidden_size,
        args.num_layers,
        args.max_gradient_norm,
        args.batch_size,
        args.learning_rate,
        dropout_rate,
        args.no_pf,
        args.no_attention)

    if cuda.is_available():
        model.cuda()

    if args.load_model:
        print("Reading model parameters from %s" % args.load_model)
        pretrained_model = torch.load(args.load_model)
        model.load_state_dict(pretrained_model)
    else:
        print("Created model with fresh parameters.")
        model.init_weights(args.param_init)
    return model


def step_tree2tree(model, encoder_inputs, init_decoder_inputs, feed_previous=False):
    """
    Performs a single training or inference step for the Tree2Tree model.
    
    param model: The Tree2TreeModel instance to perform the step on.
    param encoder_inputs: Input data for the encoder.
    param init_decoder_inputs: Initial input data for the decoder.
    param feed_previous: Boolean, indicates whether to feed previous predictions into the next time step.
    
    returns: Total loss for the training step, and output predictions if in inference mode.
    """
    if feed_previous == False:
        model.dropout_rate = args.dropout_rate
    else:
        model.dropout_rate = 0.0

    predictions_per_batch, prediction_managers = model(
        encoder_inputs, init_decoder_inputs, feed_previous=feed_previous)

    total_loss = None
    for (predictions, target) in predictions_per_batch:
        loss = model.loss_function(predictions, target)
        if total_loss is None:
            total_loss = loss
        else:
            total_loss += loss

    total_loss /= len(encoder_inputs)

    if feed_previous:
        output_predictions = []
        for prediction_manager in prediction_managers:
            output_predictions.append(model.tree2seq(prediction_manager, 1))

    if feed_previous == False:
        model.optimizer.zero_grad()
        total_loss.backward()
        if args.max_gradient_norm > 0:
            clip_grad_norm(model.parameters(), args.max_gradient_norm)
        model.optimizer.step()

    for idx in range(len(encoder_inputs)):
        encoder_inputs[idx].clear_states()

    if feed_previous:
        return total_loss.item(), output_predictions
    else:
        return total_loss.item()


def evaluate(model, test_set, source_vocab, target_vocab, source_vocab_list, target_vocab_list):
    """
    Evaluates the model on a test set and computes various metrics.
    
    param model: The Tree2TreeModel instance to evaluate.
    param test_set: The dataset to evaluate the model on.
    param source_vocab: Dictionary mapping source vocabulary tokens to indices.
    param target_vocab: Dictionary mapping target vocabulary tokens to indices.
    param source_vocab_list: List of tokens in the source vocabulary.
    param target_vocab_list: List of tokens in the target vocabulary.
    
    Outputs evaluation metrics including loss, token accuracy, and program accuracy.
    """
    test_loss = 0
    acc_tokens = 0
    tot_tokens = 0
    tot_output_tokens = 0
    acc_programs = 0
    tot_programs = len(test_set)
    res = []
    average_EDR = 0

    for idx in xrange(0, len(test_set), args.batch_size):
        # here so-called "decoder_inputs" is not actually the input of the decoder
        # it is the ground truth TreeManagers.
        # The data it represents will be called 'target_tree' in the model.forward()
        # and in the model.forward(),
        # decoder_input is the prediction_tree.root, which is t_t, the prediction result - a token id
        encoder_inputs, decoder_inputs = model.get_batch(test_set, start_idx=idx)
        eval_loss, raw_outputs = step_tree2tree(model, encoder_inputs, decoder_inputs, feed_previous=True)
        test_loss += len(encoder_inputs) * eval_loss
        for i in xrange(len(encoder_inputs)):
            if idx + i >= len(test_set):
                break
            current_output = []
            
            for j in xrange(len(raw_outputs[i])):
                current_output.append(raw_outputs[i][j])
            
            current_source, current_target, current_source_manager, current_target_manager = test_set[idx + i]
            
            current_source = data_utils.serialize_tree(current_source)
            res.append((current_source, current_target, current_output))

            tot_tokens += len(current_target)
            tot_output_tokens += len(current_output)
            all_correct = 1
            wrong_tokens = 0
            # Save Translation Result
            return save_translation_result(
                idx, current_output, source_vocab, target_vocab, source_vocab_list, target_vocab_list)


def edit_distance(output, target):
    """
    Computes the edit distance between two sequences (output and target).
    
    param output: List of tokens representing the model's output.
    param target: List of tokens representing the target sequence.
    
    return: The edit distance (integer) between the output and target sequences.
    """
    output_len = len(output)
    target_len = len(target)
    OPT = [[0 for i in range(target_len+1)]
           for j in range(output_len + 1)]  # DP table

    # OPT[i][0] = i, assign values to first column
    for i in range(1, output_len+1):
        OPT[i][0] = i
    # OPT[0][j] = j, assign values to first row
    for j in range(1, target_len+1):
        OPT[0][j] = j
    single_insert_cost = 1
    single_delete_cost = 1
    single_align_cost = 1
    for i in range(1, output_len+1):  # row
        for j in range(1, target_len+1):  # column
            delta = single_align_cost if output[i-1] != target[j-1] else 0
            alignment_cost = OPT[i-1][j-1] + delta
            delete_cost = OPT[i-1][j] + single_delete_cost
            insertion_cost = OPT[i][j-1] + single_insert_cost
            OPT[i][j] = min(alignment_cost, delete_cost, insertion_cost)
    return OPT[output_len][target_len]


def save_translation_result(count, current_output_token_id_list, source_vocab, target_vocab, source_vocab_list, target_vocab_list):
    """
    Saves the translation result to a file. If the model name includes 'eval_loss' or 'loss', 
    it creates a specific directory for those results and saves the output there.

    param count: An integer counter indicating the sequence number of the test data being processed.
    param current_output_token_id_list: List of token IDs representing the model's output.
    param source_vocab: Dictionary mapping source vocabulary tokens to their IDs.
    param target_vocab: Dictionary mapping target vocabulary tokens to their IDs.
    param source_vocab_list: List of all tokens in the source vocabulary.
    param target_vocab_list: List of all tokens in the target vocabulary.
    """
    if not os.path.isdir(args.result_dir):
        os.makedirs(args.result_dir)
    output_str = ""
    target_vocab_id_to_token = {value: key for (
        key, value) in target_vocab.items()}

    for token_id in current_output_token_id_list:
        if token_id in target_vocab_id_to_token:
            output_token = target_vocab_id_to_token[token_id]
            if isinstance(output_token, bytes):
                output_token = output_token.decode('utf-8')
            output_str = output_str + output_token
        else:
            output_str = output_str + "OOV "
    if 'eval_loss' in args.load_model:
        f_dir = args.result_dir + 'best_eval_loss/'
    elif 'loss' in args.load_model:
        f_dir = args.result_dir + 'best_loss/'
    if not os.path.isdir(f_dir):
        os.makedirs(f_dir)
    return output_str



def get_tree_depth(one_TreeManager):
    """
    Calculates the maximum depth of a tree managed by a TreeManager object.

    param one_TreeManager: A TreeManager object containing trees.

    return: depth: Integer, the maximum depth of trees within the TreeManager.
    """
    current_tree = None
    depth = 0
    for i in range(one_TreeManager.num_trees):
        current_tree = one_TreeManager.trees[i]
        current_tree_depth = current_tree.depth
        if current_tree_depth >= depth:
            depth = current_tree_depth
    return depth


def calculate_tree_depth_statistics(data_set):
    """
    Calculates and prints the average, minimum, and maximum tree depths for source and target trees
    in a dataset.

    param data_set: a list of (source JSON in token id, target JSON in token id,source TreeManager in token id, target TreeManager in token id)
    """
    source_TreeMangers = []
    target_TreeMangers = []
    for _, _, sourceTreeManager, targetTreeManager in data_set:
        source_TreeMangers.append(sourceTreeManager)
        target_TreeMangers.append(targetTreeManager)
    source_tree_depth = []
    target_tree_depth = []
    for i in range(len(source_TreeMangers)):
        sourceTreeManager = source_TreeMangers[i]
        targetTreeManager = target_TreeMangers[i]
        source_depth = get_tree_depth(sourceTreeManager)
        target_depth = get_tree_depth(targetTreeManager)
        source_tree_depth.append(source_depth)
        target_tree_depth.append(target_depth)
    source_average_depth = sum(source_tree_depth) / len(source_tree_depth)
    target_average_depth = sum(target_tree_depth) / len(target_tree_depth)
    print('Source Tree average depth ', source_average_depth)
    print('Target Tree average depth ', target_average_depth)
    print('Source Tree min depth ', min(source_tree_depth))
    print('Source Tree max depth ', max(source_tree_depth))
    print('Target Tree min depth ', min(target_tree_depth))
    print('Target Tree max depth ', max(target_tree_depth))


def train(train_data, val_data, source_vocab, target_vocab,
          source_vocab_list, target_vocab_list, source_serialize, target_serialize):
    """
    Trains the Tree2Tree model using the specified training and validation datasets, then saves the best models
    based on training loss and evaluation loss.

    param train_data, val_data: Training and validation datasets.
    param source_vocab, target_vocab: Dictionaries for source and target vocabularies, respectively.
    param source_vocab_list, target_vocab_list: Lists of tokens in the source and target vocabularies, respectively.
    param source_serialize, target_serialize: Flags indicating whether source and target data are serialized.
    """
    print("Reading training and val data :")
    train_set = data_utils.prepare_data(train_data, source_vocab, target_vocab,
                                        args.input_format, args.output_format, source_serialize, target_serialize)
    val_set = data_utils.prepare_data(val_data, source_vocab, target_vocab,
                                      args.input_format, args.output_format, source_serialize, target_serialize)

    calculate_tree_depth_statistics(train_set)
    if not os.path.isdir(args.train_dir):
        os.makedirs(args.train_dir)
    print("Creating %d layers of %d units." %
          (args.num_layers, args.hidden_size))
    model = create_model(len(source_vocab), len(target_vocab), source_vocab_list,
                         target_vocab_list, args.dropout_rate, args.max_source_len, args.max_target_len)

    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []

    train_data_size = len(train_set)

    # Record training process data
    best_eval_loss_model = None
    best_eval_loss_ckpt_path = ""
    best_eval_loss = float("inf")
    best_loss_model = None
    best_loss_ckpt_path = ""
    best_loss = float("inf")
    # create the csv writer
    f = open('train_process_data.csv', 'w', encoding='UTF8')
    header = ['step(checkpoint)', 'step_time',
              'loss', 'is_best_loss', 'eval_loss', 'is_best_eval_loss']
    csv_writer = csv.writer(f)
    csv_writer.writerow(header)
    for epoch in range(args.num_epochs):
        random.shuffle(train_set)
        for batch_idx in range(0, train_data_size, args.batch_size):
            start_time = time.time()
            encoder_inputs, decoder_inputs = model.get_batch(
                train_set, start_idx=batch_idx)


            step_loss = step_tree2tree(
                model, encoder_inputs, decoder_inputs, feed_previous=False)

            step_time += (time.time() - start_time) / args.steps_per_checkpoint
            loss += step_loss / args.steps_per_checkpoint
            current_step += 1

            print("step " + str(current_step) + " ", end="")
            if current_step % args.learning_rate_decay_steps == 0 and model.learning_rate > 0.0001:
                model.decay_learning_rate(args.learning_rate_decay_factor)

            if current_step % args.steps_per_checkpoint == 0:
                previous_losses.append(loss)
                # Save the Model with Best loss score
                csv_row = [current_step, step_time, loss]
                csv_row.append('')
                if loss <= best_loss:
                    best_loss = loss
                    best_loss_ckpt_path = os.path.join(
                        args.train_dir, "best_loss_" + "translate_" + str(current_step) + ".ckpt")
                    csv_row[-1] = 'T'
                    best_loss_model = model.state_dict()

                encoder_inputs, decoder_inputs = model.get_batch(
                    val_set, start_idx=0)
                
                eval_loss, decoder_outputs = step_tree2tree(model, encoder_inputs, decoder_inputs, feed_previous=True)

                print("learning rate %.4f step-time %.2f loss "
                      "%.2f" % (model.learning_rate, step_time, loss), end="")
                print("  eval: loss %.2f" % eval_loss)
                csv_row.append(eval_loss)
                csv_row.append('')
                # Save the Model with best evaluate loss Score.
                if eval_loss <= best_eval_loss:
                    best_eval_loss = eval_loss
                    best_eval_loss_ckpt_path = os.path.join(
                        args.train_dir, "best_eval_loss_" + "translate_" + str(current_step) + ".ckpt")
                    csv_row[-1] = 'T'
                    best_eval_loss_model = model.state_dict()
                csv_writer.writerow(csv_row)
                
                step_time, loss = 0.0, 0.0
                sys.stdout.flush()

    print('best eval path', best_eval_loss_ckpt_path)
    print('best loss path', best_loss_ckpt_path)
    torch.save(best_eval_loss_model, best_eval_loss_ckpt_path)
    print("Best Model saved with eval_loss = " +
          str(best_eval_loss), best_eval_loss_ckpt_path)
    torch.save(best_loss_model, best_loss_ckpt_path)
    print("Best Model saved with loss = " +
          str(best_loss), best_loss_ckpt_path)


def test(train_data, test_data, source_vocab, target_vocab, source_vocab_list, target_vocab_list, source_serialize, target_serialize):
    """
    Evaluates the Tree2Tree model on a test dataset and calculates tree depth statistics for both training and test datasets.

    param train_data, test_data: Datasets for training (for statistics calculation) and testing.
    param source_vocab, target_vocab: Dictionaries for source and target vocabularies, respectively.
    param source_vocab_list, target_vocab_list: Lists of tokens in the source and target vocabularies, respectively.
    param source_serialize, target_serialize: Flags indicating whether source and target data are serialized.
    """    
    
    model = create_model(len(source_vocab), len(target_vocab), source_vocab_list,
                         target_vocab_list, 0.0, args.max_source_len, args.max_target_len)
    test_set = data_utils.prepare_data(test_data, source_vocab, target_vocab,
                                       args.input_format, args.output_format, source_serialize, target_serialize)
    # train_set is only built here to calculate the statistics.
    # it is not used anywhere else while testing a trained Tree2Tree model.
    train_set = data_utils.prepare_data(train_data, source_vocab, target_vocab,
                                        args.input_format, args.output_format, source_serialize, target_serialize)
    print('train data statistics:')
    calculate_tree_depth_statistics(train_set)
    print('test data statistics:')
    calculate_tree_depth_statistics(test_set)
    return evaluate(model, test_set, source_vocab, target_vocab,
             source_vocab_list, target_vocab_list)


def save_vocabulary_in_csv(source_vocab, target_vocab):
    """
    Saves the source and target vocabularies to separate CSV files. Each row in the CSV files
    maps a vocabulary token to its corresponding integer ID.

    param source_vocab: A dictionary mapping each source language token to its unique integer ID.
    param target_vocab: A dictionary mapping each target language token to its unique integer ID.
    """
    header = ['vocab', 'token_id']
    with open('source_vocab.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)  # csv writer
        writer.writerow(header)
        for vocab in source_vocab.keys():
            writer.writerow([vocab, source_vocab[vocab]])
    with open('target_vocab.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)  # csv writer
        writer.writerow(header)
        for vocab in target_vocab.keys():
            writer.writerow([vocab, target_vocab[vocab]])
    print('File Wrote: Vocabulary Files.')


def main(test_data):
    source_serialize = False
    target_serialize = False
    
    if args.no_attention:
        args.no_pf = True
    train_data = json.load(open(args.train_data, 'r'))
    source_vocab, target_vocab, source_vocab_list, target_vocab_list = data_utils.build_vocab(
        train_data, args.vocab_filename, args.input_format, args.output_format)
    save_vocabulary_in_csv(source_vocab, target_vocab)
    
    return test(train_data, test_data, source_vocab, target_vocab, source_vocab_list,
             target_vocab_list, source_serialize, target_serialize)

