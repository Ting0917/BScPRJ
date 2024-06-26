"""
original authors: Xinyun Chen and Chang Liu and Dawn Song
paper: Tree-to-tree Neural Networks for Program Translation
url: http://arxiv.org/abs/1802.03691
"""

import numpy as np
from six.moves import xrange
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch import cuda
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import torch.nn.functional as F
from Tree import *

import data_utils


def repackage_state(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(v for v in h)


class TreeEncoder(nn.Module):
    def __init__(self,
                 source_vocab_size,
                 embedding_size,
                 hidden_size,
                 batch_size
                 ):
        super(TreeEncoder, self).__init__()
        self.source_vocab_size = source_vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.cuda_flag = cuda.is_available()

        self.encoder_embedding = nn.Embedding(
            self.source_vocab_size, self.embedding_size)

        self.ix = nn.Linear(self.hidden_size, self.embedding_size, bias=True)
        self.ilh = nn.Linear(self.hidden_size, self.hidden_size)
        self.irh = nn.Linear(self.hidden_size, self.hidden_size)

        self.fx = nn.Linear(self.hidden_size, self.embedding_size, bias=True)
        self.flh = nn.Linear(self.hidden_size, self.hidden_size)
        self.frh = nn.Linear(self.hidden_size, self.hidden_size)

        self.ox = nn.Linear(self.hidden_size, self.embedding_size, bias=True)
        self.olh = nn.Linear(self.hidden_size, self.hidden_size)
        self.orh = nn.Linear(self.hidden_size, self.hidden_size)

        self.ux = nn.Linear(self.hidden_size, self.embedding_size, bias=True)
        self.ulh = nn.Linear(self.hidden_size, self.hidden_size)
        self.urh = nn.Linear(self.hidden_size, self.hidden_size)

    def calc_root(self, inputs, child_h, child_c):
        i = F.sigmoid(self.ix(inputs) +
                      self.ilh(child_h[:, 0]) + self.irh(child_h[:, 1]))
        o = F.sigmoid(self.ox(inputs) +
                      self.olh(child_h[:, 0]) + self.orh(child_h[:, 1]))
        u = F.tanh(self.ux(inputs) +
                   self.ulh(child_h[:, 0]) + self.urh(child_h[:, 1]))

        fx = self.fx(inputs)
        fx = torch.stack([fx, fx], dim=1)
        fl = self.flh(child_h[:, 0])
        fr = self.frh(child_h[:, 1])
        f = torch.stack([fl, fr], dim=1)
        f = f + fx
        f = F.sigmoid(f)
        fc = F.torch.mul(f, child_c)
        c = F.torch.mul(i, u) + F.torch.sum(fc, 1)
        h = F.torch.mul(o, F.tanh(c))
        return h, c

    def encode(self, encoder_inputs, children_h, children_c):
        embedding = self.encoder_embedding(encoder_inputs)
        embedding = embedding.squeeze()
        if len(embedding.size()) == 1:
            embedding = embedding.unsqueeze(0)
        encoder_outputs = self.calc_root(embedding, children_h, children_c)
        return encoder_outputs

    def forward(self, encoder_managers):
        """
        Grouping nodes into several groups to calculate encoder_outputs
        The rule of grouping: 
                 - in a bottom-up fashion, 
                   add the right-most node until reaching a node that 
                   at least one of its children's LSTM is not calculated yet.
                 - Leaf node with no children are considered to have both 
                   children's LSTM states as Zeros.
        """
        queue = []
        head = 0
        max_num_trees = 0
        visited_idx = []

        # Collect the first group of nodes to be calculated.
        for encoder_manager_idx in range(len(encoder_managers)):
            encoder_manager = encoder_managers[encoder_manager_idx]
            max_num_trees = max(max_num_trees, encoder_manager.num_trees)
            idx = encoder_manager.num_trees - 1
            # compute embedding for the node from bottom up, idx from len -> 0
            while idx >= 0:
                current_tree = encoder_manager.get_tree(idx)
                canVisited = True
                if current_tree.lchild is not None:
                    ltree = encoder_manager.get_tree(current_tree.lchild)
                    if ltree.state is None:
                        canVisited = False
                if current_tree.rchild is not None:
                    rtree = encoder_manager.get_tree(current_tree.rchild)
                    if rtree.state is None:
                        canVisited = False

                if canVisited:
                    root = current_tree.root
                    if current_tree.lchild is None:
                        children_c = Variable(
                            torch.zeros(self.hidden_size))
                        children_h = Variable(
                            torch.zeros(self.hidden_size))
                        if self.cuda_flag:
                            children_c = children_c.cuda()
                            children_h = children_h.cuda()
                    else:
                        children_h, children_c = ltree.state
                        children_h = children_h
                        children_c = children_c
                    if current_tree.rchild is None:
                        rchild_c = Variable(torch.zeros(self.hidden_size))
                        rchild_h = Variable(torch.zeros(self.hidden_size))
                        if self.cuda_flag:
                            rchild_c = rchild_c.cuda()
                            rchild_h = rchild_h.cuda()
                        children_c = torch.stack(
                            [children_c, rchild_c], dim=0)
                        children_h = torch.stack(
                            [children_h, rchild_h], dim=0)
                    else:
                        rchild_h, rchild_c = rtree.state
                        rchild_h = rchild_h
                        rchild_c = rchild_c
                        children_c = torch.stack(
                            [children_c, rchild_c], dim=0)
                        children_h = torch.stack(
                            [children_h, rchild_h], dim=0)
                    queue.append((encoder_manager_idx, idx,
                                 root, children_h, children_c))
                else:
                    break
                idx -= 1
            visited_idx.append(idx)

        # Calculates the encoder_output of each group of nodes,
        # and collect the next group of nodes to be calcualted.
        while head < len(queue):
            encoder_inputs = []
            children_h = []
            children_c = []
            tree_idxes = []
            # stack the encoder_inputs, child_h, child_c of nodes grouped in one group together
            # then calculate the encoder_output of this group together by passing to self.encode().
            while head < len(queue):
                # encoder_manager_idx: the idx of an encoder_manager in encoder_managers
                # idx: the idx of a tree in one encoder_manager(TreeManager)
                encoder_manager_idx, idx, root, child_h, child_c = queue[head]
                current_tree = encoder_managers[encoder_manager_idx].get_tree(
                    idx)
                tree_idxes.append((encoder_manager_idx, idx))
                encoder_inputs.append(root)
                children_h.append(child_h)
                children_c.append(child_c)
                head += 1
            encoder_inputs = torch.stack(encoder_inputs, dim=0)
            children_h = torch.stack(children_h, dim=0)
            children_c = torch.stack(children_c, dim=0)
            if self.cuda_flag:
                encoder_inputs = encoder_inputs.cuda()
            # self.encode() is where encoder_output is calculated.
            encoder_outputs = self.encode(
                encoder_inputs, children_h, children_c)
            for i in range(len(tree_idxes)):
                current_encoder_manager_idx, current_idx = tree_idxes[i]
                child_h = encoder_outputs[0][i]
                child_c = encoder_outputs[1][i]
                # save the calculated LSTM embedding in the BinaryTree.state variable
                encoder_managers[current_encoder_manager_idx].trees[current_idx].state = child_h, child_c
                current_tree = encoder_managers[current_encoder_manager_idx].get_tree(
                    current_idx)

                # Collect the next group of nodes to be calcuated, append them into the queue.
                if current_tree.parent == visited_idx[current_encoder_manager_idx]:
                    encoder_manager_idx = current_encoder_manager_idx
                    encoder_manager = encoder_managers[encoder_manager_idx]
                    idx = visited_idx[encoder_manager_idx]
                    while idx >= 0:
                        current_tree = encoder_manager.get_tree(idx)
                        canVisited = True
                        if current_tree.lchild is not None:
                            ltree = encoder_manager.get_tree(
                                current_tree.lchild)
                            if ltree.state is None:
                                canVisited = False
                        if current_tree.rchild is not None:
                            rtree = encoder_manager.get_tree(
                                current_tree.rchild)
                            if rtree.state is None:
                                canVisited = False

                        if canVisited:
                            root = current_tree.root
                            if current_tree.lchild is None:
                                children_c = Variable(
                                    torch.zeros(self.hidden_size))
                                children_h = Variable(
                                    torch.zeros(self.hidden_size))
                                if self.cuda_flag:
                                    children_c = children_c.cuda()
                                    children_h = children_h.cuda()
                            else:
                                children_h, children_c = ltree.state
                                children_h = children_h
                                children_c = children_c

                            if current_tree.rchild is None:
                                rchild_c = Variable(
                                    torch.zeros(self.hidden_size))
                                rchild_h = Variable(
                                    torch.zeros(self.hidden_size))
                                if self.cuda_flag:
                                    rchild_c = rchild_c.cuda()
                                    rchild_h = rchild_h.cuda()
                            else:
                                rchild_h, rchild_c = rtree.state
                                rchild_h = rchild_h
                                rchild_c = rchild_c

                            children_c = torch.stack(
                                [children_c, rchild_c], dim=0)
                            children_h = torch.stack(
                                [children_h, rchild_h], dim=0)
                            queue.append(
                                (encoder_manager_idx, idx, root, children_h, children_c))
                        else:
                            break
                        idx -= 1
                    visited_idx[encoder_manager_idx] = idx

        PAD_state_token = Variable(torch.zeros(self.hidden_size))
        if self.cuda_flag:
            PAD_state_token = PAD_state_token.cuda()

        encoder_h_state = []  # h states of all root nodes of TreeManagers
        encoder_c_state = []  # c states of all root nodes of TreeManagers
        init_encoder_outputs = []  # a list of [h states of all nodes of each TreeManager]
        init_attention_masks = []  # a list of [attention mask of all nodes of each TreeManager]
        for encoder_manager in encoder_managers:
            root = encoder_manager.get_tree(0)
            h, c = root.state
            encoder_h_state.append(h)
            encoder_c_state.append(c)
            init_encoder_output = []  # h states of all nodes of this TreeManager
            for tree in encoder_manager.trees:
                init_encoder_output.append(tree.state[0])
            attention_mask = [0] * len(init_encoder_output)
            current_len = len(init_encoder_output)
            if current_len < max_num_trees:
                # PAD encoder output to the same length
                init_encoder_output = init_encoder_output + \
                    [PAD_state_token] * (max_num_trees - current_len)
                # PAD attentino mask to the same length
                attention_mask = attention_mask + \
                    [1] * (max_num_trees - current_len)
            attention_mask = Variable(torch.ByteTensor(attention_mask))
            if self.cuda_flag:
                attention_mask = attention_mask.cuda()
            init_attention_masks.append(attention_mask)
            init_encoder_output = torch.stack(init_encoder_output, dim=0)
            init_encoder_outputs.append(init_encoder_output)

        init_encoder_outputs = torch.stack(init_encoder_outputs, dim=0)
        init_attention_masks = torch.stack(init_attention_masks, dim=0)

        return init_encoder_outputs, init_attention_masks, encoder_h_state, encoder_c_state


class Tree2TreeModel(nn.Module):
    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 source_vocab,
                 target_vocab,
                 max_depth,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 dropout_rate,
                 no_pf,
                 no_attention
                 ):
        super(Tree2TreeModel, self).__init__()
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_depth = max_depth
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.max_gradient_norm = max_gradient_norm
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.no_pf = no_pf
        self.no_attention = no_attention
        self.cuda_flag = cuda.is_available()

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)

        self.encoder = TreeEncoder(
            self.source_vocab_size, self.embedding_size, self.hidden_size, self.batch_size)

        self.decoder_embedding = nn.Embedding(
            self.target_vocab_size, self.embedding_size)

        if self.no_pf:
            self.decoder_l = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size,
                                     num_layers=self.num_layers, batch_first=True, dropout=self.dropout_rate)
            self.decoder_r = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size,
                                     num_layers=self.num_layers, batch_first=True, dropout=self.dropout_rate)
        else:
            self.decoder_l = nn.LSTM(input_size=self.embedding_size + self.hidden_size, hidden_size=self.hidden_size,
                                     num_layers=self.num_layers, batch_first=True, dropout=self.dropout_rate)
            self.decoder_r = nn.LSTM(input_size=self.embedding_size + self.hidden_size, hidden_size=self.hidden_size,
                                     num_layers=self.num_layers, batch_first=True, dropout=self.dropout_rate)
        # Applies a linear transformation to the incoming data: y = Wx + b
        # (size_of_each_input_sample, size_of_each_output_sample, bias=True to learn an additive bias)
        self.attention_linear = nn.Linear(
            self.hidden_size * 2, self.hidden_size, bias=True)
        self.attention_tanh = nn.Tanh()

        self.output_linear_layer = nn.Linear(
            self.hidden_size, self.target_vocab_size, bias=True)

        self.loss_function = nn.CrossEntropyLoss(size_average=False)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def init_weights(self, param_init):
        for param in self.parameters():
            param.data.uniform_(-param_init, param_init)

    def decay_learning_rate(self, learning_rate_decay_factor):
        self.learning_rate *= learning_rate_decay_factor
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def train(self):
        if self.max_gradient_norm > 0:
            clip_grad_norm(self.parameters(), self.max_gradient_norm)
        self.optimizer.step()

    def tree2seq(self, prediction_manager, current_idx):
        current_tree = prediction_manager.get_tree(current_idx)
        if current_tree.prediction == data_utils.EOS_ID:
            return []
        prediction = [data_utils.LEFT_BRACKET_ID]
        prediction.append(current_tree.prediction)
        if current_tree.lchild is not None:
            prediction = prediction + \
                self.tree2seq(prediction_manager, current_tree.lchild)
        prediction.append(data_utils.RIGHT_BRACKET_ID)
        if current_tree.rchild is not None:
            prediction = prediction + \
                self.tree2seq(prediction_manager, current_tree.rchild)
        return prediction

    def attention(self, encoder_outputs, attention_masks, decoder_output):
        """
        This function computes softmax(e_t) with attention,
        using the hidden state(decoder_output) and e_s(encoder_attention).
        """
        dotted = torch.bmm(encoder_outputs, decoder_output.unsqueeze(2))
        dotted = dotted.squeeze()
        if len(dotted.size()) == 1:
            dotted = dotted.unsqueeze(0)
        dotted.data.masked_fill_(attention_masks.data, -float('inf'))
        attention = nn.Softmax()(dotted)
        encoder_attention = torch.bmm(torch.transpose(
            encoder_outputs, 1, 2), attention.unsqueeze(2))
        encoder_attention = encoder_attention.squeeze()
        if len(encoder_attention.size()) == 1:
            encoder_attention = encoder_attention.unsqueeze(0)
        # e_t
        res = self.attention_tanh(self.attention_linear(
            torch.cat([decoder_output, encoder_attention], 1)))
        return res

    def predict(self, decoder_output, encoder_outputs, attention_masks):
        """
        decoder_output: the hidden states

        This function returns the value of t_t before argmax.
        t_t = argmax(W * softmax(e_t))
        where e_t is the attention output calculated using e_s and h,
            h: hidden state
            e_s: attention value
        W means a linear layer.
        This function returns W * softmax(e_t)
        """
        if self.no_attention:
            output = decoder_output
            attention_output = decoder_output
        else:
            # attention_output = softmax(e_t)
            attention_output = self.attention(
                encoder_outputs, attention_masks, decoder_output)
            if self.dropout_rate > 0:
                output = self.dropout(attention_output)
            else:
                output = attention_output
        # W* softmax(e_t)
        output_linear = self.output_linear_layer(output)
        return output_linear, attention_output

    def decode(self, encoder_outputs, attention_masks, init_state, init_decoder_inputs, attention_inputs):
        """
        called from forward():
        predictions_logits_l, predictions_logits_r, states_l, states_r, attention_outputs_l, attention_outputs_r 
        = self.decode(encoder_outputs, attention_masks, (init_h_states, init_c_states), decoder_inputs, attention_inputs)
        @param: encoder_outputs: hidden states obtained from the encoder
        @param: init_decoder_inputs: the BinaryTree.root collection of the prediction manager.

        This function predicts the t_t(before argmax) value of the left child and right child of a node(BinaryTree).
        """
        embedding = self.decoder_embedding(init_decoder_inputs)
        state_l = init_state
        state_r = repackage_state(init_state)
        if self.no_pf:
            decoder_inputs = embedding
        else:  # parent feeding
            decoder_inputs = torch.cat([embedding, attention_inputs], 2)

        output_l, state_l = self.decoder_l(decoder_inputs, state_l)  # LSTM_l
        output_r, state_r = self.decoder_r(decoder_inputs, state_r)  # LSTM_r
        output_l = output_l.squeeze()
        if len(output_l.size()) == 1:
            output_l = output_l.unsqueeze(0)
        output_r = output_r.squeeze()
        if len(output_r.size()) == 1:
            output_r = output_r.unsqueeze(0)
        prediction_l, attention_output_l = self.predict(
            output_l, encoder_outputs, attention_masks)
        prediction_r, attention_output_r = self.predict(
            output_r, encoder_outputs, attention_masks)
        return prediction_l, prediction_r, state_l, state_r, attention_output_l, attention_output_r

    def forward(self, encoder_managers, decoder_managers, feed_previous=False):
        """
          decoder_managers: is a list of TreeManagers, each TreeManager represents one ground truth target data.
          First encode via self.encoder() which calls the forward() function of TreeEncoder.
          The rest of the code acts as the decoder.
        """
        init_encoder_outputs, init_attention_masks, encoder_h_state, encoder_c_state = self.encoder(
            encoder_managers)

        # expanding nodes across all data samples[(data sample idx, node idx in the data sample)]
        queue = []

        prediction_managers = []
        # Initialize empty prediction_manager(TreeManager) for all the data samples
        for idx in range(len(decoder_managers)):
            prediction_managers.append(TreeManager())

        # Copy the LSTM state of the source root node and attach to the prediction root node, 
        # BinaryTree.root = GO_ID = 1
        for idx in xrange(len(decoder_managers)):
            current_target_manager_idx = idx
            current_target_idx = 0
            # create a new binary tree in the prediction_managers[idx],
            #   and return the idx of the newly created bianary tree.
            current_prediction_idx = prediction_managers[idx].create_binary_tree(
                data_utils.GO_ID, None, 0)
            # copy the LSTM states
            prediction_managers[idx].trees[current_prediction_idx].state = encoder_h_state[idx].unsqueeze(
                0), encoder_c_state[idx].unsqueeze(0)

            prediction_managers[idx].trees[current_prediction_idx].target = 0
            queue.append((idx, current_prediction_idx))

        head = 0  

        # predictions of all data samples passed into this forward() function,
        # called `per_batch` just because this model is fed with one batch of data each time
        predictions_per_batch = []
        EOS_token = Variable(torch.LongTensor([data_utils.EOS_ID]))
   
        while head < len(queue):
            init_h_states = []  # [BinaryTree.state[0]]
            init_c_states = []  # [BinaryTree.state[1]]
            decoder_inputs = []  # [BinaryTree.root]
            attention_inputs = []  # [BinaryTree.attention or Zeros if None]
            encoder_outputs = []  # [BinaryTreee.hidden_states]
            attention_masks = []  # output from TreeEncoder. [padded BinaryTree.attention]
            target_seqs_l = []  # [Target BinaryTree.root]
            target_seqs_r = []  # [Target BinaryTree.root]
            # Record nodes(BinaryTrees) grouped in each collection, used when expanding one group of nodes.
            tree_idxes = []

            # Collect all Data into one group to be passed into the self.decode() function to be calculated.
            # The initial collection of the first group of nodes to be decoded. Literally all the root nodes of the prediction TreeManagers.
            while head < len(queue):
                current_tree = prediction_managers[queue[head][0]].get_tree(
                    queue[head][1])
                target_manager_idx = queue[head][0]
                target_idx = current_tree.target
                if target_idx is not None:
                    target_tree = decoder_managers[target_manager_idx].get_tree(
                        target_idx)
                else:
                    target_tree = None
                if target_tree is not None:
                    init_h_state = current_tree.state[0]
                    init_c_state = current_tree.state[1]
                    init_h_state = torch.cat(
                        [init_h_state] * self.num_layers, dim=0)  # torch.Size([1, 256])
                    init_c_state = torch.cat(
                        [init_c_state] * self.num_layers, dim=0)  # torch.Size([1, 256])
                    init_h_states.append(init_h_state)
                    init_c_states.append(init_c_state)
                    tree_idxes.append((queue[head][0], queue[head][1]))
                    # the input of the decoder.
                    decoder_input = current_tree.root
                    decoder_inputs.append(decoder_input)
                    if current_tree.attention is None:
                        attention_input = Variable(
                            torch.zeros(self.hidden_size))  # torch.Size([256])
                        if self.cuda_flag:
                            attention_input = attention_input.cuda()
                    else:
                        attention_input = current_tree.attention
                    attention_inputs.append(attention_input)
                    if queue[head][1] == 0:  # if the current expanding node is a root node
                        target_seq_l = target_tree.root
                        # assign the right child of the root node the <EOS> token
                        target_seq_r = EOS_token
                    else:  # if not root node
                        if target_tree is not None and target_tree.lchild is not None:
                            target_seq_l = decoder_managers[target_manager_idx].trees[target_tree.lchild].root
                        else:
                            target_seq_l = EOS_token
                        if target_tree is not None and target_tree.rchild is not None:
                            target_seq_r = decoder_managers[target_manager_idx].trees[target_tree.rchild].root
                        else:
                            target_seq_r = EOS_token
                    target_seqs_l.append(target_seq_l)
                    target_seqs_r.append(target_seq_r)
                    encoder_outputs.append(
                        init_encoder_outputs[queue[head][0]])
                    attention_masks.append(
                        init_attention_masks[queue[head][0]])
                head += 1
            if len(tree_idxes) == 0:
                break
            init_h_states = torch.stack(init_h_states, dim=1)
            init_c_states = torch.stack(init_c_states, dim=1)
            decoder_inputs = torch.stack(decoder_inputs, dim=0)
            attention_inputs = torch.stack(
                attention_inputs, dim=0).unsqueeze(1)
            target_seqs_l = torch.cat(target_seqs_l, 0)  
            target_seqs_r = torch.cat(target_seqs_r, 0)  
            if self.cuda_flag:
                decoder_inputs = decoder_inputs.cuda()
                target_seqs_l = target_seqs_l.cuda()
                target_seqs_r = target_seqs_r.cuda()

            encoder_outputs = torch.stack(encoder_outputs, dim=0)
            attention_masks = torch.stack(attention_masks, dim=0)

            # Perform Calculation
            predictions_logits_l, predictions_logits_r, states_l, states_r, attention_outputs_l, attention_outputs_r = self.decode(
                encoder_outputs, attention_masks, (init_h_states, init_c_states), decoder_inputs, attention_inputs)
            predictions_per_batch.append((predictions_logits_l, target_seqs_l))
            predictions_per_batch.append((predictions_logits_r, target_seqs_r))

            # apply argmax and obtained the actual token id(those save in the vocabulary csv files).
            if feed_previous:
                predictions_l = predictions_logits_l.max(1)[1]
                predictions_r = predictions_logits_r.max(1)[1]

            # Calculation Finished. Use the calculated result to expand nodes.
            # Collect the next group of nodes to be calculated and append them into the queue.
            for i in xrange(len(tree_idxes)):
                current_prediction_manager_idx, current_prediction_idx = tree_idxes[i]
                target_manager_idx = current_prediction_manager_idx
                current_prediction_tree = prediction_managers[current_prediction_manager_idx].get_tree(
                    current_prediction_idx)
                target_idx = current_prediction_tree.target
                if target_idx is None:
                    target_tree = None
                else:
                    target_tree = decoder_managers[target_manager_idx].get_tree(
                        target_idx)
                if feed_previous == False:
                    if current_prediction_idx == 0:
                        nxt_l_prediction_idx = prediction_managers[current_prediction_manager_idx].create_binary_tree(
                            target_tree.root, current_prediction_idx, current_prediction_tree.depth + 1)
                        prediction_managers[current_prediction_manager_idx].trees[
                            current_prediction_idx].lchild = nxt_l_prediction_idx
                        prediction_managers[current_prediction_manager_idx].trees[nxt_l_prediction_idx].target = target_idx
                        prediction_managers[current_prediction_manager_idx].trees[
                            nxt_l_prediction_idx].state = states_l[0][:, i, :], states_l[1][:, i, :]
                        prediction_managers[current_prediction_manager_idx].trees[
                            nxt_l_prediction_idx].attention = attention_outputs_l[i]
                        queue.append(
                            (current_prediction_manager_idx, nxt_l_prediction_idx))
                    else:
                        if target_tree.lchild is not None:
                            nxt_l_prediction_idx = prediction_managers[current_prediction_manager_idx].create_binary_tree(
                                decoder_managers[target_manager_idx].trees[target_tree.lchild].root, current_prediction_idx, current_prediction_tree.depth + 1)
                            prediction_managers[current_prediction_manager_idx].trees[nxt_l_prediction_idx].target = target_tree.lchild
                            prediction_managers[current_prediction_manager_idx].trees[
                                current_prediction_idx].lchild = nxt_l_prediction_idx
                            prediction_managers[current_prediction_manager_idx].trees[
                                nxt_l_prediction_idx].state = states_l[0][:, i, :], states_l[1][:, i, :]
                            prediction_managers[current_prediction_manager_idx].trees[
                                nxt_l_prediction_idx].attention = attention_outputs_l[i]
                            queue.append(
                                (current_prediction_manager_idx, nxt_l_prediction_idx))
                        if target_idx == 0:
                            continue
                        if target_tree.rchild is not None:
                            nxt_r_prediction_idx = prediction_managers[current_prediction_manager_idx].create_binary_tree(
                                decoder_managers[target_manager_idx].trees[target_tree.rchild].root, current_prediction_idx, current_prediction_tree.depth + 1)
                            prediction_managers[current_prediction_manager_idx].trees[nxt_r_prediction_idx].target = target_tree.rchild
                            prediction_managers[current_prediction_manager_idx].trees[
                                current_prediction_idx].rchild = nxt_r_prediction_idx
                            prediction_managers[current_prediction_manager_idx].trees[
                                nxt_r_prediction_idx].state = states_r[0][:, i, :], states_r[1][:, i, :]
                            prediction_managers[current_prediction_manager_idx].trees[
                                nxt_r_prediction_idx].attention = attention_outputs_r[i]
                            queue.append(
                                (current_prediction_manager_idx, nxt_r_prediction_idx))
                else:
                    if current_prediction_idx == 0:  # The root node of a prediction tree.
                        # Only generate left node.
                        # BECAUSE for CS-JS, their parse tree always start from a root node with only one children.
                        # expanding the left node of current node, assign values to its LSTM state.
                        nxt_l_prediction_idx = prediction_managers[current_prediction_manager_idx].create_binary_tree(
                            predictions_l[i].item(), current_prediction_idx, current_prediction_tree.depth + 1)
                        prediction_managers[current_prediction_manager_idx].trees[
                            current_prediction_idx].lchild = nxt_l_prediction_idx
                        prediction_managers[current_prediction_manager_idx].trees[nxt_l_prediction_idx].prediction = predictions_l[i].item(
                        )
                        prediction_managers[current_prediction_manager_idx].trees[nxt_l_prediction_idx].target = target_idx
                        prediction_managers[current_prediction_manager_idx].trees[
                            nxt_l_prediction_idx].state = states_l[0][:, i, :], states_l[1][:, i, :]
                        prediction_managers[current_prediction_manager_idx].trees[
                            nxt_l_prediction_idx].attention = attention_outputs_l[i]
                        queue.append(
                            (current_prediction_manager_idx, nxt_l_prediction_idx))
                    else: # expand if not <EOS>
                        if predictions_l[i].item() != data_utils.EOS_ID:
                            if target_tree is None or target_tree.lchild is None:
                                nxt_l_prediction_idx = prediction_managers[current_prediction_manager_idx].create_binary_tree(
                                    predictions_l[i].item(), current_prediction_idx, current_prediction_tree.depth + 1)
                                prediction_managers[current_prediction_manager_idx].trees[nxt_l_prediction_idx].target = None
                            else:
                                nxt_l_prediction_idx = prediction_managers[current_prediction_manager_idx].create_binary_tree(
                                    predictions_l[i].item(), current_prediction_idx, current_prediction_tree.depth + 1)
                                prediction_managers[current_prediction_manager_idx].trees[
                                    nxt_l_prediction_idx].target = target_tree.lchild
                            prediction_managers[current_prediction_manager_idx].trees[
                                current_prediction_idx].lchild = nxt_l_prediction_idx
                            prediction_managers[current_prediction_manager_idx].trees[nxt_l_prediction_idx].prediction = predictions_l[i].item(
                            )
                            prediction_managers[current_prediction_manager_idx].trees[
                                nxt_l_prediction_idx].state = states_l[0][:, i, :], states_l[1][:, i, :]
                            prediction_managers[current_prediction_manager_idx].trees[
                                nxt_l_prediction_idx].attention = attention_outputs_l[i]
                            queue.append(
                                (current_prediction_manager_idx, nxt_l_prediction_idx))
                        if target_idx == 0:
                            continue
                        if predictions_r[i].item() == data_utils.EOS_ID:
                            continue
                        if target_tree is None or target_tree.rchild is None:
                            nxt_r_prediction_idx = prediction_managers[current_prediction_manager_idx].create_binary_tree(
                                predictions_r[i].item(), current_prediction_idx, current_prediction_tree.depth + 1)
                            prediction_managers[current_prediction_manager_idx].trees[nxt_r_prediction_idx].target = None
                        else:
                            nxt_r_prediction_idx = prediction_managers[current_prediction_manager_idx].create_binary_tree(
                                predictions_r[i].item(), current_prediction_idx, current_prediction_tree.depth + 1)
                            prediction_managers[current_prediction_manager_idx].trees[nxt_r_prediction_idx].target = target_tree.rchild
                        prediction_managers[current_prediction_manager_idx].trees[
                            current_prediction_idx].rchild = nxt_r_prediction_idx
                        prediction_managers[current_prediction_manager_idx].trees[nxt_r_prediction_idx].prediction = predictions_r[i].item(
                        )
                        prediction_managers[current_prediction_manager_idx].trees[
                            nxt_r_prediction_idx].state = states_r[0][:, i, :], states_r[1][:, i, :]
                        prediction_managers[current_prediction_manager_idx].trees[
                            nxt_r_prediction_idx].attention = attention_outputs_r[i]
                        # collect the next group of nodes to be calculated.
                        queue.append(
                            (current_prediction_manager_idx, nxt_r_prediction_idx))
        return predictions_per_batch, prediction_managers

    def get_batch(self, data, start_idx):

        encoder_managers, decoder_managers = [], []

        for i in xrange(self.batch_size):
            if i + start_idx < len(data):
                encoder_input, decoder_input, encoder_manager, decoder_manager = data[
                    i + start_idx]
            else:
                encoder_input, decoder_input, encoder_manager, decoder_manager = data[i + start_idx - len(
                    data)]

            encoder_managers.append(encoder_manager)
            decoder_managers.append(decoder_manager)
        return encoder_managers, decoder_managers
