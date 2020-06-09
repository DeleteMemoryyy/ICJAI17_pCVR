import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchsnooper

from utility import neg_log_loss

def tensor_neg_log_loss(y_predict, y_true):
    loss = -(y_true * torch.log(y_predict) + (1 - y_true) * torch.log(1.0 - y_predict))
    return loss.mean()


class DeepFM(nn.Module):
    def __init__(self, feature_sizes, continuous_field_size, embedding_size=4,
                 hidden_dims=[128, 128], use_fm1=False, num_classes=10, dropout=[0.5, 0.5],
                 use_cuda=True, verbose=False):
        """
        Initialize a new network
        Inputs:
        - feature_size: A list of integer giving the size of features for each field.
        - embedding_size: An integer giving size of feature embedding.
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to predict. For example,
                    someone may rate 1,2,3,4 or 5 stars to a film.
        - batch_size: An integer giving size of instances used in each interation.
        - use_cuda: Bool, Using cuda or not
        - verbose: Bool
        """
        super().__init__()
        self.field_size = len(feature_sizes)
        self.continuous_field_size = continuous_field_size
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.hidden_dims = hidden_dims
        self.use_fm1 = use_fm1
        self.num_classes = num_classes
        self.dtype = torch.float

        """
            check if use cuda
        """
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        """
            init fm part
        """

        #        self.fm_first_order_embeddings = nn.ModuleList(
        #            [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])
        fm_first_order_linears = nn.ModuleList(
            [nn.Linear(feature_size, self.embedding_size) for feature_size in
             self.feature_sizes[:self.continuous_field_size]])
        fm_first_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in
             self.feature_sizes[self.continuous_field_size:]])
        self.fm_first_order_models = fm_first_order_linears.extend(
            fm_first_order_embeddings)

        #        self.fm_second_order_embeddings = nn.ModuleList(
        #            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
        fm_second_order_linears = nn.ModuleList(
            [nn.Linear(feature_size, self.embedding_size) for feature_size in
             self.feature_sizes[:self.continuous_field_size]])
        fm_second_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in
             self.feature_sizes[self.continuous_field_size:]])
        self.fm_second_order_models = fm_second_order_linears.extend(
            fm_second_order_embeddings)

        """
            init deep part
        """
        all_dims = [self.field_size * self.embedding_size] + \
                   self.hidden_dims + [self.num_classes]
        for i in range(1, len(hidden_dims) + 1):
            setattr(self, 'linear_' + str(i),
                    nn.Linear(all_dims[i - 1], all_dims[i]))
            # nn.init.kaiming_normal_(self.fc1.weight)
            setattr(self, 'batchNorm_' + str(i),
                    nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_' + str(i),
                    nn.Dropout(dropout[i - 1]))

    def forward(self, Xi, Xv):
        """
        Forward process of network.
        Inputs:
        - Xi: A tensor of input's index, shape of (N, field_size, 1)
        - Xv: A tensor of input's value, shape of (N, field_size, 1)
        """
        """
            fm part
        """

        # fm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
        #                           enumerate(self.fm_first_order_models)]
        # fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
        # fm_second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
        #                            enumerate(self.fm_second_order_models)]

        #        fm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.fm_first_order_models)]
        #        fm_first_order_emb_arr = [(emb(Xi[:, i, :]) * Xv[:, i])  for i, emb in enumerate(self.fm_first_order_models)]

        if self.use_fm1:
            fm_first_order_emb_arr = []
            for i, emb in enumerate(self.fm_first_order_models):
                if i < self.continuous_field_size:
                    Xi_tem = Xi[:, i, :].to(device=self.device, dtype=torch.float)
                    fm_first_order_emb_arr.append(
                        (torch.sum(emb(Xi_tem).unsqueeze(1), 1).t() * Xv[:, i]).t())
                else:
                    Xi_tem = Xi[:, i, :].to(device=self.device, dtype=torch.long)
                    fm_first_order_emb_arr.append(
                        (torch.sum(emb(Xi_tem), 1).t() * Xv[:, i]).t())
            #        print("successful")
            #        print(len(fm_first_order_emb_arr))
            fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
        # use 2xy = (x+y)^2 - x^2 - y^2 reduce calculation
        #        fm_second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.fm_second_order_models)]
        # fm_second_order_emb_arr = [(emb(Xi[:, i]) * Xv[:, i]) for i, emb in enumerate(self.fm_second_order_embeddings)]
        fm_second_order_emb_arr = []
        for i, emb in enumerate(self.fm_second_order_models):
            if i < self.continuous_field_size:
                Xi_tem = Xi[:, i, :].to(device=self.device, dtype=torch.float)
                fm_second_order_emb_arr.append(
                    (torch.sum(emb(Xi_tem).unsqueeze(1), 1).t() * Xv[:, i]).t())
            else:
                Xi_tem = Xi[:, i, :].to(device=self.device, dtype=torch.long)
                fm_second_order_emb_arr.append(
                    (torch.sum(emb(Xi_tem), 1).t() * Xv[:, i]).t())

        fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
        fm_sum_second_order_emb_square = fm_sum_second_order_emb * \
                                         fm_sum_second_order_emb  # (x+y)^2
        fm_second_order_emb_square = [
            item * item for item in fm_second_order_emb_arr]
        fm_second_order_emb_square_sum = sum(
            fm_second_order_emb_square)  # x^2+y^2
        fm_second_order = (fm_sum_second_order_emb_square -
                           fm_second_order_emb_square_sum) * 0.5
        """
            deep part
        """
        #        print(len(fm_second_order_emb_arr))
        #        print(torch.cat(fm_second_order_emb_arr, 1).shape)
        deep_emb = torch.cat(fm_second_order_emb_arr, 1)
        deep_out = deep_emb
        for i in range(1, len(self.hidden_dims) + 1):
            deep_out = getattr(self, 'linear_' + str(i))(deep_out)
            deep_out = getattr(self, 'batchNorm_' + str(i))(deep_out)
            deep_out = getattr(self, 'dropout_' + str(i))(deep_out)
        #            print("successful")
        """
            sum
        """
        #        print("1",torch.sum(fm_first_order, 1).shape)
        #        print("2",torch.sum(fm_second_order, 1).shape)
        #        print("deep",torch.sum(deep_out, 1).shape)
        #        print("bias",bias.shape)
        bias = torch.nn.Parameter(torch.randn(Xi.size(0))).to(self.device, self.dtype)
        # deep_out = F.sigmoid(deep_out)
        sum_fm2 = torch.sum(fm_second_order, 1)
        sum_deep = torch.sum(deep_out, 1)
        total_sum = \
            sum_fm2 + \
            sum_deep + \
            bias
        if self.use_fm1:
            sum_fm1 = torch.sum(fm_first_order, 1)
            total_sum = total_sum + sum_fm1

        result = F.sigmoid(total_sum)
        eps = 5e-3
        result = torch.clamp(result, eps, 1.0 - eps)
        return result

    def fit(self, loader_train, loader_val, optimizer, epochs=1, verbose=False, print_every=5):
        """
        Training a model and valid accuracy.
        Inputs:
        - loader_train: I
        - loader_val: .
        - optimizer: Abstraction of optimizer used in training process, e.g., "torch.optim.Adam()""torch.optim.SGD()".
        - epochs: Integer, number of epochs.
        - verbose: Bool, if print.
        - print_every: Integer, print after every number of iterations.
        """
        """
            load input data
        """
        model = self.train().to(device=self.device)
        criterion = tensor_neg_log_loss

        for epoch in range(epochs):
            for t, (xv, y) in enumerate(loader_train):
                arr = np.empty(shape=(xv.shape[0], xv.shape[1]))
                for i in range(xv.shape[0]):
                    arr[i] = np.array(list(range(xv.shape[1])))
                arr = arr.reshape((arr.shape[0], arr.shape[1], 1))
                xi = torch.Tensor(arr)
                xi = xi.to(device=self.device, dtype=self.dtype)
                xv = xv.to(device=self.device, dtype=self.dtype)
                y = y.to(device=self.device, dtype=self.dtype)
                result = model(xi, xv)
                #                print(result.shape)
                #                print(y.shape)
                loss = criterion(result, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose and t % print_every == 0:
                    valid_loss = self.validation(loader_val, model)
                    print('Epoch {} Iteration {}, loss = {:.4f}, validation loss = {:.4f}'.format(epoch, t, loss.item(),
                                                                                                  valid_loss))

        return model

    # @torchsnooper.snoop()
    def validation(self, loader, model):
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            y_test = torch.Tensor().to(device=self.device, dtype=self.dtype)
            y_predict = torch.Tensor().to(device=self.device, dtype=self.dtype)
            for xv, y in loader:
                # move to device, e.g. GPU
                arr = np.empty(shape=(xv.shape[0], xv.shape[1]))
                for i in range(xv.shape[0]):
                    arr[i] = np.array(list(range(xv.shape[1])))
                arr = arr.reshape((arr.shape[0], arr.shape[1], 1))
                xi = torch.Tensor(arr)
                xi = xi.to(device=self.device, dtype=self.dtype)
                xv = xv.to(device=self.device, dtype=self.dtype)
                y = y.to(device=self.device, dtype=self.dtype)
                result = model(xi, xv)
                preds = result.to(dtype=self.dtype)

                y_predict = torch.cat((y_predict, preds), 0)
                y_test = torch.cat((y_test, y), 0)

            # y_predict = y_predict.cpu().detach().numpy()
            # y_test = y_test.cpu().detach().numpy()
            # criterion = neg_log_loss
            # loss = criterion(y_predict, y_test)

            criterion = tensor_neg_log_loss
            loss = criterion(y_predict, y_test).item()
            return loss
