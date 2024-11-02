
import lib.config as Config
import lib.constants as Constants
import copy
import random
from sklearn.metrics import roc_auc_score

# newgraph_Facebook_equal_close_ENM.txt

class Graph:
    def __init__(
            self, config: Config,
            prior_filename="NotSpecified.txt",
            post_filename="NotSpecified.txt",
            withBuffer=False,
    ):
        self.config = config
        self.prior_filename = prior_filename
        self.post_filename = post_filename
        self.withBuffer = withBuffer

        # intializes..
        # self.graph
        self.init_graph()


    ###########################
    ### GRAPH UTILS ###########
    ###########################
    def init_graph(self):
        self.graph = [[] * self.config.node_cnt for _ in range(self.config.node_cnt)]

    def read_graph_from_file_path(self, file_path):
        self.graph_file_path = Constants.ricc_dir_path / file_path

        with open(self.graph_file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                node1 = int(line[0])
                node2 = int(line[1])
                self.graph[node1].append(node2)
        
        self.node_num = len(self.graph)
    
    def init(self, is_train=False):
        self.prior_list = [0] * self.config.node_cnt
        self.prior_list = self.read_prior(self.prior_list, is_train=is_train)
        
        post_list = copy.deepcopy(self.prior_list)

        for _ in range(Constants.iteration):
            print("Iteration: ", _)
            post_list_tmp = self.run_lbp(post_list, withBuffer=self.withBuffer)
            post_list = copy.deepcopy(post_list_tmp)
        
        post_file_path = Constants.ricc_dir_path / self.post_filename
        self.save_posterior(post_file_path, post_list)

    
    def read_prior(self, prior_list, is_train=False):
        prior_file_path = Constants.ricc_dir_path / self.prior_filename

        if is_train:
            with open(prior_file_path, "r") as f:
                train_negative = f.readline()
                train_positive = f.readline()

                self.negative_nodes = train_negative.split()
                self.positive_nodes = train_positive.split()

                for negative_idx in self.negative_nodes:
                    prior_list[int(negative_idx)] = -1 * Constants.theta
                for positive_idx in self.positive_nodes:
                    prior_list[int(positive_idx)] = +1 * Constants.theta
        else:
            prior_file_path = Constants.ricc_dir_path / self.prior_filename
            with open(prior_file_path, "r") as f:
                lines = f.read().splitlines()

                for line in lines:
                    line = line.split()

                    node_idx = int(line[0])
                    prior_score = float(line[1])

                    prior_list[node_idx] = prior_score
        
        return prior_list

    def save_posterior(self, file_path, post_list):
        """ save posterior scores in file.
        :return:
        """
        f_post = open(file_path, "w")

        for line in enumerate(post_list):
            f_post.write('{} {:.10f}\n'.format(line[0], line[1]))
            
        f_post.close()
    
    def choose_random_train_set(self):
        turn = 0

        self.random_train_negative = []
        self.random_train_positive = []

        while True:
            rand = random.choice(self.negative_nodes)
            if rand not in self.random_train_negative:
                self.random_train_negative.append(rand)
            if len(self.random_train_negative) == Constants.sampling_size:
                break
        
        while True:
            rand = random.choice(self.positive_nodes)
            if rand not in self.random_train_positive:
                self.random_train_positive.append(rand)
            if len(self.random_train_positive) == Constants.sampling_size:
                break
        
        train_path = Constants.ricc_dir_path / f"train_{turn}_prime.txt"
        with open(train_path, "w") as f_train:
            for i in range(Constants.sampling_size):
                f_train.write(f"{self.random_train_negative[i]} ")
            f_train.write("\n")

            for i in range(Constants.sampling_size):
                f_train.write(f"{self.random_train_positive[i]} ")
            f_train.write("\n")
    
    def trainset2prior(self):
        turn = 0
        num_negative = self.config.num_negative
        num_positive = self.config.num_positive
        num_unlabel = Constants.num_unlabel

        train_score_file_path = Constants.ricc_dir_path / f"prior_{turn}_prime.txt"
        with open(train_score_file_path, "w") as f_train_score:
            for i in range(0, num_negative + num_positive + num_unlabel):
                if str(i) in self.random_train_negative:
                    f_train_score.write(str(i) + " -{}\n".format(Constants.theta))
                elif str(i) in self.random_train_positive:
                    f_train_score.write(str(i) + " {}\n".format(Constants.theta))
                else:
                    f_train_score.write(str(i) + " 0\n")
    
    def check_FN_nodes(self):
        turn = 1
        num_negative = self.config.num_negative
        num_positive = self.config.num_positive
        num_unlabel = Constants.num_unlabel


        f_performance = open(Constants.ricc_dir_path / f"summary.txt", "w")
        f_score = open(Constants.ricc_dir_path / self.post_filename, "r")
        f_target = open(Constants.ricc_dir_path / "target_close.txt", "r")

        # read target nodes
        target_detect = 0
        target_list = []
        targets = f_target.readline()
        targets = targets.split()
        target_num = len(targets)
        for target in targets:
            target_list.append(int(target))
        
        # read original trainset
        f_train_ori = open(Constants.ricc_dir_path / "train_0.txt", "r")
        train_ori = []
        lines = f_train_ori.read().splitlines()
        for line in lines:
            line = line.split()
            train_ori.extend(line)
        
        # compute the posterior score after defense in this turn
        score_list = []
        score_list_no_train = []
        scores = f_score.read().splitlines()

        for (idx, score) in enumerate(scores):
            score = score.split()
            score_list.append(score[1])
            if str(idx) not in train_ori:
                score_list_no_train.append(float(score[1]))
        
        # find the FN nodes
        turn = 0
        if turn == 0:
            f1 = open(Constants.ricc_dir_path / f"post_sybilscar_before_attack({Constants.weight}_{Constants.iteration}).txt", "r")
            f2 = open(Constants.ricc_dir_path / f"post_sybilscar_0_evaluation.txt", "r")

            lines1 = f1.read().splitlines()
            lines2 = f2.read().splitlines()

            for line1, line2 in zip(lines1, lines2):
                line1 = line1.split()
                line2 = line2.split()

                if float(line1[1]) > Constants.threshold > float(line2[1]):
                    if int(line1[0]) >= num_negative:
                        Constants.FN_nodes.append(line1[0])
            
            f1.close()
            f2.close()
        
        # compute the error rates (FN rate)
        error = 0
        for node in Constants.FN_nodes:
            if float(score_list[int(node)]) < Constants.threshold:
                error += 1
        
        for node in target_list:
            if float(score_list[int(node)]) > Constants.threshold:
                target_detect += 1
        
        # save initial target detection
        if turn == 0:
            pass

        y_true = [0] * (num_negative - 100) + [1] * (num_positive - 100)
        roc_auc = roc_auc_score(y_true, score_list_no_train)

        # print and save the performance of the RICC
        msg = "FN rate : [{}/{} ({:.0f}%)]\t AUC : {:.4f}\n".format(
            target_detect, target_num, 100. * target_detect / target_num, roc_auc
        )
        print(msg)
        f_performance.write(msg)

        f_performance.close()
        f_score.close()
        f_target.close()
        f_train_ori.close()
        




        
    
    ###########################
    ### GETTERS ###############
    ###########################
    def get_graph_list(self):
        return self.graph
    
    def get_node_num(self):
        return self.node_num
    
    def get_prior_list(self):
        return self.prior_list
    
    ###########################
    ### PRINT UTILS ###########
    ###########################
    def print_graph_info(self):
        print("Graph Info")
        print("Graph Type: ", self.config.graph_type)
        print("Attack Type: ", self.config.attack_type)
        print("Graph File Path: ", self.graph_file_path)
        print("Node Count: ", self.node_num)

    def print_graph(self):
        for i in range(len(self.graph)):
            print(i, self.graph[i])


    ###########################
    ### SybilSCAR UTILS #######
    ###########################

    def run_lbp(self, post_list, withBuffer=False):
        """ 
        run LinLBP one iteration with 'graph_list' & 'prior_list'
        :return:
        """

        
        graph_list = self.graph
        node_num = self.node_num
        prior_list = self.prior_list

        wei = Constants.weight
        if withBuffer:
            wei *= Constants.buffer

        next_post_list = [0.0] * node_num

        # graph
        # 1: [4, 2, 3]
        # 2: [3, 4,5 ]
        for nei_list in enumerate(graph_list):
            #score_tmp = float(post_list[nei_list[0]])
            score_tmp = 0
            prior_score = prior_list[nei_list[0]]

            for nei in nei_list[1]:
                post_score = post_list[int(nei)]
                score_tmp += (2.0 * wei * post_score)

            score_tmp += prior_score

            if score_tmp > 0.5:
                score_tmp = 0.5
            if score_tmp < -0.5:
                score_tmp = -0.5

            next_post_list[nei_list[0]] = score_tmp

        post_list = copy.deepcopy(next_post_list)

        return post_list