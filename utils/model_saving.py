import os
import numpy as np
import pickle
import csv

from datetime import date
today = date.today()


class save_info(object):
    def __init__(self, assets_dir, exp_num, exp_name, env_name):
        self.assets_dir = assets_dir
        self.experiment_num = 'exp-{}'.format(exp_num)
        # common path
        self.saving_path = 'learned_models/{}/{}-{}-{}'.format(exp_name, today, self.experiment_num, env_name)

    def create_all_paths(self):
        """create all the paths to save learned models/data"""

        # model for best training reward saving path
        self.model_training_saving_path = os.path.join(self.assets_dir, self.saving_path, 'best_training_model.p')
        if not os.path.exists(os.path.dirname(self.model_training_saving_path)):
            try:
                os.makedirs(os.path.dirname(self.model_training_saving_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        # model saving path
        self.model_saving_path = os.path.join(self.assets_dir, self.saving_path, 'model.p')
        if not os.path.exists(os.path.dirname(self.model_saving_path)):
            try:
                os.makedirs(os.path.dirname(self.model_saving_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        # intermediate model saving path
        self.intermediate_model_saving_path = os.path.join(self.assets_dir, self.saving_path, 'intermediate_model/')
        if not os.path.exists(os.path.dirname(self.intermediate_model_saving_path)):
            try:
                os.makedirs(os.path.dirname(self.intermediate_model_saving_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        # avg reward saving path
        self.avg_reward_saving_path = os.path.join(self.assets_dir, self.saving_path, 'avg_reward.p')

        # num of steps saving path
        self.num_of_steps_saving_path = os.path.join(self.assets_dir, self.saving_path, 'num_of_steps.p')

        # num of episodes saving path
        self.num_of_episodes_saving_path = os.path.join(self.assets_dir, self.saving_path, 'num_of_episodes.p')

        # total num of episodes saving path
        self.total_num_of_episodes_saving_path = os.path.join(self.assets_dir, self.saving_path,
                                                              'total_num_of_episodes.p')

        # total steps num saving path
        self.total_num_of_steps_saving_path = os.path.join(self.assets_dir, self.saving_path, 'total_num_of_steps.p')

        # total num of steps saving path
        self.rewards_std_saving_path = os.path.join(self.assets_dir, self.saving_path, 'rewards_std.p')

        # total num of steps saving path
        self.env_avg_reward_saving_path = os.path.join(self.assets_dir, self.saving_path, 'env_avg_reward.p')

        # v loss saving path
        self.true_v_loss_list_saving_path = os.path.join(self.assets_dir, self.saving_path, 'true_v_loss_list.p')

        # decayed v loss saving path
        self.decayed_v_loss_list_saving_path = os.path.join(self.assets_dir, self.saving_path, 'decayed_v_loss_list.p')

        # p loss saving path
        self.p_loss_list_saving_path = os.path.join(self.assets_dir, self.saving_path, 'p_loss_list.p')

        # evaluation average reward
        self.eval_avg_R_saving_path = os.path.join(self.assets_dir, self.saving_path, 'eval_avg_R.p')

        # evaluation average reward std
        self.eval_avg_R_std_saving_path = os.path.join(self.assets_dir, self.saving_path, 'eval_avg_R_std.p')

    def dump_lists(self, num_of_steps, num_of_episodes, total_num_episodes, total_num_steps, rewards_std,
                   env_avg_reward, v_loss_list, p_loss_list, eval_avg_reward, eval_avg_reward_std):

        # dump expert_avg_reward, num_of_steps, num_of_episodes
        pickle.dump(num_of_steps,
                    open(os.path.join(self.assets_dir, self.num_of_steps_saving_path), 'wb'))
        pickle.dump(num_of_episodes,
                    open(os.path.join(self.assets_dir, self.num_of_episodes_saving_path), 'wb'))
        pickle.dump(total_num_episodes,
                    open(os.path.join(self.assets_dir, self.total_num_of_episodes_saving_path), 'wb'))
        pickle.dump(total_num_steps,
                    open(os.path.join(self.assets_dir, self.total_num_of_steps_saving_path), 'wb'))
        pickle.dump(rewards_std,
                    open(os.path.join(self.assets_dir, self.rewards_std_saving_path), 'wb'))
        pickle.dump(env_avg_reward,
                    open(os.path.join(self.assets_dir, self.env_avg_reward_saving_path), 'wb'))
        pickle.dump(v_loss_list,
                    open(os.path.join(self.assets_dir, self.true_v_loss_list_saving_path), 'wb'))
        pickle.dump(p_loss_list,
                    open(os.path.join(self.assets_dir, self.p_loss_list_saving_path), 'wb'))
        pickle.dump(eval_avg_reward,
                    open(os.path.join(self.assets_dir, self.eval_avg_R_saving_path), 'wb'))
        pickle.dump(eval_avg_reward_std,
                    open(os.path.join(self.assets_dir, self.eval_avg_R_std_saving_path), 'wb'))

    def save_models(self, policy_net, value_net, running_state):
        pickle.dump((policy_net, value_net, running_state),
                    open(os.path.join(self.assets_dir, self.model_saving_path), 'wb'))

    def save_training_models(self, policy_net, running_state):
        pickle.dump((policy_net, running_state),
                    open(os.path.join(self.assets_dir, self.model_training_saving_path), 'wb'))

    def save_info(self, reward, training_reward, constrain_value_list, info_list, time_list):

        path = os.path.join(self.assets_dir, self.saving_path, 'losses.csv')

        with open(path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write data
            writer.writerow(reward)
            writer.writerow(constrain_value_list)

            info_list = np.array(info_list).T
            for info in info_list:
                writer.writerow(info)

            writer.writerow(time_list)
            writer.writerow(training_reward)

    def save_parameters(self, d_k, max_kl, cg_iter, max_iter, batch_size):

        path = os.path.join(self.assets_dir, self.saving_path, 'parameters.csv')
        field_names = ['max iterations', 'constraint limit', 'trust region size (max_kl)', 'CG iterations',
                       'batch size']
        parameters = {'max iterations': max_iter, 'constraint limit': d_k, 'trust region size (max_kl)': max_kl,
                      'CG iterations': cg_iter, 'batch size': batch_size}

        with open(path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=field_names)

            # write data
            writer.writerow(parameters)

    def save_intermediate_models(self, policy_net, value_net, running_state, i_iter):
        pickle.dump((policy_net, value_net, running_state), open(
            os.path.join(self.assets_dir, self.intermediate_model_saving_path, 'model_iter_{}.p'.format(i_iter + 1)),
            'wb'))
