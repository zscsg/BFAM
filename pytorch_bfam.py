#coding: utf-8

import numpy as np
from tqdm import tqdm
import time


class BruteforceAttackMethod(object):
    """
    暴力攻击方法，pytorch 版本
    """
    def __init__(self, model, target=None):
        """
        :param model: 目标模型，默认为sklearn model，具备predict_proba方法
        :param target: 目标标签
        """
        self.model = model
        self.target = target

    def generate(self, x, alpha=1., p_index=[]):
        """
        根据输入的原始x，生成对抗样本
        :param x: 输入的原始样本
        :param alhpa: 扰动强度
        :param p_index: 可扰动特征，特征的索引数组

        :return 对抗样本
        """

        adv_x = []
        ox = np.copy(x)
        n = 0

        if len(p_index) == 0:
            p_index = list(range(x.shape[-1]))

        preds = self.model.predict(ox)

        indexes, = np.where(preds == self.target)

        if len(indexes) != 0:
            adv_x.extend(ox[indexes, :])
            ox = np.delete(ox, indexes, axis=0)

        if len(ox) == 0:
            return np.array(adv_x), n

        for i in tqdm(range(len(p_index))):

            f0 = self.model.predict_proba(ox)

            ov = np.copy(ox[:, p_index[i]])

            ox[:, p_index[i]] += alpha
            ox[:, p_index[i]] = np.minimum(ox[:, p_index[i]], 1)

            f1 = self.model.predict_proba(ox)
            n += 1

            cancel = (f1[:, self.target] - f0[:, self.target]) < 0

            ox[cancel, p_index[i]] = ov[cancel]

            preds = self.model.predict(ox)

            indexes, = np.where(preds == self.target)

            if len(indexes) != 0:
                adv_x.extend(ox[indexes, :])
                ox = np.delete(ox, indexes, 0)

            if len(ox) == 0:
                print("All examples are transformed into adversarial exmaples!")
                break

        if len(ox) != 0:
            print("Not all the original inputs have been transformed!")
            adv_x.extend(ox)

        return np.array(adv_x), n
