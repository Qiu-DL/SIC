"""
Forked from SCAN (https://github.com/wvangansbeke/Unsupervised-Classification).
"""
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'cifar-10', 'stl-10', 'cifar-20', 'imagenet_dog', 'imagenet_tiny'}
        assert(database in db_names)

        if database == 'cifar-10':
            return '/dataset/cifar-10'
        elif database == 'stl-10':
            return '/dataset/stl-10'
        elif database == 'cifar-20':
            return '/dataset/cifar-20/'
        elif database == 'imagenet_dog':
            return '/dataset/imagenet_dog'
        elif database == 'imagenet_tiny':
            return '/dataset/imagenet_tiny'
        else:
            raise NotImplementedError
