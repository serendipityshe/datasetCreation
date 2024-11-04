from utils import dataEnhancement

A = dataEnhancement.aug('/media/hz/新加卷/data/1/img/train',
                        '/media/hz/新加卷/data/1/label/train',
                        '/media/hz/新加卷/data/1/img/train_c',
                        '/media/hz/新加卷/data/1/label/train_c')


A.Rotate(angle=45)
A.RandomResize()
A.MirrorHorizon()