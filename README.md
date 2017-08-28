# clevr_task9
Relation Networks (https://arxiv.org/abs/1706.01427) for CLEVR implemented in PyTorch

To train:

1. Download and extract CLEVR v1.0 dataset from http://cs.stanford.edu/people/jcjohns/clevr/

2. Preprocessing question data
'''
python preprocess.py [CLEVR directory]
'''
e.g.
'''
python preprocess.py /home/zhangxifan/CLEVR_v1.0
'''

3. Run segmentation
'''
python felzen.py [CLEVR directory/images] train
python felzen.py [CLEVR directory/images] valid
'''
e.g.
'''
python felzen.py /home/zhangxifan/CLEVR_v1.0/images train
python felzen.py /home/zhangxifan/CLEVR_v1.0/images valid
'''

4. Run train.py
'''
python train.py --batch-size=[batchsize] --is-multi-gpu=[True/False] --data-dir=[CLEVR directory] --segs-dir=[CLEVR directory/images]
'''
e.g.
'''
python train.py --batch-size=256 --is-multi-gpu=True --data-dir=/home/zhangxifan/CLEVR_v1.0 
		--segs-dir=/home/zhangxifan/CLEVR_v1.0/images
'''
